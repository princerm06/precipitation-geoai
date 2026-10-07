import builtins
import json
import symtable

SRC = "UNG_PIDF_Model.ipynb"
DST = "UNG_PIDF_Model_ci.ipynb"

ALLOWED_PREDEFINED = set(dir(builtins)) | {"get_ipython", "display", "__name__"}

TARGET_MARKERS = (
    "MLP Training vs Validation Loss",
    "Spatial Train/Validation Split for Hyperparameter Tuning",
    "study.optimize(objective",
    "best_params = study.best_params",
    "Selected training epochs:",
    "Tuned MLP Spatial Test Metrics",
    "comparison_models = pd.DataFrame",
)

REQUIRED_MARKERS = (
    "study.optimize(objective",
    "best_params = study.best_params",
    "Tuned MLP Spatial Test Metrics",
)

LEGACY_PATTERNS = (
    "dem_tiles",
    "_elev",
    "feature_importance",
    "xgboost",
    "xgb",
)


def source_text(cell):
    return "".join(cell.get("source", []))


def clean_for_ast(code):
    out = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            out.append("# CI ignored IPython magic")
        else:
            out.append(line)
    return "\n".join(out)


def sanitize(code):
    kept = []
    for line in code.splitlines():
        if (
            "from google.colab import files" in line
            or "files.upload()" in line
            or "files.download(" in line
            or "drive.mount(" in line
        ):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def symbols(code):
    cleaned = clean_for_ast(code)
    try:
        table = symtable.symtable(cleaned, "<cell>", "exec")
    except SyntaxError:
        return set(), set()

    defs = set()
    uses = set()

    for symbol in table.get_symbols():
        name = symbol.get_name()
        if symbol.is_imported() or symbol.is_assigned() or symbol.is_namespace():
            defs.add(name)
        if symbol.is_referenced():
            uses.add(name)

    def add_global_refs(t):
        for child in t.get_children():
            for symbol in child.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    uses.add(symbol.get_name())
            add_global_refs(child)

    add_global_refs(table)
    return defs, uses


def is_legacy(code):
    lower = code.lower()
    if any(p in lower for p in LEGACY_PATTERNS):
        return True
    if 'open("data.txt"' in code or "open('data.txt'" in code:
        return True
    if "elevation" in lower and "pidf" not in lower:
        return True
    return False


with open(SRC, "r", encoding="utf-8") as f:
    original_nb = json.load(f)

cells = original_nb.get("cells", [])
meta = {}

for i, cell in enumerate(cells):
    if cell.get("cell_type") != "code":
        continue
    original = source_text(cell)
    code = sanitize(original)

    # Keep the source notebook untouched while allowing callers to choose the
    # execution budget. GitHub CI defaults to 50k rows / 3 trials; Colab can
    # request the full 250k rows / 25 trials through environment variables.
    code = code.replace(
        "pidf.sample(n=250000, random_state=42)",
        f"pidf.sample(n={CI_SAMPLE_N}, random_state=42)",
    )
    code = code.replace(
        "study.optimize(objective, n_trials=25)",
        f"study.optimize(objective, n_trials={CI_OPTUNA_TRIALS})",
    )

    # In CI the NOAA rasters are downloaded deterministically by the workflow,
    # so replace any Colab/upload-derived ASC discovery with a recursive glob.
    # This prevents asc_files_pidf from being defined as an empty list.
    probe_defs, _ = symbols(code)
    if "asc_files_pidf" in probe_defs:
        code = (
            "import glob\n"
            "asc_files_pidf = sorted(glob.glob('se*yr*a/se*yr*a.asc'))\n"
            "print('CI ASC files:', len(asc_files_pidf))\n"
            "assert len(asc_files_pidf) == 28, "
            "'Expected 28 NOAA PIDF ASC rasters'\n"
        )

    defs, uses = symbols(code)
    meta[i] = {
        "original": original,
        "code": code,
        "defs": defs,
        "uses": uses,
        "legacy": is_legacy(original),
    }

target_indices = [
    i for i, m in meta.items()
    if any(marker in m["original"] for marker in TARGET_MARKERS)
]
if not target_indices:
    raise RuntimeError("No current PIDF/MLP/Optuna target cells found.")

missing_markers = [
    marker for marker in REQUIRED_MARKERS
    if not any(marker in m["original"] for m in meta.values())
]
if missing_markers:
    raise RuntimeError(
        "Source notebook is missing required current-experiment cells: "
        + ", ".join(missing_markers)
    )

# Build a focused slice starting where the full PIDF table is first created.
# This intentionally excludes the earlier historical RF/elevation experiments
# that were causing GitHub runner cancellations.
pidf_start_candidates = [
    i for i, m in meta.items()
    if "pidf" in m["defs"]
    and i < max(target_indices)
    and not m["legacy"]
]
if not pidf_start_candidates:
    raise RuntimeError("Could not find the upstream cell that defines pidf.")

pidf_start = max(i for i in pidf_start_candidates if i < min(target_indices))
end_idx = max(target_indices)

selected = {
    i for i in range(pidf_start, end_idx + 1)
    if i in meta and not meta[i]["legacy"]
}

# Recursively pull in definitions that the focused slice needs from earlier
# cells (imports, helper functions, Georgia boundary/data loaders, etc.).
changed = True
while changed:
    changed = False
    needed = set()
    defined_in_slice = set(ALLOWED_PREDEFINED)

    for i in sorted(selected):
        needed.update(
            name for name in meta[i]["uses"]
            if name not in defined_in_slice and name not in meta[i]["defs"]
        )
        defined_in_slice.update(meta[i]["defs"])

    for name in sorted(needed):
        candidates = [
            j for j, m in meta.items()
            if j < pidf_start
            and name in m["defs"]
            and not m["legacy"]
        ]
        if candidates:
            j = max(candidates)
            if j not in selected:
                selected.add(j)
                changed = True

# Final static validation in execution order.
defined = set(ALLOWED_PREDEFINED)
unresolved = []
for i in sorted(selected):
    m = meta[i]
    missing = sorted(
        name for name in m["uses"]
        if name not in defined and name not in m["defs"] and name not in ALLOWED_PREDEFINED
    )
    if missing:
        unresolved.append((i, missing, m["code"][:220]))
    defined.update(m["defs"])

if unresolved:
    report = "\n".join(
        f"cell {idx}: unresolved={names} :: {snippet!r}"
        for idx, names, snippet in unresolved
    )
    raise RuntimeError(
        "Focused CI notebook still has unresolved dependencies.\n" + report
    )

focused_cells = []
for i in sorted(selected):
    src = meta[i]["code"]
    if not src.strip():
        continue
    focused_cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"source_cell_index": i},
        "outputs": [],
        "source": [line + "\n" for line in src.rstrip("\n").split("\n")],
    })

focused_nb = {
    "cells": focused_cells,
    "metadata": original_nb.get("metadata", {}),
    "nbformat": original_nb.get("nbformat", 4),
    "nbformat_minor": original_nb.get("nbformat_minor", 5),
}

with open(DST, "w", encoding="utf-8") as f:
    json.dump(focused_nb, f)

print(f"Built focused notebook: {DST}")
print(f"Original code cells: {len(meta)}")
print(f"Focused code cells: {len(focused_cells)}")
print(f"PIDF slice start cell: {pidf_start}")
print(f"Final target cell: {end_idx}")
print("Focused dependency preflight: PASS")
print("Required Optuna/tuned-MLP markers: PASS")
