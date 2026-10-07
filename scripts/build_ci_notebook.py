import ast
import builtins
import json
import re
import symtable

SRC = "UNG_PIDF_Model.ipynb"
DST = "UNG_PIDF_Model_ci.ipynb"

EXPLICIT_SKIP_PATTERNS = (
    "from google.colab import files",
    "files.upload()",
    "drive.mount(",
    "dem_tiles",
    "_elev",
    "feature_importance",
)

ALLOWED_PREDEFINED = set(dir(builtins)) | {
    "get_ipython",
    "display",
    "__name__",
}

REQUIRED_MARKERS = (
    'print("Full PIDF data points:"',
    'study.optimize(objective',
    'best_params = study.best_params',
    'Tuned MLP Spatial Test Metrics',
)


def source_text(cell):
    return "".join(cell.get("source", []))


def clean_for_ast(code):
    lines = []
    for line in code.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            lines.append("# CI ignored IPython magic")
        else:
            lines.append(line)
    return "\n".join(lines)


def sanitize_downloads(code):
    if "files.download(" not in code:
        return code
    kept = [
        line for line in code.splitlines()
        if "files.download(" not in line
    ]
    kept.append("print('CI: skipped Colab-only files.download call')")
    return "\n".join(kept) + "\n"


def module_symbols(code):
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
    return defs, uses


def is_explicit_legacy_skip(code):
    lower = code.lower()
    if any(pattern in code for pattern in EXPLICIT_SKIP_PATTERNS):
        return True
    if 'open("data.txt"' in code or "open('data.txt'" in code:
        return True
    if "elevation" in lower and "pidf" not in lower:
        return True
    if "xgb" in lower or "xgboost" in lower:
        return True
    return False


def replace_with_skip(cell, reason):
    cell["source"] = [f"print({reason!r})\n"]
    cell["execution_count"] = None
    cell["outputs"] = []


with open(SRC, "r", encoding="utf-8") as f:
    nb = json.load(f)

tainted = set()
defined = set(ALLOWED_PREDEFINED)
skipped = []
unresolved = []

for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue

    original = source_text(cell)
    code = sanitize_downloads(original)
    defs, uses = module_symbols(code)

    explicit = is_explicit_legacy_skip(original)
    inherited = sorted(uses & tainted)

    if explicit or inherited:
        tainted.update(defs)
        reason = (
            f"CI: skipped legacy/Colab cell {i}"
            if explicit
            else f"CI: skipped cell {i}; depends on skipped names: {', '.join(inherited)}"
        )
        replace_with_skip(cell, reason)
        skipped.append((i, reason, sorted(defs)))
        continue

    # Keep sanitized source if only files.download lines were removed.
    if code != original:
        cell["source"] = [line + "\n" for line in code.rstrip("\n").split("\n")]
        cell["execution_count"] = None
        cell["outputs"] = []

    missing = sorted(
        name for name in uses
        if name not in defined and name not in defs and name not in ALLOWED_PREDEFINED
    )
    if missing:
        unresolved.append((i, missing, code[:240]))

    defined.update(defs)

all_kept_code = "\n".join(
    source_text(cell)
    for cell in nb.get("cells", [])
    if cell.get("cell_type") == "code"
)

missing_markers = [m for m in REQUIRED_MARKERS if m not in all_kept_code]
if missing_markers:
    raise RuntimeError(
        "CI builder removed required current-experiment cells: "
        + ", ".join(missing_markers)
    )

if unresolved:
    report = "\n".join(
        f"cell {idx}: unresolved={names} :: {snippet!r}"
        for idx, names, snippet in unresolved
    )
    raise RuntimeError(
        "Static preflight found names used before definition in retained cells. "
        "Refusing to execute notebook until dependency graph is clean.\n" + report
    )

with open(DST, "w", encoding="utf-8") as f:
    json.dump(nb, f)

print(f"Built {DST}")
print(f"Skipped {len(skipped)} cells through explicit + transitive dependency filtering.")
for item in skipped:
    print(item)
print("Static undefined-name preflight: PASS")
print("Required current-experiment markers: PASS")
