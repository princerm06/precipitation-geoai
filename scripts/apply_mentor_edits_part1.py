import json
from pathlib import Path

NB = Path("UNG_PIDF_Model.ipynb")

with NB.open("r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb.get("cells", [])

def text(cell):
    return "".join(cell.get("source", []))

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip().split("\n")],
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip().split("\n")],
    }

# Idempotency guards
all_text = "\n".join(text(c) for c in cells)

# EDIT 1: dataset + spatial-block accounting
marker1 = "## Dataset and Spatial-Block Accounting"
if marker1 not in all_text:
    split_idx = None
    for i, cell in enumerate(cells):
        t = text(cell)
        if "GroupShuffleSplit" in t and "train_idx" in t and "test_idx" in t:
            split_idx = i
    if split_idx is None:
        for i, cell in enumerate(cells):
            t = text(cell)
            if "train_idx" in t and "test_idx" in t and "X_train_pidf" in t:
                split_idx = i
    if split_idx is None:
        raise RuntimeError("Could not locate PIDF spatial train/test split cell.")

    stats_md = md_cell("""## Dataset and Spatial-Block Accounting

To make the spatial validation setup explicit, the following cell reports the size of the full PIDF dataset, the 250,000-point modeling sample, the train/test split, and the number of unique 0.25° spatial blocks represented in each subset. The test set remains spatially disjoint from the training set.""")

    stats_code = code_cell("""print("Full PIDF data points:", len(pidf))
print("Sampled data points:", len(pidf_sample))
print("Training data points:", len(X_train_pidf))
print("Test data points:", len(X_test_pidf))
print("Total unique spatial blocks:", pidf["spatial_block"].nunique())
print("Sample unique spatial blocks:", pidf_sample["spatial_block"].nunique())
print("Training spatial blocks:", pidf_sample.iloc[train_idx]["spatial_block"].nunique())
print("Test spatial blocks:", pidf_sample.iloc[test_idx]["spatial_block"].nunique())""")

    cells[split_idx + 1:split_idx + 1] = [stats_md, stats_code]

# Refresh text after edit 1
all_text = "\n".join(text(c) for c in cells)

# EDIT 2: MLP learning curves
marker2 = "## MLP Learning Curves"
if marker2 not in all_text:
    fit_idx = None
    for i, cell in enumerate(cells):
        t = text(cell)
        if "mlp_model.fit" in t or ("history" in t and ".fit(" in t and "X_train_mlp" in t):
            fit_idx = i
            break
    if fit_idx is None:
        raise RuntimeError("Could not locate MLP training cell.")

    curves_md = md_cell("""## MLP Learning Curves

Because this is a regression problem, classification accuracy is not an appropriate metric. Instead, the training history is visualized with two separate curves: mean squared error (loss) and mean absolute error (MAE) for training versus validation. These plots help diagnose underfitting, overfitting, and training stability before hyperparameter optimization.""")

    curves_code = code_cell("""import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("MLP Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["mae"], label="Training MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE (inches)")
plt.title("MLP Training vs Validation MAE")
plt.legend()
plt.grid(alpha=0.3)
plt.show()""")

    cells[fit_idx + 1:fit_idx + 1] = [curves_md, curves_code]

nb["cells"] = cells

with NB.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print("Applied mentor-requested edits part 1:")
print("- Dataset/spatial-block accounting")
print("- MLP loss and MAE learning curves")
