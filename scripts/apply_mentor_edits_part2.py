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

all_text = "\n".join(text(c) for c in cells)

# Find insertion point immediately after the MLP learning-curve code cell added in part 1.
curve_md_idx = None
for i, cell in enumerate(cells):
    if "## MLP Learning Curves" in text(cell):
        curve_md_idx = i
        break

if curve_md_idx is None:
    raise RuntimeError("Could not locate '## MLP Learning Curves' section.")

insert_idx = curve_md_idx + 1
while insert_idx < len(cells) and cells[insert_idx].get("cell_type") != "code":
    insert_idx += 1
if insert_idx >= len(cells):
    raise RuntimeError("Could not locate learning-curve code cell.")
insert_idx += 1

new_cells = []

# EDIT 3: spatially disjoint train/validation split for hyperparameter tuning
marker_split = "## Spatial Train/Validation Split for Hyperparameter Tuning"
if marker_split not in all_text:
    new_cells += [
        md_cell("""## Spatial Train/Validation Split for Hyperparameter Tuning

Hyperparameters should be selected without using the final spatial test set. To preserve the geographic-validation design, the original training portion is split again by spatial block into an inner training set and validation set. The final test set remains untouched until the tuned model is selected."""),
        code_cell("""from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# Spatial-block labels corresponding only to the original training rows
groups_train_pidf = pidf_sample.iloc[train_idx]["spatial_block"].reset_index(drop=True)

# Work with reset indices so positional indices from GroupShuffleSplit align cleanly
X_train_opt = X_train_pidf.reset_index(drop=True)
y_train_opt = y_train_pidf.reset_index(drop=True)

gss_opt = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
inner_train_idx, val_idx = next(
    gss_opt.split(X_train_opt, y_train_opt, groups_train_pidf)
)

X_opt_train = X_train_opt.iloc[inner_train_idx]
X_opt_val = X_train_opt.iloc[val_idx]
y_opt_train = y_train_opt.iloc[inner_train_idx]
y_opt_val = y_train_opt.iloc[val_idx]

# Fit preprocessing only on the inner training subset to avoid validation leakage
optuna_scaler = StandardScaler()
X_opt_train_scaled = optuna_scaler.fit_transform(X_opt_train)
X_opt_val_scaled = optuna_scaler.transform(X_opt_val)

print("Optuna inner-training points:", len(X_opt_train))
print("Optuna validation points:", len(X_opt_val))
print("Optuna inner-training spatial blocks:", groups_train_pidf.iloc[inner_train_idx].nunique())
print("Optuna validation spatial blocks:", groups_train_pidf.iloc[val_idx].nunique())""")
    ]

# EDIT 4: Optuna MLP hyperparameter search
marker_optuna = "## Optuna Hyperparameter Optimization"
if marker_optuna not in all_text:
    new_cells += [
        md_cell("""## Optuna Hyperparameter Optimization

Optuna is used to tune the MLP on the spatially disjoint validation subset. The objective minimizes validation MAE, while early stopping limits unnecessary training. The final spatial test set is not used during optimization."""),
        code_cell("""!pip -q install optuna"""),
        code_cell("""import optuna
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def objective(trial):
    tf.keras.backend.clear_session()

    n_layers = trial.suggest_int("n_layers", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512, 1024])
    activation = trial.suggest_categorical("activation", ["relu", "elu"])

    model = keras.Sequential()
    model.add(layers.Input(shape=(X_opt_train_scaled.shape[1],)))

    for layer_idx in range(n_layers):
        units = trial.suggest_categorical(
            f"units_{layer_idx}",
            [32, 64, 128, 256]
        )
        model.add(layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_mae",
        patience=8,
        mode="min",
        restore_best_weights=True,
    )

    history_opt = model.fit(
        X_opt_train_scaled,
        y_opt_train,
        validation_data=(X_opt_val_scaled, y_opt_val),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    return min(history_opt.history["val_mae"])

study = optuna.create_study(
    direction="minimize",
    study_name="pidf_mlp_spatial_validation",
)

study.optimize(objective, n_trials=25)

print("Best validation MAE:", study.best_value)
print("Best hyperparameters:")
print(study.best_params)""")
    ]

if new_cells:
    cells[insert_idx:insert_idx] = new_cells
    nb["cells"] = cells
    with NB.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("Applied mentor-requested edits part 2:")
    print("- Spatial train/validation split for hyperparameter tuning")
    print("- Optuna MLP hyperparameter search")
else:
    print("Part 2 edits already present; no changes needed.")
