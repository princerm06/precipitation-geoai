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

# Insert after the Optuna search cell.
optuna_idx = None
for i, cell in enumerate(cells):
    t = text(cell)
    if "study.optimize(objective" in t:
        optuna_idx = i
        break

if optuna_idx is None:
    raise RuntimeError("Could not locate Optuna optimization cell.")

insert_idx = optuna_idx + 1
new_cells = []

# EDIT 5: rebuild/retrain the best Optuna model without using the final test set
marker_retrain = "## Retrain the Best Optuna MLP"
if marker_retrain not in all_text:
    new_cells += [
        md_cell("""## Retrain the Best Optuna MLP

The best hyperparameters selected by Optuna are rebuilt and trained again. Early stopping is first used only on the spatially disjoint inner validation set to determine a reasonable epoch count. The model is then retrained from scratch on the full original training set for that number of epochs. The final spatial test set remains untouched during model selection and retraining."""),
        code_cell("""import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

best_params = study.best_params

def build_best_mlp(params, input_dim):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for layer_idx in range(params["n_layers"]):
        model.add(
            layers.Dense(
                params[f"units_{layer_idx}"],
                activation=params["activation"],
            )
        )
        if params["dropout_rate"] > 0:
            model.add(layers.Dropout(params["dropout_rate"]))

    model.add(layers.Dense(1))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=params["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model

# Determine the training length using only the inner spatial validation split
tf.keras.backend.clear_session()
best_mlp_probe = build_best_mlp(
    best_params,
    X_opt_train_scaled.shape[1],
)

best_early_stop = keras.callbacks.EarlyStopping(
    monitor="val_mae",
    patience=8,
    mode="min",
    restore_best_weights=True,
)

best_history = best_mlp_probe.fit(
    X_opt_train_scaled,
    y_opt_train,
    validation_data=(X_opt_val_scaled, y_opt_val),
    epochs=100,
    batch_size=best_params["batch_size"],
    callbacks=[best_early_stop],
    verbose=0,
)

best_epoch = int(np.argmin(best_history.history["val_mae"]) + 1)
print("Selected training epochs:", best_epoch)

# Retrain from scratch on the full original training split
final_scaler = StandardScaler()
X_train_final = final_scaler.fit_transform(X_train_pidf)
X_test_final = final_scaler.transform(X_test_pidf)

tf.keras.backend.clear_session()
mlp_tuned = build_best_mlp(
    best_params,
    X_train_final.shape[1],
)

mlp_tuned.fit(
    X_train_final,
    y_train_pidf,
    epochs=best_epoch,
    batch_size=best_params["batch_size"],
    verbose=1,
)""")
    ]

# EDIT 6: final untouched spatial-test evaluation and direct comparison
marker_eval = "## Tuned MLP Final Spatial Test Evaluation"
if marker_eval not in all_text:
    new_cells += [
        md_cell("""## Tuned MLP Final Spatial Test Evaluation

After hyperparameter selection is complete, the tuned MLP is evaluated once on the untouched spatial test set. MAE, RMSE, and R² are reported and compared with the existing Random Forest and baseline MLP results."""),
        code_cell("""from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

tuned_pred = mlp_tuned.predict(X_test_final, verbose=0).ravel()

tuned_mae = mean_absolute_error(y_test_pidf, tuned_pred)
tuned_rmse = np.sqrt(mean_squared_error(y_test_pidf, tuned_pred))
tuned_r2 = r2_score(y_test_pidf, tuned_pred)

print("Tuned MLP Spatial Test Metrics")
print("MAE:", tuned_mae)
print("RMSE:", tuned_rmse)
print("R²:", tuned_r2)

comparison_models = pd.DataFrame({
    "Model": ["Random Forest", "Baseline MLP", "Tuned MLP"],
    "MAE": [0.06575968986647435, 0.09625851826663205, tuned_mae],
    "RMSE": [0.20451353960310295, 0.1787704209069349, tuned_rmse],
    "R²": [0.9913424680862872, 0.9933848256362094, tuned_r2],
})

comparison_models""")
    ]

if new_cells:
    cells[insert_idx:insert_idx] = new_cells
    nb["cells"] = cells
    with NB.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("Applied mentor-requested edits part 3:")
    print("- Retrain best Optuna MLP")
    print("- Final untouched spatial-test evaluation")
else:
    print("Part 3 edits already present; no changes needed.")
