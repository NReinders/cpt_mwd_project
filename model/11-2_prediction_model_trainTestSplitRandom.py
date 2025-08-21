import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data inladen
df = pd.read_csv("data/3_processed/processed_cpt_mwd_with_new_features.csv")

# Configuratie per target
configs = [
    {
        "name": "qc",
        "target": "qc",
        "features": [
            "air_temperature", "operation_mode", "sonic_pressure", "rotation_pressure",
            "pullup_pressure", "pulldown_pressure", "inclination_x", "inclination_y",
            "flushing_pressure", "flushing_debit", "sonic_speed", "rotation_speed",
            "penetration_speed", "pull_diff", "torque_per_rotation", "g_total",
            "energy_input", "mechanical_stress", "torque_to_penetration"
        ],
        "inverse_transform": False
    },
    {
        "name": "fs",
        "target": "fs",
        "features": [
            "air_temperature", "operation_mode", "sonic_pressure", "rotation_pressure",
            "pullup_pressure", "pulldown_pressure", "inclination_x", "inclination_y",
            "flushing_pressure", "flushing_debit", "sonic_speed", "rotation_speed",
            "penetration_speed", "pull_diff", "torque_per_rotation", "g_total",
            "energy_input", "mechanical_stress", "torque_to_penetration"
        ],
        "inverse_transform": False
    },
    {
        "name": "fs_sqrt",
        "target": "fs_sqrt",
        "true_target": "fs",
        "features": [
            "air_temperature", "operation_mode", "sonic_pressure", "rotation_pressure",
            "pullup_pressure", "pulldown_pressure", "inclination_x", "inclination_y",
            "flushing_pressure", "flushing_debit", "sonic_speed", "rotation_speed",
            "penetration_speed", "pull_diff", "torque_per_rotation", "g_total",
            "energy_input", "mechanical_stress", "torque_to_penetration"
        ],
        "inverse_transform": True
    }
]

# Resultaten per model/target
results = []

# Loop over targets
for config in configs:
    print(f"\n===== Evaluatie voor target: {config['name']} =====")

    target = config["target"]
    true_target = config.get("true_target", target)

    # ➕ Alleen rijen waar target geldig is
    valid_rows = df[target].notna()
    X = df.loc[valid_rows, config["features"]].copy()
    y = df.loc[valid_rows, target]
    y_true_full = df.loc[valid_rows, true_target].copy()

    # Train/test split voor evaluatie
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_test_true = y_true_full.loc[y_test.index].copy()

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    # Voorspelling op hele dataset
    rf_pred_all = rf_model.predict(X)
    xgb_pred_all = xgb_model.predict(X)

    # Voorspelling op test-set voor evaluatie
    rf_pred_test = rf_model.predict(X_test)
    xgb_pred_test = xgb_model.predict(X_test)

    # Terugtransformeren indien nodig
    if config["inverse_transform"]:
        rf_pred_all = rf_pred_all ** 2
        xgb_pred_all = xgb_pred_all ** 2
        rf_pred_test = rf_pred_test ** 2
        xgb_pred_test = xgb_pred_test ** 2

    # Voeg voorspellingen toe aan df
    df.loc[valid_rows, f"pred_{config['name']}_rf"] = rf_pred_all
    df.loc[valid_rows, f"pred_{config['name']}_xgb"] = xgb_pred_all

    # Evaluatie op test-set
    def evaluate(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {"model": name, "target": config["name"], "mae": mae, "rmse": rmse, "r2": r2}

    rf_result = evaluate("Random Forest", y_test_true, rf_pred_test)
    xgb_result = evaluate("XGBoost", y_test_true, xgb_pred_test)
    results.append(rf_result)
    results.append(xgb_result)

    # Print resultaten
    print(f"Min en max van '{true_target}': {y_true_full.min():.2f}, {y_true_full.max():.2f}")
    print(f"RF   - MAE: {rf_result['mae']:.2f}, RMSE: {rf_result['rmse']:.2f}, R²: {rf_result['r2']:.2f}")
    print(f"XGB  - MAE: {xgb_result['mae']:.2f}, RMSE: {xgb_result['rmse']:.2f}, R²: {xgb_result['r2']:.2f}")

    # Scatterplot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_true, rf_pred_test, alpha=0.3, label="RF")
    plt.scatter(y_test_true, xgb_pred_test, alpha=0.3, label="XGB")
    min_val = min(y_test_true.min(), rf_pred_test.min(), xgb_pred_test.min())
    max_val = max(y_test_true.max(), rf_pred_test.max(), xgb_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"y_true vs y_pred – {config['name']}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gesorteerde lijnplot
    sorted_idx = np.argsort(y_test_true)
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_true.iloc[sorted_idx].values, label="Actual", linewidth=2)
    plt.plot(rf_pred_test[sorted_idx], label="RF", linestyle="--")
    plt.plot(xgb_pred_test[sorted_idx], label="XGB", linestyle="--")
    plt.title(f"Linechart - {config['name']}")
    plt.xlabel("Observation (sorted)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Verrijkte dataset met voorspellingen opslaan
df.to_csv("data/4_predictions/processed_cpt_mwd_with_predictions_trainTestSplitRandom.csv", index=False)
print("\nVerrijkte dataset opgeslagen in 'processed_cpt_mwd_with_predictions_trainTestSplitRandom.csv'")