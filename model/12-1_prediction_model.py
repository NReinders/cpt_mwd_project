import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data inladen
df = pd.read_csv("data/3_processed/processed_cpt_mwd_with_new_features.csv")
df = df[df["fs"] >= 0].copy()

# Configuraties per target
configs = [
    {
        "name": "qc",
        "target": "qc",
        "true_target": "qc",
        "features": ["sonic_speed", "g_total", "flushing_debit", "air_temperature"],
        "inverse_transform": False
    },
    {
        "name": "fs",
        "target": "fs",
        "true_target": "fs",
        "features": [
            "flushing_debit", "g_total", "sonic_speed", "pullup_pressure",
            "sonic_pressure", "air_temperature", "rotation_pressure", "energy_input"
        ],
        "inverse_transform": False
    },
    {
        "name": "fs_sqrt",
        "target": "fs_sqrt",
        "true_target": "fs",
        "features": [
            "sonic_speed", "g_total", "flushing_pressure", "rotation_pressure",
            "pullup_pressure", "pulldown_pressure", "flushing_debit"
        ],
        "inverse_transform": True
    }
]

fold_results = []

for config in configs:
    print(f"\n===== Evaluatie voor target: {config['name']} =====")

    X = df[config["features"]].copy()
    y = df[config["target"]]
    y_true_all = df[config["true_target"]]
    groups = df["number_of_rods"]

    logo = LeaveOneGroupOut()
    fold = 1

    rf_preds_all = np.zeros_like(y_true_all, dtype=float)
    xgb_preds_all = np.zeros_like(y_true_all, dtype=float)

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        y_test_true = y_true_all.iloc[test_idx]

        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        if config["inverse_transform"]:
            rf_pred = rf_pred ** 2
        rf_preds_all[test_idx] = rf_pred

        # XGBoost
        xgb_model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        if config["inverse_transform"]:
            xgb_pred = xgb_pred ** 2
        xgb_preds_all[test_idx] = xgb_pred

        # Resultaten opslaan
        rf_result = {
            "target": config["name"], "fold": fold, "model": "RF",
            "mae": mean_absolute_error(y_test_true, rf_pred),
            "rmse": np.sqrt(mean_squared_error(y_test_true, rf_pred)),
            "r2": r2_score(y_test_true, rf_pred),
            "y_true": y_test_true.values, "y_pred": rf_pred
        }
        xgb_result = {
            "target": config["name"], "fold": fold, "model": "XGB",
            "mae": mean_absolute_error(y_test_true, xgb_pred),
            "rmse": np.sqrt(mean_squared_error(y_test_true, xgb_pred)),
            "r2": r2_score(y_test_true, xgb_pred),
            "y_true": y_test_true.values, "y_pred": xgb_pred
        }
        fold_results.append(rf_result)
        fold_results.append(xgb_result)

        # Print evaluatie per fold
        print(f"Fold {fold}")
        print(f"  RF  - MAE: {rf_result['mae']:.2f}, RMSE: {rf_result['rmse']:.2f}, R2: {rf_result['r2']:.2f}")
        print(f"  XGB - MAE: {xgb_result['mae']:.2f}, RMSE: {xgb_result['rmse']:.2f}, R2: {xgb_result['r2']:.2f}\n")

        fold += 1

    # Gemiddelde prestaties over alle folds
    rf_subset = [res for res in fold_results if res["target"] == config["name"] and res["model"] == "RF"]
    xgb_subset = [res for res in fold_results if res["target"] == config["name"] and res["model"] == "XGB"]

    rf_mae_scores = [res["mae"] for res in rf_subset]
    rf_rmse_scores = [res["rmse"] for res in rf_subset]
    rf_r2_scores = [res["r2"] for res in rf_subset]

    xgb_mae_scores = [res["mae"] for res in xgb_subset]
    xgb_rmse_scores = [res["rmse"] for res in xgb_subset]
    xgb_r2_scores = [res["r2"] for res in xgb_subset]

    print("=== Gemiddelde prestaties over alle folds ===")
    print(f"Random Forest - MAE: {np.mean(rf_mae_scores):.2f} ± {np.std(rf_mae_scores):.2f}")
    print(f"Random Forest - RMSE: {np.mean(rf_rmse_scores):.2f} ± {np.std(rf_rmse_scores):.2f}")
    print(f"Random Forest - R²: {np.mean(rf_r2_scores):.2f} ± {np.std(rf_r2_scores):.2f}")
    print(f"XGBoost       - MAE: {np.mean(xgb_mae_scores):.2f} ± {np.std(xgb_mae_scores):.2f}")
    print(f"XGBoost       - RMSE: {np.mean(xgb_rmse_scores):.2f} ± {np.std(xgb_rmse_scores):.2f}")
    print(f"XGBoost       - R²: {np.mean(xgb_r2_scores):.2f} ± {np.std(xgb_r2_scores):.2f}")

    # Extra: min en max van qc printen per target
    true_target = config["true_target"]
    print(f"\nMin en max van '{true_target}' zijn:")
    print(f"  Min: {df[true_target].min():.3f}")
    print(f"  Max: {df[true_target].max():.3f}")


    # Voeg voorspellingen toe aan dataframe
    df[f"predicted_{config['name']}_rf"] = rf_preds_all
    df[f"predicted_{config['name']}_xgb"] = xgb_preds_all

# Visualisaties
df_scores = pd.DataFrame(fold_results)

# Lineplots per metric
for metric in ["mae", "rmse", "r2"]:
    plt.figure(figsize=(10, 5))
    for target in df_scores["target"].unique():
        for model in ["RF", "XGB"]:
            subset = df_scores[(df_scores["target"] == target) & (df_scores["model"] == model)]
            plt.plot(subset["fold"], subset[metric], marker='o', label=f"{target} - {model}")
    plt.title(f"{metric.upper()} per fold")
    plt.xlabel("Fold")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Scatterplots y_true vs y_pred
for target in df_scores["target"].unique():
    plt.figure(figsize=(10, 5))
    for model in ["RF", "XGB"]:
        subset = df_scores[(df_scores["target"] == target) & (df_scores["model"] == model)]
        y_true_all = np.concatenate(subset["y_true"].values)
        y_pred_all = np.concatenate(subset["y_pred"].values)
        plt.scatter(y_true_all, y_pred_all, alpha=0.3, label=f"{target} - {model}")
    min_val = min(y_true_all.min(), y_pred_all.min())
    max_val = max(y_true_all.max(), y_pred_all.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title(f"y_true vs y_pred - {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Lijngrafieken (gesorteerd)
for target in df_scores["target"].unique():
    for model in ["RF", "XGB"]:
        subset = df_scores[(df_scores["target"] == target) & (df_scores["model"] == model)]
        y_true_all = np.concatenate(subset["y_true"].values)
        y_pred_all = np.concatenate(subset["y_pred"].values)
        sorted_idx = np.argsort(y_true_all)
        plt.figure(figsize=(12, 4))
        plt.plot(y_true_all[sorted_idx], label="Actual", linewidth=2)
        plt.plot(y_pred_all[sorted_idx], label="Predicted", linestyle="--")
        plt.title(f"Linechart - {target} ({model})")
        plt.xlabel("Observation (sorted)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# CSV met voorspellingen opslaan
df.to_csv("data/4_predictions/processed_cpt_mwd_with_predictions.csv", index=False)
print("Bestand opgeslagen als 'processed_cpt_mwd_with_predictions.csv'")
