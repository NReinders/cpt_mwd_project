import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# === Inlezen dataset ===
df = pd.read_csv("data/3_processed/processed_cpt_mwd_with_new_features.csv")

# Features en targets instellen
targets = ['qc', 'fs', 'fs_sqrt']
features = df.drop(columns=[col for col in df.columns if '_outlier' in col or col in ['qc', 'fs', 'fs_sqrt',
                                                                                      'G_force_x', 'G_force_y', 'G_force_z',
                                                                                      'depth','depth_raw', 'number_of_rods',
                                                                                      'sonic_frequency', 'torque']])
features = features.select_dtypes(include=[np.number])
feature_names = features.columns.tolist()

# === Analyse per target ===
for target in targets:
    print(f"\nSHAP feature importance for target: {target}")

    # Data splitsen
    data = pd.concat([features, df[target]], axis=1).dropna()
    X = data[feature_names]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBRegressor(random_state=42)
    model.fit(X_train, y_train)

    # SHAP berekening
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # SHAP samenvattingstabel
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Importance': shap_importance
    }).sort_values(by='SHAP Importance', ascending=False)

    print("\nSHAP Summary Table:")
    print(shap_df)

    # Beeswarm plot
    shap.plots.beeswarm(shap_values, max_display=25, show=False)
    plt.title(f"SHAP Beeswarm Plot for target: {target}")
    plt.tight_layout()
    plt.show()

    # SHAP bar chart
    plt.figure(figsize=(10, 6))
    shap_df_sorted = shap_df.sort_values(by='SHAP Importance', ascending=True)
    plt.barh(shap_df_sorted['Feature'], shap_df_sorted['SHAP Importance'])
    plt.xlabel("Average SHAP-value (absolute impact)")
    plt.title(f"SHAP Bar Chart for target: {target}")
    plt.tight_layout()
    plt.show()

    # SHAP 80% cumulatieve selectie
    shap_df['Cumulative'] = shap_df['SHAP Importance'].cumsum()
    total_importance = shap_df['SHAP Importance'].sum()
    selected_features_df = shap_df[shap_df['Cumulative'] <= 0.8 * total_importance]
    selected_features = selected_features_df['Feature'].tolist()

    print(f"\nGeselecteerde features (≥80% SHAP impact) voor target '{target}':")
    print(selected_features)