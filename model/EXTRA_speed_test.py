import pandas as pd
import numpy as np
import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("data/3_processed/processed_cpt_mwd_with_new_features.csv")

# 2. Filter out rows with implausible values
df = df[df['fs'] >= 0]

# 3. Start timer for feature engineering
start_feat_eng = time.time()

# 4. Feature engineering (same as before)
df['g_total'] = np.sqrt(df['G_force_x']**2 + df['G_force_y']**2 + df['G_force_z']**2)
df['pull_diff'] = df['pulldown_pressure'] - df['pullup_pressure']
df['torque_per_rotation'] = df['torque'] / df['rotation_speed'].replace(0, np.nan)
df['energy_input'] = df['sonic_pressure'] * df['sonic_frequency']
df['mechanical_stress'] = df['pulldown_pressure'] * df['torque']

# 5. Clean dataset (handle infs/NaNs)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

end_feat_eng = time.time()
print(f" Feature engineering took: {end_feat_eng - start_feat_eng:.4f} seconds")

# 6. Define features and target
features = ["air_temperature", "operation_mode", "sonic_pressure", "rotation_pressure",
            "pullup_pressure", "pulldown_pressure", "inclination_x", "inclination_y",
            "flushing_pressure", "flushing_debit", "sonic_speed", "rotation_speed",
            "penetration_speed", "pull_diff", "torque_per_rotation", "g_total",
            "energy_input", "mechanical_stress", "torque_to_penetration"
]

target = 'qc'
X = df[features]
y = df[target]

# 7. Train/test split (just to fit a model, not for evaluation here)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train models
rf_model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.2, random_state=42)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# 9. Measure inference time
start_rf = time.time()
_ = rf_model.predict(X_test)
end_rf = time.time()

start_xgb = time.time()
_ = xgb_model.predict(X_test)
end_xgb = time.time()

print(f"⚡ Random Forest prediction time: {end_rf - start_rf:.6f} seconds for {len(X_test)} rows")
print(f"⚡ XGBoost prediction time:       {end_xgb - start_xgb:.6f} seconds for {len(X_test)} rows")
