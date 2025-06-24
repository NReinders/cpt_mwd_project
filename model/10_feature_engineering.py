import pandas as pd
import numpy as np

# === Stap 1: Inlezen en feature engineering ===
df = pd.read_csv("data/3_processed/processed_cpt_mwd_with_outlier_flags.csv")

df['pull_diff'] = df['pullup_pressure'] - df['pulldown_pressure']
df['torque_per_rotation'] = df['torque'] / (df['rotation_speed'] + 1e-6)
df['g_total'] = np.sqrt(df['G_force_x']**2 + df['G_force_y']**2 + df['G_force_z']**2)
df['energy_input'] = df['sonic_pressure'] * df['sonic_speed']
df['mechanical_stress'] = df['torque'] * df['rotation_pressure']
df['torque_to_penetration'] = df['torque'] / (df['penetration_speed'] + 1e-6)

# === Stap 2: Nieuwe features toevoegen en opslaan in csv ===
output_path = "data/3_processed/processed_cpt_mwd_with_new_features.csv"
df.to_csv(output_path, index=False)
print(f"Bestand opgeslagen als: {output_path}")