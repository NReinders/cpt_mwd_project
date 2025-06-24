import pandas as pd

# Inlezen van het originele bestand
df = pd.read_csv("data/1_source/mwd_2b.csv")

# Kolommen verwijderen die volledig nul zijn
cols_all_zero = ['rotation_speed_duo', 'rotation_pressure_duo', 'torque_duo']
df = df.drop(columns=cols_all_zero, errors='ignore')

# Duplicaat datetime-kolom verwijderen
if 'date_time' in df.columns and 'datetime' in df.columns:
    if df['datetime'].equals(df['date_time']):
        df = df.drop(columns=['date_time'])

# Opslaan als cleaned versie
df.to_csv("data/2_clean/clean_mwd_2b.csv", index=False)

print(" clean_mwd_2b.csv is succesvol opgeslagen met de opgeschoonde kolommen.")
