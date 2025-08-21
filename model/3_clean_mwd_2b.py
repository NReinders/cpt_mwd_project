# 3_clean_mwd_2b.py
import pandas as pd
import matplotlib.pyplot as plt

SRC = "data/1_source/mwd_2b.csv"
DST = "data/2_clean/clean_mwd_2b.csv"

# Rod changes: gebruik de EINDwaarden van  tabel
# 1→2: 3.01, 2→3: 6.33, 3→4: 9.11, 4→5: 12.15, 5→6: 15.20, 6→7: 18.26, 7→8: 21.31, 8→9: 24.36
rod_change_depths_end = [3.01, 6.33, 9.11, 12.15, 15.20, 18.26, 21.31, 24.36]
BUFFER = 0.45  # meters na elke wissel

# -----------------------------
# 1) Inlezen
# -----------------------------
df = pd.read_csv(SRC)

# -----------------------------
# 2) Kolommen opruimen
# -----------------------------
# - kolommen met alleen nullen
cols_all_zero = ['rotation_speed_duo', 'rotation_pressure_duo', 'torque_duo']
df = df.drop(columns=[c for c in cols_all_zero if c in df.columns], errors='ignore')

# - dubbele datetime-kolom
if 'date_time' in df.columns and 'datetime' in df.columns:
    try:
        if df['datetime'].equals(df['date_time']):
            df = df.drop(columns=['date_time'])
    except Exception:
        # bij dtype verschillen blijft 'date_time' vaak toch redundant -> verwijder alsnog
        df = df.drop(columns=['date_time'])

# -----------------------------
# 3) Rod-nummering corrigeren
# -----------------------------
# Fout zat bij overgang 2→3; vanaf ~6.06 m (jouw begin van dat segment) schuift alles één rod omhoog.
if 'number_of_rods' in df.columns:
    df.loc[df['depth'] >= 6.06, 'number_of_rods'] = df.loc[df['depth'] >= 6.06, 'number_of_rods'] + 1

# -----------------------------
# 4) Verwijder buffers na rod changes
# -----------------------------
mask_keep = pd.Series(True, index=df.index)
for end_depth in rod_change_depths_end:
    start_buf = end_depth
    end_buf = end_depth + BUFFER
    mask_keep &= ~((df['depth'] >= start_buf) & (df['depth'] <= end_buf))

df_clean = df[mask_keep].reset_index(drop=True)

# -----------------------------
# 5) Opslaan
# -----------------------------
df_clean.to_csv(DST, index=False)
print(f"✔️ Cleaned MWD saved to: {DST}")
print(f"Rows before: {len(df)}, after: {len(df_clean)}, removed: {len(df) - len(df_clean)}")

# -----------------------------
# 6) Spike-check plots (na cleaning)
# -----------------------------
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
if 'sonic_speed' in df_clean.columns:
    plt.plot(df_clean['depth'], df_clean['sonic_speed'], color="blue")
plt.title('Sonic Speed vs Depth (cleaned)')
plt.ylabel('Sonic Speed')
plt.grid(True)

plt.subplot(3, 1, 2)
if 'rotation_speed' in df_clean.columns:
    plt.plot(df_clean['depth'], df_clean['rotation_speed'], color="green")
plt.title('Rotation Speed vs Depth (cleaned)')
plt.ylabel('Rotation Speed')
plt.grid(True)

plt.subplot(3, 1, 3)
if 'penetration_speed' in df_clean.columns:
    plt.plot(df_clean['depth'], df_clean['penetration_speed'], color="red")
plt.title('Penetration Speed vs Depth (cleaned)')
plt.xlabel('Depth (m)')
plt.ylabel('Penetration Speed')
plt.grid(True)

plt.tight_layout()
plt.savefig("data/5_figures/figure_2_mwd_lineplots_cleaned.png", dpi=200)
plt.show()
