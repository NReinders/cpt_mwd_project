import pandas as pd
import matplotlib.pyplot as plt

# Bestandspad aanpassen naar de dataset 
file_path = "data/1_source/mwd_2b.csv"  

# laat alle kolommen zien
pd.set_option('display.max_columns', None)

# Inlezen
df = pd.read_csv(file_path)

print(f"\n Dataset: {file_path}")
print("=" * 50)

# Globaal overzicht
print("\n Algemene info:")
df.info()

# Statistische samenvatting van numerieke kolommen
print("\n Beschrijvende statistieken (numeriek):")
print(df.describe())

# Statistieken voor niet-numerieke kolommen
object_cols = df.select_dtypes(include=["object"])
if not object_cols.empty:
    print("\n Voorbeeldwaarden van categorische/tekstkolommen:")
    for col in object_cols.columns:
        print(f"\nKolom: {col}")
        print(f"- Unieke waarden: {object_cols[col].nunique()}")
        print(f"- Voorbeeldwaarden: {object_cols[col].dropna().unique()[:5]}")


# Zorg dat depth oplopend is
df = df.sort_values("depth")

# Zoek dieptes waar het aantal rods verandert
rod_change_depths = df.loc[df['number_of_rods'].diff() != 0, 'depth'].values

# Zet de figuur op
plt.figure(figsize=(12, 8))

# Subplot 1: Sonic Speed
plt.subplot(3, 1, 1)
plt.plot(df['depth'], df['sonic_speed'], label='Sonic Speed', color='blue')
for d in rod_change_depths:
    plt.axvline(x=d, color='gray', linestyle='--', alpha=0.5)
plt.title('Sonic Speed vs Depth')
plt.ylabel('Sonic Speed')
plt.grid(True)

# Subplot 2: Rotation Speed
plt.subplot(3, 1, 2)
plt.plot(df['depth'], df['rotation_speed'], label='Rotation Speed', color='green')
for d in rod_change_depths:
    plt.axvline(x=d, color='gray', linestyle='--', alpha=0.5)
plt.title('Rotation Speed vs Depth')
plt.ylabel('Rotation Speed')
plt.grid(True)

# Subplot 3: Penetration Speed
plt.subplot(3, 1, 3)
plt.plot(df['depth'], df['penetration_speed'], label='Penetration Speed', color='red')
for d in rod_change_depths:
    plt.axvline(x=d, color='gray', linestyle='--', alpha=0.5)
plt.title('Penetration Speed vs Depth')
plt.xlabel('Depth (m)')
plt.ylabel('Penetration Speed')
plt.grid(True)

plt.tight_layout()
plt.savefig("data/5_figures/figure_1_mwd_lineplots.png", dpi=200)
plt.show()