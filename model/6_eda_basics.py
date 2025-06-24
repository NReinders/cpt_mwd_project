import pandas as pd
import matplotlib.pyplot as plt

# Laad de gemergde dataset
df = pd.read_csv("data/2_clean/merged_cpt_mwd.csv")

print("\nEDA - BASIC EXPLORATION")
print("=" * 40)

# Alle kolommen zichtbaar maken
pd.set_option('display.max_columns', None)

# .info()
print("\n.info()")
print("-" * 20)
df.info()

# .describe()
print("\n.describe()")
print("-" * 20)
print(df.describe().transpose())

# .head()
print("\n.head()")
print("-" * 20)
print(df.head())

# Nulls
print("\nAantal nulls per kolom:")
print("-" * 20)
print(df.isnull().sum())

# Duplicaten
duplicates = df.duplicated().sum()
print(f"\nAantal volledig dubbele rijen: {duplicates}")

# Unieke waarden per kolom
print("\nAantal unieke waarden per kolom:")
print("-" * 20)
print(df.nunique())

# Constante kolommen
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print("\nConstante kolommen:")
print(constant_cols if constant_cols else "Geen constante kolommen gevonden.")

# Datatype overzicht
print("\nDatatypes per kolom:")
print("-" * 20)
print(df.dtypes)

# Rod wissels over diepte
df_sorted = df.sort_values("depth")
plt.figure(figsize=(6, 10))
plt.step(df_sorted["number_of_rods"], df_sorted["depth"], where='post')
plt.title("Rod Changes Over Depth")
plt.xlabel("Number of Rods")
plt.ylabel("Depth (m)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()

# Value counts categorische kolommen
print("\nValue counts per categorische kolom:")
print("-" * 20)
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    print(f"\nKolom: {col}")
    print(df[col].value_counts())

# Scatterplot depth vs depth_raw
plt.figure(figsize=(6, 4))
plt.scatter(df["depth"], df["depth_raw"], alpha=0.5)
plt.title("Depth vs Depth Raw")
plt.xlabel("Depth (m)")
plt.ylabel("Depth Raw (m)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Check of depth uniek is
is_unique = df["depth"].is_unique
print(f"\nZijn de depth-waarden uniek?: {is_unique}")

# Scatterplot qc en fs vs depth
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(df["qc"], df["depth"], alpha=0.5)
plt.xlabel("qc")
plt.ylabel("Depth (m)")
plt.title("qc vs Depth")
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.scatter(df["fs"], df["depth"], alpha=0.5)
plt.xlabel("fs")
plt.ylabel("Depth (m)")
plt.title("fs vs Depth")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# === Dataset verwerken ===

# 1. Verwijder constante kolommen
to_drop = ["running_status", "location_status", "location"]
df_processed = df.drop(columns=to_drop)

# 2. Binaire encodering voor 'sonic_status'
df_processed["sonic_status"] = df_processed["sonic_status"].map({"started": 1, "stopped": 0})

# 3. Binaire encodering voor 'operation_mode'
df_processed["operation_mode"] = df_processed["operation_mode"].map({"Dynamic": 1, "Manual": 0})

# Opslaan als processed dataset
df_processed.to_csv("data/3_processed/processed_cpt_mwd.csv", index=False)
print("\nProcessed dataset saved as 'data/3_processed/processed_cpt_mwd.csv'")
