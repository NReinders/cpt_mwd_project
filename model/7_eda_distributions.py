import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os

# ========================================
# DISTRIBUTIE-ANALYSE
# ========================================

# Laad dataset
df = pd.read_csv("data/3_processed/processed_cpt_mwd.csv")
print("\nDistribution-analyses started")

# Toon alle kolommen
pd.set_option("display.max_columns", None)

# Selecteer numerieke kolommen
numerieke_kolommen = df.select_dtypes(include=["float64", "int64"]).columns

# Bereken skewness
print("\nSkewness of numerical columns")
skewness = df[numerieke_kolommen].skew().sort_values(ascending=False)
print(skewness)

# Visualiseer distributies met skewness
print("\nDistributionplots are being generated...")
for kolom in skewness.index:
    data = df[kolom].dropna()
    skew_val = data.skew()

    plt.figure(figsize=(7, 4))
    sns.histplot(data, kde=True, bins=30)
    plt.title(f'Distribution of {kolom}\nSkewness = {skew_val:.2f}')
    plt.xlabel(kolom)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# ===========================================
# QQ-plots en Shapiro-Wilk toets
# ===========================================

print("\nNormaalverdelingstoets (Shapiro-Wilk) & QQ-plots")

shapiro_results = []

for kolom in numerieke_kolommen:
    data = df[kolom].dropna()
    if len(data) >= 3:
        # Shapiro-Wilk test op volledige kolom
        stat, p = stats.shapiro(data)
        conclusie = "not normal" if p < 0.05 else "possibly normal"
        shapiro_results.append({
            "column": kolom,
            "W_statistic": round(stat, 3),
            "p_value": f"{p:.2e}",
            "interpretation": conclusie
        })

        # QQ-plot alleen tonen
        plt.figure(figsize=(6, 4))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"QQ-plot of {kolom}")
        plt.tight_layout()
        plt.show()

# Maak DataFrame van Shapiro-resultaten en print
shapiro_results_df = pd.DataFrame(shapiro_results)
print("\nShapiro-Wilk test results:")
print(shapiro_results_df.to_string(index=False))


# ===========================================
# fs transformatie en vergelijking
# ===========================================

print("\nComparison original vs transformed 'fs'")

# Voeg getransformeerde kolom toe aan DataFrame
df["fs_sqrt"] = np.sqrt(df["fs"])

# Skewness-vergelijking
skew_orig = df["fs"].skew()
skew_trans = df["fs_sqrt"].skew()
print(f"Skewness fs original       : {skew_orig:.3f}")
print(f"Skewness fs (sqrt-transf.) : {skew_trans:.3f}")

# Visualisatie distributies
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df["fs"].dropna(), kde=True, bins=30, ax=axs[0])
axs[0].set_title(f"Original fs\nSkewness = {skew_orig:.2f}")
axs[0].set_xlabel("fs")

sns.histplot(df["fs_sqrt"].dropna(), kde=True, bins=30, ax=axs[1])
axs[1].set_title(f"fs after sqrt-transformation\nSkewness = {skew_trans:.2f}")
axs[1].set_xlabel("fs_sqrt")

plt.tight_layout()
plt.show()

# ===========================================
# Data exporteren naar CSV
# ===========================================

# Opslaan als processed dataset
df.to_csv("data/3_processed/processed_cpt_mwd_with_fs_sqrt.csv", index=False)
print("\nProcessed dataset opgeslagen als 'data/3_processed/processed_cpt_mwd_with_fs_sqrt.csv'")