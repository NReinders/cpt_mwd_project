import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Pad naar de dataset met getransformeerde target
input_path = "data/3_processed/processed_cpt_mwd_with_fs_sqrt.csv"
output_path = "data/3_processed/processed_cpt_mwd_with_outlier_flags.csv"

# Laad de dataset
df = pd.read_csv(input_path)

# Selecteer alleen numerieke kolommen (excl. depth en binaire kolommen)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
binary_cols = [col for col in numerical_cols if df[col].nunique() == 2]
numerical_cols = [col for col in numerical_cols if col not in binary_cols and col != "depth"]

# IQR outlier flags toevoegen
outlier_flags = pd.DataFrame(index=df.index)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_flags[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)

# Voeg de outlier flags toe aan de originele dataframe
df_with_flags = pd.concat([df, outlier_flags], axis=1)

# Opslaan van de nieuwe dataset
df_with_flags.to_csv(output_path, index=False)
print(f"Dataset met outlier flags opgeslagen als: {output_path}")

# Outlier-aantallen per kolom berekenen
outlier_counts = outlier_flags.sum().sort_values(ascending=False)
print("\nAantal outliers per kolom:")
print(outlier_counts)

# Visualiseer alleen kolommen met outliers > 0
cols_with_outliers = outlier_counts[outlier_counts > 0].index.str.replace('_outlier', '').tolist()

# Genereer meerdere figuren met max 6 boxplots per figuur
batch_size = 6
num_batches = math.ceil(len(cols_with_outliers) / batch_size)

for i in range(num_batches):
    batch = cols_with_outliers[i * batch_size:(i + 1) * batch_size]
    rows = (len(batch) + 2) // 3  # max 3 kolommen per rij
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for j, col in enumerate(batch):
        sns.boxplot(x=df[col], ax=axes[j], color='skyblue')
        axes[j].set_title(col)

    # Verwijder lege subplots
    for k in range(len(batch), len(axes)):
        fig.delaxes(axes[k])

    fig.suptitle(f"Boxplots van kolommen met outliers (batch {i + 1})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
