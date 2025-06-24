import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# === STEP 1: Load and Clean Data ===

input_path = "data/3_processed/processed_cpt_mwd_with_outlier_flags.csv"
df = pd.read_csv(input_path)

# 1.1 Filter negatieve fs-waarden eruit
initial_len = len(df)
df = df[df['fs'] >= 0].copy()
removed_rows = initial_len - len(df)
print(f"Removed {removed_rows} rows where fs < 0.")

# 1.2 Check op NaN in fs_sqrt
n_nan_fs_sqrt = df['fs_sqrt'].isna().sum()
print(f"Number of NaN values in fs_sqrt after filtering: {n_nan_fs_sqrt}")

targets = ['fs', 'fs_sqrt', 'qc']

# === STEP 2: Scatterplots Depth vs Targets ===

for target in targets:
    sns.scatterplot(
        data=df, 
        x=target, 
        y='depth', 
        hue='number_of_rods',
        palette='tab10',
        s=30,
        alpha=0.6,
        linewidth=0
    )
    plt.title(f'{target} vs Depth (per Rod)')
    plt.xlabel(target)
    plt.ylabel('Depth')
    plt.gca().invert_yaxis()
    plt.legend(title='Rod', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# === STEP 3: Correlation depth vs target ===

for target in targets:
    corr = df['depth'].corr(df[target])
    print(f"Pearson correlation between depth and {target}: {corr:.3f}")

# === STEP 4: Feature Importance per Target ===

exclude_cols = ['fs', 'fs_sqrt', 'qc'] + [col for col in df.columns if '_outlier' in col]
X_base = df.drop(columns=exclude_cols, errors='ignore')
X_base = X_base.select_dtypes(include=[np.number])

for target in targets:
    y = df[target]
    X = X_base.copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)

    print(f"\nFeature importances for target '{target}':")
    print(importance_df)

    sns.barplot(data=importance_df.head(10), x='importance', y='feature', palette='Blues_d')
    plt.title(f'Top 10 Feature Importances for {target}')
    plt.tight_layout()
    plt.show()

# === STEP 5: Correlation Analysis and Scatterplots ===

# Voor correlatieanalyse: verwijder _outlier kolommen en number_of_rods
columns_to_exclude = [col for col in df.columns if '_outlier' in col] + ['number_of_rods']
df_corr = df.drop(columns=columns_to_exclude, errors='ignore')

# Houd alleen numerieke kolommen
df_corr = df_corr.select_dtypes(include=[np.number])

# Verwijder 'depth' (bewust behouden bij feature importance!)
df_corr = df_corr.drop(columns=["depth"], errors="ignore")

# === 5.1a Pearson correlation with target variables ===
print("\n=== Pearson correlation with target variables ===")
correlations_with_targets = df_corr.corr(method="pearson")[targets].drop(index=targets, errors='ignore')
print(correlations_with_targets.sort_values(by='qc', ascending=False))

# === 5.1b Spearman correlation with target variables ===
print("\n=== Spearman correlation with target variables ===")
spearman_with_targets = df_corr.corr(method="spearman")[targets].drop(index=targets, errors='ignore')
print(spearman_with_targets.sort_values(by='qc', ascending=False))

# === 5.2a Scatterplots best Pearson-correlated feature per target ===
print("\n=== Best Pearson-correlated feature per target ===")
best_features_pearson = {}
for target in targets:
    top_feature = correlations_with_targets[target].abs().sort_values(ascending=False).index[0]
    best_features_pearson[target] = top_feature

for target, feature in best_features_pearson.items():
    corr_value = df_corr[feature].corr(df_corr[target], method="pearson")
    sns.scatterplot(data=df, x=feature, y=target, alpha=0.5)
    plt.title(f'{feature} vs {target} (Pearson r = {corr_value:.2f})')
    plt.tight_layout()
    plt.show()

# === 5.2b Scatterplots best Spearman-correlated feature per target ===
print("\n=== Best Spearman-correlated feature per target ===")
best_features_spearman = {}
for target in targets:
    top_feature = spearman_with_targets[target].abs().sort_values(ascending=False).index[0]
    best_features_spearman[target] = top_feature

for target, feature in best_features_spearman.items():
    corr_value = df_corr[feature].corr(df_corr[target], method="spearman")
    sns.scatterplot(data=df, x=feature, y=target, alpha=0.5)
    plt.title(f'{feature} vs {target} (Spearman ρ = {corr_value:.2f})')
    plt.tight_layout()
    plt.show()


# === 5.3a Heatmap van sterke Pearson-correlaties tussen inputfeatures ===

input_features = df_corr.drop(columns=targets, errors="ignore")
binary_cols = [col for col in input_features.columns if input_features[col].nunique() == 2]
input_features = input_features.drop(columns=binary_cols)

corr_matrix = input_features.corr()
mask = (abs(corr_matrix) > 0.7) & (corr_matrix != 1.0)
relevant_features = mask.any(axis=1)
filtered_corr = corr_matrix.loc[relevant_features, relevant_features]

plt.figure(figsize=(8, 6))
sns.heatmap(filtered_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': 0.75})
plt.title("Strong Pearson Correlations between Features (|r| > 0.7)")
plt.tight_layout()
plt.show()

# === 5.3b Heatmap van sterke Spearman-correlaties tussen inputfeatures ===

print("\n=== Heatmap van sterke Spearman-correlaties tussen inputfeatures ===")

# Start vanaf dezelfde input_features als bij Pearson
input_features_spearman = df_corr.drop(columns=targets, errors="ignore")
binary_cols_spearman = [col for col in input_features_spearman.columns if input_features_spearman[col].nunique() == 2]
input_features_spearman = input_features_spearman.drop(columns=binary_cols_spearman)

# Spearman correlatiematrix
spearman_corr_matrix = input_features_spearman.corr(method="spearman")

# Filter op sterke correlaties
mask_spearman = (abs(spearman_corr_matrix) > 0.7) & (spearman_corr_matrix != 1.0)
relevant_features_spearman = mask_spearman.any(axis=1)
filtered_spearman_corr = spearman_corr_matrix.loc[relevant_features_spearman, relevant_features_spearman]

# Plot de heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(filtered_spearman_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': 0.75})
plt.title("Strong Spearman Correlations between Features (|ρ| > 0.7)")
plt.tight_layout()
plt.show()
