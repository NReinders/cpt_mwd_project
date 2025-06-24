import pandas as pd

# Bestanden inlezen
cpt_df = pd.read_csv("data/2_clean/CPT_clean.csv")
mwd_df = pd.read_csv("data/2_clean/clean_mwd_2b.csv")

# Merge op exacte 'depth' waarden (inner join)
merged_df = pd.merge(mwd_df, cpt_df, on='depth', how='inner')

# Opslaan als nieuwe merged dataset
merged_df.to_csv("data/2_clean/merged_cpt_mwd.csv", index=False)

print(f"Merge completed. Rows in merged dataset: {len(merged_df)}")
