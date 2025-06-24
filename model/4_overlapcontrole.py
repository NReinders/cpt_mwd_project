import pandas as pd

# Bestanden inlezen
cpt_df = pd.read_csv("data/2_clean/CPT_clean.csv")
mwd_df = pd.read_csv("data/2_clean/clean_mwd_2b.csv")

# Unieke depth-waarden extraheren
cpt_depths = set(cpt_df['depth'].round(2))
mwd_depths = set(mwd_df['depth'].round(2))

# Overlap en verschillen bepalen
matching_depths = cpt_depths & mwd_depths
only_in_cpt = cpt_depths - mwd_depths
only_in_mwd = mwd_depths - cpt_depths

# Resultaten weergeven
print(f" Total unique CPT depths: {len(cpt_depths)}")
print(f" Total unique MWD depths: {len(mwd_depths)}")
print(f" Overlapping depths: {len(matching_depths)}")
print(f" Depths only in CPT: {len(only_in_cpt)}")
print(f" Depths only in MWD: {len(only_in_mwd)}")

# Voorbeeldwaarden tonen
print("\n Example depths only in CPT:", sorted(list(only_in_cpt))[:10])
print(" Example depths only in MWD:", sorted(list(only_in_mwd))[:10])
