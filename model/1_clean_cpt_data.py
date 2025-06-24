import csv

input_path = "data/1_source/CPT-en-Drilling-01.csv"
output_path = "data/2_clean/CPT_clean.csv"

rows = []
with open(input_path, "r", encoding="latin1") as f:
    reader = csv.reader(f, delimiter="\t")
    for i, row in enumerate(reader):
        if i == 0:
            continue  # sla de eerste rij over
        if len(row) >= 3:
            # verwijder aanhalingstekens en pak de eerste 3 kolommen
            cleaned = [col.replace('"', '').strip() for col in row[:3]]
            rows.append(cleaned)

# Save naar CSV
with open(output_path, "w", newline='', encoding="utf-8") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["depth", "qc", "fs"])
    writer.writerows(rows)

print(f"✔️ Bestand opgeslagen als: {output_path}")


