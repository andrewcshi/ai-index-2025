import os
import pandas as pd

csv_file = "inst_to_country.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=["name", "country"])

inst_to_country = {}
for _, row in df.iterrows():
    inst_to_country[row["name"]] = row["country"]

with open(csv_file, "w", encoding="utf-8") as f:
    for inst, country in inst_to_country.items():
        f.write(f"{inst},{country}\n")
