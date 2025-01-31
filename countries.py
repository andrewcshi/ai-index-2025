import csv
import ast
import pycountry
from Levenshtein import distance
import pandas as pd
import math

def match_pycountry(aff):
    s = aff.lower()
    for c in pycountry.countries:
        if c.name.lower() in s:
            return c.name
    return ""

def match_by_levenshtein(aff, inst_dict):
    best_dist = math.inf
    best_country = ""
    for inst, country in inst_dict.items():
        d = distance(aff.lower(), inst.lower())
        if d < best_dist:
            best_dist = d
            best_country = country
    return best_country if best_dist <= 3 else ""

def find_country(aff, cache, inst_dict):
    if aff in cache:
        return cache[aff]
    c = match_pycountry(aff)
    if not c:
        c = match_by_levenshtein(aff, inst_dict)
    cache[aff] = c
    return c

def main(input_csv, univ_csv, output_csv):
    df = pd.read_csv(univ_csv)
    inst_to_country = {}
    for _, row in df.iterrows():
        inst_to_country[row["name"]] = row["country"]
    cache = {}
    with open(input_csv, "r", encoding="utf-8") as fin, open(output_csv, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames + ["author_countries"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            raw_aff = row["author_affiliations"]
            try:
                aff_list = ast.literal_eval(raw_aff)
                if not isinstance(aff_list, list):
                    aff_list = [raw_aff]
            except:
                aff_list = [raw_aff]
            countries = []
            for a in aff_list:
                c = find_country(a.strip(), cache, inst_to_country)
                countries.append(c)
            row["author_countries"] = str(countries)
            writer.writerow(row)

if __name__ == "__main__":
    main("data/aaai_papers.csv", "inst_to_country.csv", "output_with_countries.csv")