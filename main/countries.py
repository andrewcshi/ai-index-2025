import pandas as pd
import pycountry
import ast
import re
from Levenshtein import distance
from collections import Counter, defaultdict

###############################################################################
# 0) Data Loading (Institutions, Cities, etc.)
###############################################################################

inst_df = pd.read_csv("datasets/institutions.csv")
inst_dict = {}
for _, r in inst_df.iterrows():
    name = r.get("name", "")
    country = r.get("country", "")
    if isinstance(name, str) and isinstance(country, str):
        name = name.strip()
        country = country.strip()
        if name:
            inst_dict[name] = country

city_df = pd.read_csv("datasets/cities.csv")
city_dict = {}
for _, r in city_df.iterrows():
    c_ascii = r.get("city_ascii", "")
    c_country = r.get("country", "")
    if isinstance(c_ascii, str) and isinstance(c_country, str):
        c_ascii = c_ascii.strip()
        c_country = c_country.strip()
        if c_ascii:
            city_dict[c_ascii] = c_country

synonyms = {
    "usa": "United States",
    "USA": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "china": "China",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "russia": "Russian Federation",
}

nationality_dict = {
    "chinese": "China",
    "japanese": "Japan",
    "korean": "South Korea",
    "french": "France",
    "german": "Germany",
    "indian": "India",
    "british": "United Kingdom",
    "american": "United States",
}

tld_country_map = {
    ".cn": "China",
    ".kr": "South Korea",
    ".jp": "Japan",
    ".uk": "United Kingdom",
    ".ac.uk": "United Kingdom",
    ".edu.cn": "China",
    ".edu.hk": "Hong Kong",
    ".edu.sg": "Singapore",
    ".edu.tw": "Taiwan",
    ".edu.au": "Australia",
    ".edu": "United States",
}

lower_inst = {k.lower(): v for k, v in inst_dict.items()}
lower_city = {k.lower(): v for k, v in city_dict.items()}
all_py_countries = list(pycountry.countries)
pycountry_names = [c.name.lower() for c in all_py_countries]

###############################################################################
# 1) Helpers
###############################################################################

def levenshtein_similarity(a, b):
    """Normalized Levenshtein similarity."""
    if not a or not b:
        return 0.0
    return 1 - distance(a, b) / max(len(a), len(b))

def tokenize_affiliation(affil_str):
    """
    Split affiliation on commas, semicolons, parentheses, etc.
    Return a list of trimmed tokens.
    """
    tokens = re.split(r"[,\(\);]+", affil_str)
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens

def analyze_token(token):
    """
    Examine a single token and return a list of (country, confidence).
    We'll accumulate them outside and then combine results per affiliation.
    """
    results = []
    txt = token.lower()

    # 1) Email TLD check
    emails = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", txt)
    for email in emails:
        for tld in sorted(tld_country_map.keys(), key=lambda x: -len(x)):
            if email.endswith(tld):
                results.append((tld_country_map[tld], 0.9))
                break

    # 2) Pycountry direct mention
    for pc_name, pc_obj in zip(pycountry_names, all_py_countries):
        if pc_name in txt:
            results.append((pc_obj.name, 0.8))
            break

    # 3) Substring match with known institutions
    for inst_name, inst_country in lower_inst.items():
        if inst_name in txt:
            results.append((inst_country, 0.8))

    # 4) Substring match with known cities
    for city_name, city_country in lower_city.items():
        if city_name in txt:
            results.append((city_country, 0.75))

    # 5) Synonyms
    for syn_key, syn_country in synonyms.items():
        if syn_key in txt:
            results.append((syn_country, 0.7))

    # 6) Nationalities
    for nat_key, nat_country in nationality_dict.items():
        if nat_key in txt:
            results.append((nat_country, 0.6))

    # 7) Fuzzy match with institutions (fallback)
    best_fuzzy_score = 0.0
    best_fuzzy_country = ""
    for inst_name, inst_country in lower_inst.items():
        s_val = levenshtein_similarity(txt, inst_name)
        if s_val > best_fuzzy_score:
            best_fuzzy_score = s_val
            best_fuzzy_country = inst_country
    if best_fuzzy_score >= 0.65:
        results.append((best_fuzzy_country, best_fuzzy_score))

    return results

def aggregate_token_results(all_token_results):
    """
    Combine (country, confidence) from tokens by summing confidences for each country.
    Return (best_country, sum_conf).
    """
    counter = defaultdict(float)
    for token_result_list in all_token_results:
        for (country, conf) in token_result_list:
            counter[country] += conf

    if not counter:
        return "", 0.0

    best_country = ""
    best_sum_conf = 0.0
    for ctry, sum_conf in counter.items():
        if sum_conf > best_sum_conf:
            best_country = ctry
            best_sum_conf = sum_conf

    return best_country, best_sum_conf

def get_country_and_confidence(affil_str):
    """
    Tokenize affiliation, gather signals, pick best (country, confidence sum).
    """
    tokens = tokenize_affiliation(affil_str)
    all_token_results = [analyze_token(t) for t in tokens]
    best_country, best_conf = aggregate_token_results(all_token_results)
    return best_country, best_conf

###############################################################################
# 2) Multi-pass routine for a single DataFrame
###############################################################################

def run_multi_pass(df_data, pass1_threshold=0.6, recheck_threshold=0.6):
    """
    Given a DataFrame with "author_affiliations",
    produce final "author_countries" after Pass 1 → 2 → 2.5 → 3.
    Returns df_data + a dict for pass3 global unification (affil cache).
    """
    # For pass 3, store {exact_affil_str: [(country, conf), ...]}
    all_affils_cache = {}

    ###########################################################################
    # PASS 1: Basic token-level assignment
    ###########################################################################

    pass1_countries_list = []
    pass1_confidences_list = []

    for idx, row in df_data.iterrows():
        affil_col = row.get("author_affiliations", "")
        if not isinstance(affil_col, str) or not affil_col.strip():
            pass1_countries_list.append([])
            pass1_confidences_list.append([])
            continue

        try:
            affils = ast.literal_eval(affil_col)
            if not isinstance(affils, list):
                affils = [affil_col]
        except:
            affils = [affil_col]

        row_countries = []
        row_confidences = []
        for affil_str in affils:
            ctry, conf = get_country_and_confidence(affil_str)
            if conf < pass1_threshold:
                ctry = ""
                conf = 0.0

            row_countries.append(ctry)
            row_confidences.append(conf)

            # accumulate for pass 3
            if affil_str not in all_affils_cache:
                all_affils_cache[affil_str] = []
            all_affils_cache[affil_str].append((ctry, conf))

        pass1_countries_list.append(row_countries)
        pass1_confidences_list.append(row_confidences)

    df_data["pass1_countries"] = pass1_countries_list
    df_data["pass1_confidences"] = pass1_confidences_list

    ###########################################################################
    # PASS 2: Row-level majority correction
    ###########################################################################

    pass2_countries_list = []
    for idx, row in df_data.iterrows():
        row_countries = row["pass1_countries"]
        affil_col = row.get("author_affiliations", "")
        if not isinstance(row_countries, list):
            pass2_countries_list.append([])
            continue

        assigned = [c for c in row_countries if c]
        c_count = Counter(assigned)
        if not c_count:
            pass2_countries_list.append(row_countries)
            continue

        majority_country, majority_count = c_count.most_common(1)[0]
        total_affils = len(row_countries)

        # If single country >= 50% of total (including unassigned)
        if majority_count / total_affils >= 0.5:
            # re-check outliers/unassigned
            try:
                affils2 = ast.literal_eval(affil_col)
                if not isinstance(affils2, list):
                    affils2 = [affil_col]
            except:
                affils2 = [affil_col]

            new_countries = []
            for i, old_country in enumerate(row_countries):
                if old_country == majority_country:
                    new_countries.append(old_country)
                    continue

                affil_str = affils2[i]
                ctry2, conf2 = get_country_and_confidence(affil_str)
                if ctry2 == majority_country and conf2 >= recheck_threshold:
                    new_countries.append(majority_country)
                else:
                    new_countries.append(old_country)

            pass2_countries_list.append(new_countries)
        else:
            pass2_countries_list.append(row_countries)

    df_data["pass2_countries"] = pass2_countries_list

    ###########################################################################
    # PASS 2.5: If top country is >60% of assigned, unify
    ###########################################################################

    pass2_5_countries_list = []
    for idx, row in df_data.iterrows():
        row_countries = row["pass2_countries"]
        if not isinstance(row_countries, list):
            pass2_5_countries_list.append(row_countries)
            continue

        assigned_countries = [c for c in row_countries if c]
        c_count = Counter(assigned_countries)
        if not c_count:
            pass2_5_countries_list.append(row_countries)
            continue

        majority_country, majority_count = c_count.most_common(1)[0]
        total_assigned = sum(c_count.values())  # ignoring empty
        share = majority_count / total_assigned

        # If top country's share > 60% among assigned, unify
        if share > 0.6:
            new_row = [majority_country if c else "" for c in row_countries]
            pass2_5_countries_list.append(new_row)
        else:
            pass2_5_countries_list.append(row_countries)

    df_data["pass2_5_countries"] = pass2_5_countries_list

    ###########################################################################
    # PASS 3: Global unification for repeated affiliation strings
    ###########################################################################

    final_affil_assignments = {}
    for affil_str, cvals in all_affils_cache.items():
        aggregator = defaultdict(float)
        for (ctry, conf) in cvals:
            aggregator[ctry] += conf

        best_ctry = ""
        best_conf_sum = 0.0
        for ctry, total_conf in aggregator.items():
            if total_conf > best_conf_sum:
                best_ctry = ctry
                best_conf_sum = total_conf
        final_affil_assignments[affil_str] = best_ctry

    final_countries_list = []
    for idx, row in df_data.iterrows():
        row_countries = row["pass2_5_countries"]
        if not isinstance(row_countries, list):
            final_countries_list.append([])
            continue

        affil_col = row.get("author_affiliations", "")
        try:
            affils3 = ast.literal_eval(affil_col)
            if not isinstance(affils3, list):
                affils3 = [affil_col]
        except:
            affils3 = [affil_col]

        new_countries = []
        for i, old_country in enumerate(row_countries):
            affil_str = affils3[i]
            best_global = final_affil_assignments.get(affil_str, "")
            if (not old_country) and best_global:
                new_countries.append(best_global)
            elif old_country and best_global and (old_country != best_global):
                new_countries.append(best_global)
            else:
                new_countries.append(old_country)

        final_countries_list.append(new_countries)

    df_data["author_countries"] = final_countries_list

    return df_data, all_affils_cache

###############################################################################
# 3) Master function to process multiple files, handle outliers, remove Turkey/Burma,
#    then do a final pass to unify minority countries if there's a single >50% majority
###############################################################################

def process_all_files(
    data_paths,
    pass1_threshold=0.6,
    recheck_threshold=0.6,
    outlier_pct=0.03
):
    """
    For each CSV in data_paths:
      1) Load the file
      2) run multi-pass assignment
      3) store results in a combined big_df
    After all files:
      4) find outlier countries (<3% of total assigned)
      5) re-check any row assigned to outlier countries
      6) forcibly remove Turkey/Burma
      7) unify minority countries if there's a single strict majority (>50%)
      8) re-split big_df by source_file, save each
    """
    df_list = []

    #-----------------------
    # Step A: run multi-pass
    #-----------------------
    for p in data_paths:
        print(f"Processing {p}")
        df_tmp = pd.read_csv(p)
        # Keep track of which file each row came from
        df_tmp["source_file"] = p

        df_processed, _ = run_multi_pass(
            df_tmp,
            pass1_threshold=pass1_threshold,
            recheck_threshold=recheck_threshold
        )
        df_list.append(df_processed)

    # Combine them
    big_df = pd.concat(df_list, ignore_index=True)

    #---------------------------------------
    # Step B: Identify countries <3% (outliers)
    #---------------------------------------
    all_country_assignments = big_df["author_countries"].explode()
    all_country_assignments = all_country_assignments[all_country_assignments != ""]
    country_counts = all_country_assignments.value_counts()
    total_assigned = country_counts.sum()
    threshold = outlier_pct * total_assigned

    outlier_countries = set(country_counts[country_counts < threshold].index)
    print("\nOutlier countries (<3%):", outlier_countries, "\n")

    #---------------------------------------
    # Step C: Re-check rows with outlier countries
    #---------------------------------------
    revised_countries = []
    for idx, row in big_df.iterrows():
        affil_col = row.get("author_affiliations", "")
        row_countries = row.get("author_countries", [])
        if not isinstance(row_countries, list):
            revised_countries.append(row_countries)
            continue

        try:
            affils = ast.literal_eval(affil_col)
            if not isinstance(affils, list):
                affils = [affil_col]
        except:
            affils = [affil_col]

        new_row_countries = []
        for affil_str, ctry in zip(affils, row_countries):
            if ctry in outlier_countries and ctry.strip():
                # Attempt a re-check
                ctry2, conf2 = get_country_and_confidence(affil_str)
                if ctry2 and (ctry2 not in outlier_countries) and (conf2 > 0.6):
                    new_row_countries.append(ctry2)
                else:
                    # keep old or set to "" -- we'll just keep old here
                    new_row_countries.append(ctry)
            else:
                new_row_countries.append(ctry)

        revised_countries.append(new_row_countries)

    big_df["author_countries"] = revised_countries

    #---------------------------------------
    # Step D: Remove Turkey/Burma
    #---------------------------------------
    final_countries = []
    for idx, row in big_df.iterrows():
        affil_col = row.get("author_affiliations", "")
        row_countries = row.get("author_countries", [])
        if not isinstance(row_countries, list):
            final_countries.append(row_countries)
            continue

        try:
            affils = ast.literal_eval(affil_col)
            if not isinstance(affils, list):
                affils = [affil_col]
        except:
            affils = [affil_col]

        new_row_countries = []
        for affil_str, ctry in zip(affils, row_countries):
            if ctry in ["Turkey", "Burma"]:
                # forcibly re-check
                ctry2, conf2 = get_country_and_confidence(affil_str)
                if ctry2 not in ["Turkey", "Burma"] and conf2 > 0.6:
                    new_row_countries.append(ctry2)
                else:
                    new_row_countries.append("")
            else:
                new_row_countries.append(ctry)

        final_countries.append(new_row_countries)

    big_df["author_countries"] = final_countries

    #---------------------------------------
    # Step E (Final Pass): If there's a single strict majority (>50%) in each row,
    # unify all assigned to that majority. Exact ties => do nothing.
    #---------------------------------------
    unified_countries = []
    for idx, row in big_df.iterrows():
        row_countries = row.get("author_countries", [])
        if not isinstance(row_countries, list) or not row_countries:
            unified_countries.append(row_countries)
            continue

        assigned = [c for c in row_countries if c]
        if not assigned:
            unified_countries.append(row_countries)
            continue

        c_count = Counter(assigned)
        most_common = c_count.most_common(2)
        if len(most_common) == 1:
            # Only one distinct country => unify trivially
            majority_country = most_common[0][0]
            new_row = [majority_country if c else "" for c in row_countries]
            unified_countries.append(new_row)
        else:
            (top_country, top_count) = most_common[0]
            (second_country, second_count) = most_common[1]
            total_assigned = sum(c_count.values())

            if top_count > (0.5 * total_assigned):
                # If it's not a tie with second place
                if top_count == second_count:
                    # exact tie -> do nothing
                    unified_countries.append(row_countries)
                else:
                    # unify everything to the majority
                    new_row = [top_country if c else "" for c in row_countries]
                    unified_countries.append(new_row)
            else:
                # no >50% majority
                unified_countries.append(row_countries)

    big_df["author_countries"] = unified_countries

    #---------------------------------------
    # Step F: Re-split by source_file and write
    #---------------------------------------
    for src_file in big_df["source_file"].unique():
        df_sub = big_df[big_df["source_file"] == src_file].copy()
        out_path = src_file.replace(".csv", "_with_countries.csv")
        print(f"Writing updated file for {src_file} -> {out_path}")
        df_sub.to_csv(out_path, index=False)

    return big_df  # if you want the final DataFrame in memory

###############################################################################
# 4) Run script
###############################################################################

def __main__():
    data_paths = [
        "data/2024/aaai2024.csv",
        "data/2024/aies2024.csv",
        "data/2024/facct2024.csv",
        "data/2024/icml2024.csv",
        "data/2024/iclr2024.csv",
        "data/2024/neurips2024.csv"
    ]
    _ = process_all_files(data_paths)

if __name__ == "__main__":
    __main__()
