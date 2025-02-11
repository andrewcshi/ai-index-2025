import pandas as pd

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
categories = ["Transparency & Explainability", "Fairness & Bias", "Security", "Privacy & Data Governance"]
conferences = {
    0: "aaai",
    1: "aies",
    2: "icml",
    3: "facct",
    4: "iclr",
    5: "neurips"
    
}

def print_dict(d):
    for key, value in d.items():
        print("-" * 100)
        print(f"Conference: {key}")
        for category, count in value.items():
            print(f"{category}: {count}")

count = 0
result = {}

for year in years:
    for i, conference in conferences.items():
        if year == 2018 and i in [0, 2, 3, 4, 5]:
            continue

        filename = f"data/{year}/{conference}{year}.csv"
        df = pd.read_csv(filename)

        total_papers = len(df)

        # filter out papers that are not in the category
        category_counts = df["category"].value_counts()
        for category in categories:
            category_counts[category] = category_counts.get(category, 0)

        category_dict = dict(category_counts)
        assert sum(category_dict.values()) == total_papers

        category_dict["Total"] = total_papers

        result[f"{conference}{year}"] = category_dict

print_dict(result)