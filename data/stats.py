import pandas as pd

def get_stats(filename):
    df = pd.read_csv(filename)
    
    total = len(df)
    transparency_count = df[df["category"] == "Transparency & Explainability"].shape[0]
    fairness_count = df[df["category"] == "Fairness & Bias"].shape[0]
    security_count = df[df["category"] == "Security"].shape[0]
    privacy_count = df[df["category"] == "Privacy & Data Governance"].shape[0]

    # sanity check the sum of the counts
    assert total == transparency_count + fairness_count + security_count + privacy_count

    return total, transparency_count, fairness_count, security_count, privacy_count

def print_stats(filename, stats):
    print("=" * 100)
    print(f"Conference: {filename.replace('_papers.csv', '')}")
    print(f"Total filtered papers: {stats[0]}")
    print(f"Transparency & Explainability papers: {stats[1]}")
    print(f"Fairness & Bias papers: {stats[2]}")
    print(f"Security papers: {stats[3]}")
    print(f"Privacy & Data papers: {stats[4]}")

def main():
    filenames = ["aaai_papers.csv", "aies_papers.csv", "icml_papers.csv", "facct_papers.csv", "iclr_papers.csv", "neurips_papers.csv"]
    for filename in filenames:
        stats = get_stats(filename)
        print_stats(filename, stats)

if __name__ == "__main__":
    main()
