import pandas as pd

KEYWORDS = {
    "Transparency & Explainability": [
        'Algorithmic Transparency',
        'Explainable AI',
        'Explainable Artificial Intelligence',
        'XAI',
        'Interpretability',
        'Model Explainability',
        'Explainability',
        'Transparency',
        'Human-understandable decisions',
        'Audit',
        'Auditing',
        'Outcome explanation',
        'Causality',
        'Causal reasoning',
        'Interpretable models',
        'Explainable models',
    ],
    "Fairness & Bias": [
        'Algorithmic Fairness',
        'Bias Detection',
        'Bias',
        'Discrimination',
        'Fair ML',
        'Fair Machine Learning',
        'Unfairness',
        'Unfair',
        'Ethical algorithm design',
        'Bias mitigation',
        'Representational fairness',
        'Group fairness',
        'Individual fairness',
        'Fair data practices',
        'Equity in AI',
        'Equity in Artificial Intelligence',
        'Justice',
        'Non-discrimination',
    ],
    "Privacy & Data Governance": [
        'Data privacy',
        'Data governance',
        'Differential privacy',
        'Data protection',
        'Data breach',
        'Secure data storage',
        'Data ethics',
        'Data integrity',
        'Data transparency',
        'Privacy by design',
        'Confidentiality',
        'Inference privacy',
        'Machine unlearning',
        'Privacy-preserving',
        'Data protection',
        'Anonymity',
        'Trustworthy data curation',
    ],
    "Security": [
        'Red teaming',
        'Adversarial attack',
        'Cybersecurity',
        'Threat detection',
        'Vulnerability assessment',
        'Ethical hacking',
        'Fraud detection',
        'Security ethics',
        'AI incident',
        'Artificial Intelligence incident',
        'Security',
        'Safety',
        'Audits',
        'Attacks',
        'Forensic analysis',
        'Adversarial learning',
    ],
}

datasets = [
    "aaai2024.csv",
    "aies2024.csv",
    "facct2024.csv",
    "iclr2024.csv",
    "icml2024.csv",
    "neurips2024.csv"
]

def contains_keyword(text, kw_list):
    """
    Checks if any keyword in kw_list appears (case-insensitive) in text.
    Returns True if there's at least one match.
    """
    if pd.isnull(text):
        return False
    
    text_lower = str(text).lower()
    for kw in kw_list:
        if kw.lower() in text_lower:
            return True
    return False

for dataset in datasets:
    # Load the CSV
    df = pd.read_csv(f"data/2024/{dataset}")
    
    # Prepare a container for all new “additions” rows
    new_rows = []
    
    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        
        # Combine the relevant text columns into one string for simplicity
        combined_text = ""
        for col in ["title", "abstract", "keywords", "ccs_concepts"]:
            val = row.get(col, "")
            if pd.notnull(val):
                combined_text += f"{val} "
        
        combined_text = combined_text.lower()
        
        # Check each custom category in our KEYWORDS dict
        for cat, cat_keywords in KEYWORDS.items():
            # Only consider categories different from the row’s existing category
            if row["category"] == cat:
                continue
            
            # If ANY keyword for this cat is in the combined text, we add a new row
            if any(kw.lower() in combined_text for kw in cat_keywords):
                new_row = row.copy()
                new_row["category"] = cat
                new_rows.append(new_row)
                # Break if we only want one match per category
                # If you want to allow multiple categories from the same row, remove break
                break
    
    # If we found new rows for this dataset, save them
    if new_rows:
        additions_df = pd.DataFrame(new_rows)
        # Create an “additions” filename from the original .csv name
        conference_name = dataset.split(".")[0]
        
        # NOTE: You asked to save as “{conference}_additions.py” 
        # but typically you’d store CSV data in “.csv”. Adjust as needed.
        additions_filename = f"data/2024/{conference_name}_additions.csv"
        
        additions_df.to_csv(additions_filename, index=False)
        print(f"Created {additions_filename} with {len(additions_df)} new rows.")
    else:
        print(f"No new category matches found in {dataset}.")