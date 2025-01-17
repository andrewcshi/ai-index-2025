import requests
from bs4 import BeautifulSoup
import csv
import concurrent.futures

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


def get_paper_attributes(url):
    """
    Given an OpenReview paper URL, scrapes and returns a dictionary of attributes:
      - link: The OpenReview URL.
      - title: The paper title.
      - abstract: The paper abstract.
      - author_names: A list of authors.
    
    Process:
      1. Load the OpenReview page.
      2. Scrape the title from the <h2> tag with class "note_content_title".
      3. Find the field with the label "Abstract:" and get its associated value.
      4. Find the field with the label "Authors:" and get the author list.
    """
    pass


def valid_paper(paper):
    """
    Determines whether a paper is valid based on keyword filtering.
    
    (Function implementation to be added later.)
    """
    for category, keywords in KEYWORDS.items():
        # check title and abstract (case-insensitive)
        if any(keyword.lower() in paper.get("title", "").lower() for keyword in keywords) \
           or any(keyword.lower() in paper.get("abstract", "").lower() for keyword in keywords):
            paper["category"] = category  # set the category
            return True
    return False


def write_to_csv(paper):
    """
    Writes the paper data as a row to a CSV file.
    
    (Function implementation to be added later.)
    """
    with open("data/icml_papers.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(paper.values())


if __name__ == "__main__":
    pass