import time
import requests
from bs4 import BeautifulSoup
import csv
import re

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

def get_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h2', class_='citation_title').text.strip()
    return title

def save_html(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

def valid_paper(abstract, title):
    for _, keywords in KEYWORDS.items():
        if any(keyword.lower() in title.lower() for keyword in keywords) or \
           any(keyword.lower() in abstract.lower() for keyword in keywords):
            return True
    return False

def write_to_csv(paper):
    with open("data/icml_papers.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(paper.values())

def parse_abstract_from_meta(filename):
    with open(filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    meta_description = soup.find("meta", attrs={"name": "description"})
    if meta_description:
        return meta_description.get("content", "").strip()
    meta_og_description = soup.find("meta", property="og:description")
    if meta_og_description:
        return meta_og_description.get("content", "").strip()
    return None

def parse_authors_from_meta(filename):
    with open(filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
    authors = [m.get("content", "").strip() for m in author_metas]
    return authors

def get_author_affiliations(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")
    base_link = "https://openreview.net/"
    profile_links = [base_link + a.get("href") for a in soup.find_all("a") if a.get("href") and "/profile?id" in a.get("href")]
    affiliations = []
    for profile_link in profile_links:
        response = requests.get(profile_link)
        soup = BeautifulSoup(response.text, "html.parser")
        affiliation = soup.find("div", class_="institution").text.strip()
        affiliation = re.sub(r"\s*\(\S*?\.\S*?\)$", "", affiliation).strip()
        affiliations.append(affiliation)
    return affiliations
    
if __name__ == "__main__":
    with open("links/icml_openreview_links.txt", "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    print(f"Found {len(links)} links")

    for link in links[0:1]:        
        save_html(link, "html/icml_html.html")
        link_ = link
        title = get_title(link)
        abstract = parse_abstract_from_meta("html/icml_html.html")
        authors = parse_authors_from_meta("html/icml_html.html")
        author_affiliations = get_author_affiliations(link)
        print(link, title, abstract, authors, author_affiliations)
        if valid_paper(abstract, title):
            print("Valid paper")
        else:
            print("Invalid paper")