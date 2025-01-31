import os
import csv
import re
import time
import requests
from bs4 import BeautifulSoup
import random
KEYWORDS = {
    "Transparency & Explainability": [
        'Algorithmic Transparency', 'Explainable AI', 'Explainable Artificial Intelligence', 'XAI',
        'Interpretability', 'Model Explainability', 'Explainability', 'Transparency',
        'Human-understandable decisions', 'Audit', 'Auditing', 'Outcome explanation',
        'Causality', 'Causal reasoning', 'Interpretable models', 'Explainable models'
    ],
    "Fairness & Bias": [
        'Algorithmic Fairness', 'Bias Detection', 'Bias', 'Discrimination', 'Fair ML',
        'Fair Machine Learning', 'Unfairness', 'Unfair', 'Ethical algorithm design',
        'Bias mitigation', 'Representational fairness', 'Group fairness', 'Individual fairness',
        'Fair data practices', 'Equity in AI', 'Equity in Artificial Intelligence', 'Justice',
        'Non-discrimination'
    ],
    "Privacy & Data Governance": [
        'Data privacy', 'Data governance', 'Differential privacy', 'Data protection',
        'Data breach', 'Secure data storage', 'Data ethics', 'Data integrity', 'Data transparency',
        'Privacy by design', 'Confidentiality', 'Inference privacy', 'Machine unlearning',
        'Privacy-preserving', 'Data protection', 'Anonymity', 'Trustworthy data curation'
    ],
    "Security": [
        'Red teaming', 'Adversarial attack', 'Cybersecurity', 'Threat detection',
        'Vulnerability assessment', 'Ethical hacking', 'Fraud detection', 'Security ethics',
        'AI incident', 'Artificial Intelligence incident', 'Security', 'Safety', 'Audits',
        'Attacks', 'Forensic analysis', 'Adversarial learning'
    ],
}

CSV_FILE = "data/neurips_papers.csv"
HEADER = [
    "link",
    "category",
    "title",
    "abstract",
    "keywords",
    "ccs_concepts",
    "author_names",
    "author_affiliations",
    "author_countries"
]

def get_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    title_tag = soup.find("h2", class_="citation_title")
    if title_tag:
        return title_tag.text.strip()
    return ""

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
    return ""

def parse_authors_from_meta(filename):
    with open(filename, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    author_metas = soup.find_all("meta", attrs={"name": "citation_author"})
    return [m.get("content", "").strip() for m in author_metas]

def get_author_affiliations(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")
    base_link = "https://openreview.net/"
    profile_links = [
        base_link + a.get("href")
        for a in soup.find_all("a") 
        if a.get("href") and "/profile?id" in a.get("href")
    ]
    affiliations = []
    for profile_link in profile_links:
        resp = requests.get(profile_link)
        profile_soup = BeautifulSoup(resp.text, "html.parser")
        inst = profile_soup.find("div", class_="institution")
        if inst:
            affiliation = inst.text.strip()
            affiliation = re.sub(r"\s*\(\S*?\.\S*?\)$", "", affiliation).strip()
            affiliations.append(affiliation)
        else:
            affiliations.append("")
    return affiliations

def get_category(title, abstract):
    for category, keywords in KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in title.lower() or kw.lower() in abstract.lower():
                return category
    return None

def save_html(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(response.text)

def initialize_csv():
    if not os.path.exists(os.path.dirname(CSV_FILE)):
        os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    if not os.path.isfile(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

def get_processed_links():
    processed = set()
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed.add(row["link"])
    return processed

if __name__ == "__main__":
    initialize_csv()
    processed_links = get_processed_links()
    
    with open("links/neurips_openreview_links.txt", "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    for i in range(len(links)):
        link = links[i]

        if link in processed_links:
            print(f"Already processed: {link}")
            continue
        
        save_html(link, "html/neurips_html.html")
        title = get_title(link)
        abstract = parse_abstract_from_meta("html/neurips_html.html")
        authors = parse_authors_from_meta("html/neurips_html.html")
        author_affiliations = get_author_affiliations(link)
        
        category = get_category(title, abstract)
        if category:
            paper = {
                "link": link,
                "category": category,
                "title": title,
                "abstract": abstract,
                "keywords": [], 
                "ccs_concepts": "",
                "author_names": authors,
                "author_affiliations": author_affiliations,
                "author_countries": ""
            }
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    paper["link"],
                    paper["category"],
                    paper["title"],
                    paper["abstract"],
                    paper["keywords"],       
                    paper["ccs_concepts"],  
                    paper["author_names"],  
                    paper["author_affiliations"],
                    paper["author_countries"]
                ])
            processed_links.add(link)
            print(f"Saved: {link}, paper # {i}")
        else:
            print(f"Invalid paper: {link}, paper # {i}")

        time.sleep(random.randint(2, 4))