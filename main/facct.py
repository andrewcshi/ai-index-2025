import csv
import time
import requests
from bs4 import BeautifulSoup
import os
import random
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
        'Explainable models'
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
        'Non-discrimination'
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
        'Trustworthy data curation'
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
        'Adversarial learning'
    ]
}

def valid_paper(paper):
    for category, keywords in KEYWORDS.items():
        text_title = paper.get("title", "").lower()
        text_abstract = paper.get("abstract", "").lower()
        text_keywords = " ".join(paper.get("keywords", [])).lower()
        text_ccs = paper.get("ccs_concepts", "").lower()
        for kw in keywords:
            kw_lower = kw.lower()
            if (
                kw_lower in text_title
                or kw_lower in text_abstract
                or kw_lower in text_keywords
                or kw_lower in text_ccs
            ):
                paper["category"] = category
                return True
    return False

def write_to_csv(paper, filename="data/facct_papers.csv"):
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            paper.get("link", ""),
            paper.get("category", ""),
            paper.get("title", ""),
            paper.get("abstract", ""),
            paper.get("keywords", []),
            paper.get("ccs_concepts", ""),
            paper.get("author_names", []),
            paper.get("author_affiliations", []),
            paper.get("author_countries", [])
        ])

def convert_doi_link(doi_link):
    base_url = "https://dl.acm.org/doi/fullHtml/10.1145/"
    doi_number = doi_link.split("/")[-1]
    return f"{base_url}{doi_number}"

def get_title(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title_span = soup.find("h1", property="name")
        if title_span:
            return title_span.get_text(strip=True)
    except:
        pass
    return ""

def get_abstract(url):
    try:
        page_response = requests.get(url)
        if page_response.status_code == 200:
            page_soup = BeautifulSoup(page_response.content, 'html.parser')
            abstract_tag = page_soup.find('div', role='paragraph')
            if abstract_tag:
                return abstract_tag.get_text().strip()
    except:
        pass
    return ""

def get_keywords(url):
    full_url = convert_doi_link(url)
    try:
        response = requests.get(full_url)
        soup = BeautifulSoup(response.text, "html.parser")
        keywords = []
        for span in soup.find_all("span", class_="keyword"):
            for small in span.find_all("small"):
                keywords.append(small.get_text(strip=True))
        return keywords
    except:
        return []

def get_ccs_concepts(url):
    full_url = convert_doi_link(url)
    try:
        response = requests.get(full_url)
        soup = BeautifulSoup(response.text, "html.parser")
        ccs_div = soup.find("div", class_="CCSconcepts")
        if ccs_div:
            strong_tags = ccs_div.find_all("strong")
            return " ".join([tag.get_text(strip=True).replace(";", "") for tag in strong_tags])
    except:
        pass
    return ""

def get_authors(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        first_names = soup.find_all("span", attrs={"property": "givenName"})
        last_names = soup.find_all("span", attrs={"property": "familyName"})
        affiliations = soup.find_all("span", attrs={"property": "name"})
        names = []
        for fn, ln in zip(first_names, last_names):
            name = f"{fn.get_text(strip=True)} {ln.get_text(strip=True)}"
            names.append(name)
        aff_texts = [aff.get_text(strip=True) for aff in affiliations]
        seen = set()
        unique_names = []
        for name in names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        output = {}
        for name, affiliation in zip(unique_names, aff_texts):
            output[name] = affiliation
        return output
    except:
        return {}

def get_facct_paper(url):
    paper = {}
    paper["link"] = url
    paper["title"] = get_title(url)
    paper["abstract"] = get_abstract(url)
    paper["keywords"] = get_keywords(url)
    paper["ccs_concepts"] = get_ccs_concepts(url)
    paper["category"] = ""
    authors_data = get_authors(url)
    paper["author_names"] = list(authors_data.keys())
    paper["author_affiliations"] = list(authors_data.values())
    paper["author_countries"] = []
    return paper

def get_already_processed_links(csv_filename="data/facct_papers.csv"):
    if not os.path.exists(csv_filename):
        return set()
    processed_links = set()
    with open(csv_filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                link = row[0]
                processed_links.add(link)
    return processed_links

if __name__ == "__main__":
    csv_filename = "data/facct_papers.csv"
    base_url = "https://doi.org/10.1145/3630106.365"
    links = [f"{base_url}{i}" for i in range(8537, 8550)]
    if not os.path.exists(csv_filename):
        header_row = [
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
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header_row)
    processed_links = get_already_processed_links(csv_filename)
    processed_count = 0
    for link in links:
        processed_count += 1
        if link in processed_links:
            continue
        paper_data = get_facct_paper(link)
        if valid_paper(paper_data):
            write_to_csv(paper_data, filename=csv_filename)
        processed_links.add(link)
        time.sleep(random.randint(40, 50))