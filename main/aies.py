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


def get_paper_links(url):
    """
    Given a URL, scrapes and returns a list of paper links.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }

    try:
        session = requests.Session()
        response = session.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        exit(1)

    soup = BeautifulSoup(response.text, "html.parser")

    # On the AIES website, paper links are found in
    # <h3> tags with class "title" -> nested <a> tag -> href attribute
    h3_tags = soup.find_all("h3", class_="title")
    href_list = []

    for h3 in h3_tags:
        a_tag = h3.find("a")
        if a_tag and a_tag.get("href"):
            href_list.append(a_tag["href"])

    return href_list


def get_paper_attributes(url):
    """
    Given a paper URL, scrapes and returns a dictionary of attributes:
      - link
      - category
      - title
      - abstract
      - keywords (list)
      - ccs_concepts
      - author_names (list)
      - author_affiliations (list)
      - author_countries (list)
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the URL {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    
    attributes = {}
    attributes["link"] = url

    # placeholder
    cat_meta = soup.find("meta", {"name": "citation_section"})
    attributes["category"] = cat_meta.get("content", "") if cat_meta else ""

    # extract title from meta tag
    title_meta = soup.find("meta", {"name": "citation_title"})
    if title_meta:
        attributes["title"] = title_meta.get("content", "")
    else:
        title_tag = soup.find("h1")
        attributes["title"] = title_tag.get_text(strip=True) if title_tag else ""

    # extract abstract from <section> tag with class "item abstract"
    abstract_section = soup.find("section", class_="item abstract")
    if abstract_section:
        # remove header element (if any) inside the abstract section
        header = abstract_section.find(['h2', 'h3'])
        if header:
            header.extract()
        attributes["abstract"] = abstract_section.get_text(" ", strip=True).replace("\n", " ")
    else:
        attributes["abstract"] = ""

    # extract keywords from <section> tag with class "item keywords"
    keywords_section = soup.find("section", class_="item keywords")
    keywords = []
    if keywords_section:
        span_value = keywords_section.find("span", class_="value")
        if span_value:
            raw_text = span_value.get_text(" ", strip=True)
            # split by commas and strip each keyword
            keywords = [kw.strip() for kw in raw_text.split(",") if kw.strip()]
    attributes["keywords"] = keywords

    # extract ccs concepts from <section> tag with class "item ccs"
    ccs_section = soup.find("section", class_="item ccs")
    if ccs_section:
        span_value = ccs_section.find("span", class_="value")
        attributes["ccs_concepts"] = span_value.get_text(" ", strip=True) if span_value else ccs_section.get_text(" ", strip=True)
    else:
        attributes["ccs_concepts"] = ""

    # extract author information from meta tags
    author_names = [meta.get("content", "").strip() for meta in soup.find_all("meta", {"name": "citation_author"})]
    author_affiliations = [meta.get("content", "").strip() for meta in soup.find_all("meta", {"name": "citation_author_institution"})]
    author_countries = [meta.get("content", "").strip() for meta in soup.find_all("meta", {"name": "citation_author_country"})]
    
    attributes["author_names"] = author_names
    attributes["author_affiliations"] = author_affiliations
    attributes["author_countries"] = author_countries

    return attributes


def valid_paper(paper):
    """
    Determines whether the paper is valid based on the presence of any keywords
    in the title, abstract, keywords or CCS concepts. If a match is found, updates
    the paper's category accordingly.
    """
    for category, keywords in KEYWORDS.items():
        # check title and abstract (case-insensitive)
        if any(keyword.lower() in paper.get("title", "").lower() for keyword in keywords) \
           or any(keyword.lower() in paper.get("abstract", "").lower() for keyword in keywords) \
           or any(keyword.lower() in " ".join(paper.get("keywords", [])).lower() for keyword in keywords) \
           or any(keyword.lower() in paper.get("ccs_concepts", "").lower() for keyword in keywords):
            paper["category"] = category  # set the category
            return True
    return False


def write_to_csv(paper):
    """
    Writes the paper to a CSV file.
    """
    with open("data/aies_papers.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(paper.values())


if __name__ == "__main__":
    issue_url = "https://ojs.aaai.org/index.php/AIES/issue/view/609"
    print(f"collecting papers from {issue_url}...")

    # get all paper links from the issue page
    paper_links = get_paper_links(issue_url)
    print(f"found a total of {len(paper_links)} paper links.")

    papers = []
    # multi-threading the code to make it faster
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # map the get_paper_attributes function to all paper links concurrently
        future_to_url = {executor.submit(get_paper_attributes, url): url for url in paper_links}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                paper_data = future.result()
                if paper_data:
                    print(f"collected paper: {paper_data.get('title', 'No Title')}")
                    papers.append(paper_data)
            except Exception as exc:
                print(f"error processing {url}: {exc}")

    header_row = ["link", "category", "title", "abstract", "keywords", "ccs_concepts", "author_names", "author_affiliations", "author_countries"]
    with open("data/aies_papers.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_row)

    # process the collected papers and write valid ones to CSV
    for paper in papers:
        if valid_paper(paper):
            print(f"found valid paper: {paper.get('title', 'No Title')}")
            write_to_csv(paper)

    print("finished collecting and writing to csv!")