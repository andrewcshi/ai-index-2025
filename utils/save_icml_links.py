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


def process_intermediate(relative_link, base_url="https://icml.cc", headers=None):
    """
    Given a relative link (e.g. "/virtual/2024/poster/33112"), this function performs
    the following steps:
      1. Constructs the intermediate URL.
      2. Loads that page and finds the anchor with text "Paper PDF".
      3. Loads the "Paper PDF" page and finds the anchor with text "OpenReview".
      4. Returns the final OpenReview link.
    If any step fails, returns None.
    """
    if headers is None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            )
        }

    # Step 1: Construct the intermediate URL and load the page.
    intermediate_url = base_url + relative_link
    try:
        resp_intermediate = requests.get(intermediate_url, headers=headers)
        resp_intermediate.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching intermediate link {intermediate_url}: {e}")
        return None

    soup_intermediate = BeautifulSoup(resp_intermediate.text, "html.parser")
    # Step 2: Find the anchor with text exactly "Paper PDF"
    pdf_anchor = soup_intermediate.find("a", string="Paper PDF")
    if not (pdf_anchor and pdf_anchor.get("href")):
        print(f"'Paper PDF' link not found on page: {intermediate_url}")
        return None

    pdf_link = pdf_anchor.get("href")
    # If pdf_link is relative, prepend the base_url.
    if not pdf_link.startswith("http"):
        pdf_link = base_url + pdf_link

    try:
        resp_pdf = requests.get(pdf_link, headers=headers)
        resp_pdf.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching PDF link {pdf_link}: {e}")
        return None

    soup_pdf = BeautifulSoup(resp_pdf.text, "html.parser")
    # Step 3: Find the anchor with text exactly "OpenReview"
    openreview_anchor = soup_pdf.find("a", string="OpenReview")
    if not (openreview_anchor and openreview_anchor.get("href")):
        print(f"'OpenReview' link not found on page: {pdf_link}")
        return None

    final_link = openreview_anchor.get("href")
    return final_link


def get_paper_links(url):
    """
    Given the ICML papers URL, scrapes and returns a list of final paper links.
    
    Process:
      1. Load the main page.
      2. Select all <li> elements containing an <a> tag whose href begins with '/virtual/2024/'.
      3. For each anchor, use process_intermediate() to follow the chain:
         - Main page relative link → intermediate URL,
         - From that page, get "Paper PDF" URL,
         - Then from that page, get "OpenReview" URL.
      4. Return the list of final OpenReview links.
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
        print(f"Error fetching the main URL: {e}")
        exit(1)
    
    soup = BeautifulSoup(response.text, "html.parser")
    # Select all <a> tags within <li> elements whose href starts with '/virtual/2024/'
    anchors = soup.select("li a[href^='/virtual/2024/']")
    # Get the relative links
    relative_links = [a.get("href") for a in anchors if a.get("href")]
    
    final_links = []
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_intermediate, rlink, "https://icml.cc", headers): rlink 
            for rlink in relative_links
        }
        for future in concurrent.futures.as_completed(futures):
            rlink = futures[future]
            try:
                result = future.result()
                if result:
                    final_links.append(result)
            except Exception as exc:
                print(f"Error processing {rlink}: {exc}")
    
    return final_links


if __name__ == "__main__":
    icml_url = "https://icml.cc/virtual/2024/papers.html?filter=titles"
    print(f"collecting papers from {icml_url}...")
    
    # Get all final OpenReview links.
    paper_links = get_paper_links(icml_url)
    print(f"found {len(paper_links)} final paper links:")
    
    # Save all OpenReview links to a text file.
    with open("icml_openreview_links.txt", "w", encoding="utf-8") as f:
        for link in paper_links:
            f.write(link + "\n")
    
    print("OpenReview links saved to icml_openreview_links.txt")