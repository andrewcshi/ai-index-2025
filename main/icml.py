import time
import threading
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

# Global rate-limiting variables and lock.
rate_lock = threading.Lock()
last_request_time = 0

def rate_limit():
    """
    Ensures that at least 1 second has passed since the previous request.
    This function is thread-safe.
    """
    global last_request_time
    with rate_lock:
        now = time.time()
        elapsed = now - last_request_time
        if elapsed < 1:
            # Sleep until 1 second has passed.
            time.sleep(1 - elapsed)
        last_request_time = time.time()


def fetch_url(url, headers, max_retries=3):
    """
    Attempts to GET the given URL with the provided headers.
    If a 429 status is returned, it will sleep for the time specified
    in the Retry-After header (or 5 seconds if missing), and retry.
    Uses the rate_limit() function to limit to ~1 request per second.
    Returns a response object on success or None on failure.
    """
    attempt = 0
    while attempt < max_retries:
        try:
            rate_limit()  # Enforce rate limiting
            response = requests.get(url, headers=headers)
            # If the request was successful, return the response.
            if response.status_code == 200:
                return response
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                print(f"429 error for {url}. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
                attempt += 1
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    print(f"Giving up on {url} after {max_retries} attempts.")
    return None


def process_intermediate(relative_link, base_url="https://icml.cc", headers=None):
    """
    Given a relative link (e.g. "/virtual/2024/poster/33112"), this function performs:
      1. Constructs the intermediate URL.
      2. Loads that page and finds the anchor with text "Paper PDF".
      3. Loads the "Paper PDF" page and finds the anchor with text "OpenReview".
      4. Returns the final OpenReview link.
    Returns None if any step fails.
    """
    if headers is None:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            )
        }

    # Construct the intermediate URL.
    intermediate_url = base_url + relative_link
    resp_intermediate = fetch_url(intermediate_url, headers=headers)
    if not resp_intermediate:
        return None

    soup_intermediate = BeautifulSoup(resp_intermediate.text, "html.parser")
    pdf_anchor = soup_intermediate.find("a", string="Paper PDF")
    if not (pdf_anchor and pdf_anchor.get("href")):
        print(f"'Paper PDF' link not found on page: {intermediate_url}")
        return None

    pdf_link = pdf_anchor.get("href")
    if not pdf_link.startswith("http"):
        pdf_link = base_url + pdf_link

    resp_pdf = fetch_url(pdf_link, headers=headers)
    if not resp_pdf:
        return None

    soup_pdf = BeautifulSoup(resp_pdf.text, "html.parser")
    openreview_anchor = soup_pdf.find("a", string="OpenReview")
    if not (openreview_anchor and openreview_anchor.get("href")):
        print(f"'OpenReview' link not found on page: {pdf_link}")
        return None

    final_link = openreview_anchor.get("href")
    return final_link


def get_paper_links(url):
    """
    Given the ICML papers URL, scrapes and returns a list of final OpenReview links.
    Process:
      1. Load the main page.
      2. Select all <a> tags within <li> elements whose href begins with '/virtual/2024/'.
      3. For each, use process_intermediate() to follow the request chain.
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
        resp_main = requests.get(url, headers=headers)
        resp_main.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the main URL: {e}")
        exit(1)
    
    soup = BeautifulSoup(resp_main.text, "html.parser")
    anchors = soup.select("li a[href^='/virtual/2024/']")
    relative_links = [a.get("href") for a in anchors if a.get("href")]

    final_links = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
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


def get_paper_attributes(url):
    """
    Given an OpenReview paper URL, scrapes and returns a dictionary of attributes:
      - link: The OpenReview URL.
      - title: The paper title.
      - abstract: The paper abstract (obtained from the <p> tag immediately after the <strong> with text "Abstract").
      - author_names: A list of authors (obtained from the <p> tag immediately after a <strong> containing "author").
    
    Other keys are set to empty defaults.
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
        print(f"Error fetching OpenReview URL {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    attributes = {"link": url}

    # Extract the title.
    title_tag = soup.find("h2", class_="note_content_title")
    attributes["title"] = title_tag.get_text(strip=True) if title_tag else ""

    # Extract the abstract by finding the <strong> tag with text "Abstract"
    # then taking the immediately following <p> tag.
    strong_abstract = soup.find("strong", string=lambda s: s and s.strip().lower() == "abstract")
    if strong_abstract:
        abstract_p = strong_abstract.find_next_sibling("p")
        attributes["abstract"] = abstract_p.get_text(strip=True) if abstract_p else ""
    else:
        attributes["abstract"] = ""

    # Extract authors by finding a <strong> tag that contains "author" (case-insensitive)
    # then taking the immediately following <p> tag's text.
    strong_authors = soup.find("strong", string=lambda s: s and "author" in s.strip().lower())
    if strong_authors:
        authors_p = strong_authors.find_next_sibling("p")
        authors_str = authors_p.get_text(strip=True) if authors_p else ""
    else:
        authors_str = ""
    if authors_str:
        authors = [a.strip() for a in authors_str.split(",") if a.strip()]
    else:
        authors = []
    attributes["author_names"] = authors

    # Set remaining keys as empty/default.
    attributes["keywords"] = []
    attributes["ccs_concepts"] = ""
    attributes["author_affiliations"] = ""
    attributes["author_countries"] = ""
    attributes["category"] = ""

    return attributes


def valid_paper(paper):
    """
    Determines whether a paper is valid based on keyword filtering.
    Returns True if any keyword from a category is found in the title or abstract,
    and sets the paper's category accordingly.
    """
    for category, keywords in KEYWORDS.items():
        if any(keyword.lower() in paper.get("title", "").lower() for keyword in keywords) or \
           any(keyword.lower() in paper.get("abstract", "").lower() for keyword in keywords):
            paper["category"] = category
            return True
    return False


def write_to_csv(paper):
    """
    Appends the paper data as a row to a CSV file.
    """
    with open("data/icml_papers.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(paper.values())


if __name__ == "__main__":
    # Load links from the text file containing OpenReview links.
    with open("links/icml_openreview_links.txt", "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    print(f"Found {len(links)} links")

    # Write header row to CSV.
    header_row = ["link", "category", "title", "abstract", "keywords", "ccs_concepts", "author_names", "author_affiliations", "author_countries"]
    with open("data/icml_papers.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_row)

    for link in links:
        paper = get_paper_attributes(link)
        if paper and valid_paper(paper):
            write_to_csv(paper)
