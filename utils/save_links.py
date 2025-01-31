import requests
from bs4 import BeautifulSoup
import concurrent.futures

def fetch_openreview_link(relative_link, base_url="https://nips.cc"):
    """
    Given a relative link, tries to find the OpenReview link on that page.
    Returns the OpenReview link if found, otherwise None.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(base_url + relative_link, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {base_url + relative_link}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    # The 'href_URL' button usually holds the OpenReview link.
    openreview_anchor = soup.find("a", class_="btn btn btn-outline-dark btn-sm href_URL")
    if openreview_anchor and openreview_anchor.get("href"):
        return openreview_anchor.get("href")

    return None

def save_openreview_links():
    """
    Scrapes the main page for relative paper links, fetches each in parallel,
    and writes the found OpenReview links to a file immediately.
    """
    url = "https://nips.cc/virtual/2024/papers.html?filter=titles"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/117.0.0.0 Safari/537.36"
        )
    }

    # Fetch the main page.
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching the main NeurIPS page '{url}': {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    # Find all relative links that start with '/virtual/2024/'.
    anchors = soup.select("li a[href^='/virtual/2024/']")
    relative_links = [a.get("href") for a in anchors if a.get("href")]

    print(f"Found {len(relative_links)} paper links on the main page.")
    
    # We'll keep track of how many links we've actually found.
    found_count = 0

    # Open the file once in 'w' mode. Each time we get a valid link, we'll write immediately.
    with open("neurips_openreview_links.txt", "w", encoding="utf-8") as outfile:
        # Use a thread pool to process links concurrently.
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_link = {
                executor.submit(fetch_openreview_link, link): link
                for link in relative_links
            }
            for future in concurrent.futures.as_completed(future_to_link):
                link = future_to_link[future]
                try:
                    result = future.result()
                    if result:
                        found_count += 1
                        outfile.write(result + "\n")
                        # Flush each time so it's saved to disk immediately
                        outfile.flush()
                        print(f"Found OpenReview link for {link} (total so far: {found_count})")
                except Exception as e:
                    print(f"Error processing {link}: {e}")

    print(f"Done. A total of {found_count} OpenReview links were saved to neurips_openreview_links.txt.")

if __name__ == "__main__":
    save_openreview_links()
