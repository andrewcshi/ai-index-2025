import requests
from bs4 import BeautifulSoup

def get_title(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        title_span = soup.find("h1", property="name")
        if title_span:
            return title_span.get_text(strip=True)
    except Exception as e:
        print(f"Error in get_title for {url}: {e}")
    return ""

print(get_title("https://dl.acm.org/doi/10.1145/3630106.3659049"))