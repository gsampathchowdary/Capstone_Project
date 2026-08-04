import os
import json
import requests
from bs4 import BeautifulSoup

# Base configuration
BASE_URL = "https://books.toscrape.com/catalogue/"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_books.json")

def scrape_books(total_pages=5):
    """Scrapes books across multiple catalog pages and saves raw JSON data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_books = []

    print(f"[SCRAPER] Starting to scrape {total_pages} pages...")

    for page_num in range(1, total_pages + 1):
        url = f"{BASE_URL}page-{page_num}.html"
        print(f" -> Fetching page {page_num}: {url}")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"    [Warning] Page {page_num} returned status code {response.status_code}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            book_cards = soup.find_all("article", class_="product_pod")

            for card in book_cards:
                # Extract basic attributes
                title = card.h3.a["title"]
                price_text = card.find("p", class_="price_color").text
                star_text = card.find("p", class_="star-rating")["class"][1]
                stock_text = card.find("p", class_="instock availability").text.strip()

                # Fetch category from the individual book page
                book_rel_url = card.h3.a["href"]
                detail_url = BASE_URL + book_rel_url if not book_rel_url.startswith("catalogue/") else "https://books.toscrape.com/" + book_rel_url

                category = "General"
                try:
                    detail_resp = requests.get(detail_url, timeout=5)
                    if detail_resp.status_code == 200:
                        detail_soup = BeautifulSoup(detail_resp.text, "html.parser")
                        crumbs = detail_soup.find("ul", class_="breadcrumb").find_all("li")
                        if len(crumbs) >= 3:
                            category = crumbs[2].text.strip()
                except Exception:
                    pass  # Keep 'General' if detail page request fails

                all_books.append({
                    "title": title,
                    "price_raw": price_text,
                    "star_rating_raw": star_text,
                    "availability_raw": stock_text,
                    "category": category
                })

        except Exception as err:
            print(f"    [Error] Failed scraping page {page_num}: {err}")

    # Save raw results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_books, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Scraped {len(all_books)} books. Output saved to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    scrape_books(total_pages=5)