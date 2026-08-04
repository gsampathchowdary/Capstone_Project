import os
import re
import json
import sqlite3
import pandas as pd

# File paths
BASE_DIR = os.path.dirname(__file__)
RAW_JSON_PATH = os.path.join(BASE_DIR, "data", "raw_books.json")
DB_PATH = os.path.join(BASE_DIR, "data", "zepto_catalog.db")

# Fixed baseline conversion rate (Required)
FIXED_GBP_TO_INR = 105.50

# Map star rating text to numbers
STAR_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

# -------------------------------------------------------------
# EMBEDDED SQL SCHEMA DEFINITION (2-Table PK/FK Relationship)
# -------------------------------------------------------------
CREATE_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Table 1: Categories
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category_name TEXT UNIQUE NOT NULL
);

-- Table 2: Books (References categories via foreign key)
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL CHECK(price_gbp >= 0),
    price_inr REAL NOT NULL CHECK(price_inr >= 0),
    rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE CASCADE
);
"""

def clean_data(raw_records):
    """Cleans raw scraped records and converts currency."""
    cleaned_list = []

    for item in raw_records:
        try:
            # 1. Clean price GBP
            price_match = re.search(r"[\d\.]+", item.get("price_raw", ""))
            if not price_match:
                continue  # Drop row if price missing
            price_gbp = float(price_match.group())

            # 2. Convert GBP to INR using fixed rate
            price_inr = round(price_gbp * FIXED_GBP_TO_INR, 2)

            # 3. Convert rating text to integer (default to 3 if missing)
            rating_text = str(item.get("star_rating_raw", "")).capitalize()
            rating = STAR_MAP.get(rating_text, 3)

            # 4. Convert availability to boolean int (1/0)
            avail_text = item.get("availability_raw", "").lower()
            in_stock = 1 if "in stock" in avail_text else 0

            # 5. Title & Category
            title = item.get("title", "").strip()
            category = item.get("category", "General").strip()

            if not title:
                continue

            cleaned_list.append({
                "title": title,
                "price_gbp": price_gbp,
                "price_inr": price_inr,
                "rating": rating,
                "in_stock": in_stock,
                "category": category
            })
        except Exception:
            continue  # Gracefully skip any corrupt row

    return cleaned_list

def run_pipeline():
    """Main ETL Function."""
    print("[ETL] Starting Data Processing Pipeline...")

    # Load raw JSON
    if not os.path.exists(RAW_JSON_PATH):
        raise FileNotFoundError(f"Raw data file not found at '{RAW_JSON_PATH}'. Run scraper.py first.")

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Clean data
    cleaned_records = clean_data(raw_data)
    print(f"[ETL] Cleaned {len(cleaned_records)} records out of {len(raw_data)} raw records.")

    # Connect to SQLite and create schema
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(CREATE_SCHEMA_SQL)

    # Insert Categories
    df = pd.DataFrame(cleaned_records)
    unique_categories = df["category"].unique()
    for cat_name in unique_categories:
        cursor.execute("INSERT OR IGNORE INTO categories (category_name) VALUES (?)", (cat_name,))
    conn.commit()

    # Map Category Names to IDs
    cat_map = dict(cursor.execute("SELECT category_name, category_id FROM categories").fetchall())

    # Insert Books
    for _, row in df.iterrows():
        cursor.execute(
            """INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["title"], row["price_gbp"], row["price_inr"], row["rating"], row["in_stock"], cat_map[row["category"]])
        )

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Database created and populated at '{DB_PATH}'.")

if __name__ == "__main__":
    run_pipeline()