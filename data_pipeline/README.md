# Module 1: Data Engineering Pipeline (`/data_pipeline`)

## Overview
This module scrapes product catalog data from `books.toscrape.com`, cleans raw fields, applies currency conversion, loads data into a 2-table relational SQLite database, and runs verification SQL/Pandas queries.

## Key Design Decisions
- **Embedded Database Schema**: The database schema is embedded inside `pipeline.py` using `CREATE TABLE IF NOT EXISTS` statements.
- **Fixed Currency Conversion Rate**: **1 GBP = 105.50 INR** (fixed project baseline rate).
- **Data Cleaning Rules**:
  - `price_gbp`: Extracted numeric float from text using regex.
  - `rating`: Text converted to integer (1–5). Unknown values default to `3`.
  - `in_stock`: Text parsed to binary integer (`1` for in stock, `0` for out of stock).
  - Bad rows with missing prices or titles are safely dropped.

## Execution Steps (Windows)
```cmd
python data_pipeline\scraper.py
python data_pipeline\pipeline.py
python data_pipeline\query_and_compare.py