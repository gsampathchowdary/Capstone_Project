import os
import sys
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "data", "zepto_catalog.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pipeline_output.txt")


class TeeLogger:
    """Mirrors console prints directly into an output log file."""
    def __init__(self, filepath, mode="w"):
        self.terminal = sys.stdout
        self.logfile = open(filepath, mode, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


def run_queries():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = TeeLogger(OUTPUT_FILE, mode="w")
    sys.stdout = logger

    try:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Database file not found at {DB_PATH}. Run pipeline.py first.")

        conn = sqlite3.connect(DB_PATH)

        print("=========================================================")
        print("                EXECUTING SQL QUERIES                    ")
        print("=========================================================\n")

        # 1. SELECT / WHERE
        print("1. High-Rated Books with rating >= 4 [SELECT / WHERE]:")
        q1 = "SELECT title, price_gbp, price_inr, rating FROM books WHERE rating >= 4 LIMIT 5;"
        print(pd.read_sql_query(q1, conn).to_string(index=False), "\n")

        # 2. ORDER BY / LIMIT
        print("2. Most Expensive Books [ORDER BY / LIMIT]:")
        q2 = "SELECT title, price_gbp, price_inr FROM books ORDER BY price_gbp DESC LIMIT 5;"
        print(pd.read_sql_query(q2, conn).to_string(index=False), "\n")

        # 3. DISTINCT
        print("3. Available Rating Levels [DISTINCT]:")
        q3 = "SELECT DISTINCT rating FROM books ORDER BY rating ASC;"
        print(pd.read_sql_query(q3, conn).to_string(index=False), "\n")

        # 4. BETWEEN & IN
        print("4. Mid-Range Price with Rating 4 or 5 [BETWEEN / IN]:")
        q4 = "SELECT title, price_inr, rating, in_stock FROM books WHERE price_inr BETWEEN 2000 AND 5000 AND rating IN (4, 5) LIMIT 5;"
        print(pd.read_sql_query(q4, conn).to_string(index=False), "\n")

        # 5. JOIN
        print("5. Top Rated Books with Category Name [JOIN]:")
        q5 = """
        SELECT b.title, c.category_name, b.price_gbp, b.price_inr, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp ASC
        LIMIT 5;
        """
        df_sql = pd.read_sql_query(q5, conn)
        print(df_sql.to_string(index=False), "\n")

        print("=========================================================")
        print("        PANDAS pd.merge EQUIVALENCE CHECK               ")
        print("=========================================================\n")

        # Recreate JOIN query in memory using Pandas pd.merge
        df_books = pd.read_sql("SELECT * FROM books;", conn)
        df_categories = pd.read_sql("SELECT * FROM categories;", conn)

        df_pandas = pd.merge(df_books, df_categories, on="category_id", how="inner")

        # Sort and project identically
        df_pandas_result = (
            df_pandas[["title", "category_name", "price_gbp", "price_inr", "rating"]]
            .sort_values(by=["rating", "price_gbp"], ascending=[False, True])
            .head(5)
            .reset_index(drop=True)
        )

        print("Pandas pd.merge Output:")
        print(df_pandas_result.to_string(index=False), "\n")

        # Verify equivalence
        are_equal = df_sql.equals(df_pandas_result)
        print(f"Equivalence Verification: Both outputs match exactly -> {are_equal}")

        conn.close()

    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"\n[OUTPUT SAVED] Module 1 query outputs saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_queries()