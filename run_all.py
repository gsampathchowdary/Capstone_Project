import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASTER_LOG = os.path.join(OUTPUT_DIR, "master_execution.log")
FINAL_MD = os.path.join(OUTPUT_DIR, "FINAL_PROJECT_OUTPUT.md")

# Ensure UTF-8 execution environment for Windows subprocesses
SUBPROCESS_ENV = os.environ.copy()
SUBPROCESS_ENV["PYTHONUTF8"] = "1"
SUBPROCESS_ENV["PYTHONIOENCODING"] = "utf-8"

commands = [
    ("Module 1: Scraping", [sys.executable, "data_pipeline/scraper.py"]),
    ("Module 1: ETL Pipeline", [sys.executable, "data_pipeline/pipeline.py"]),
    ("Module 1: Queries and Verification", [sys.executable, "data_pipeline/query_compare.py"]),
    ("Module 2: EDA and Profiling", [sys.executable, "analytics/01_eda.py"]),
    ("Module 2: ML Model Training and Tuning", [sys.executable, "analytics/02_modeling.py"]),
    ("Module 2: Pipeline Reload Verification", [sys.executable, "analytics/test_pipeline_reload.py"]),
    ("Module 3: Policy Ingestion", [sys.executable, "support_assistant/ingest_docs.py"]),
    ("Module 3: Assistant Graph Execution", [sys.executable, "support_assistant/test_assistant.py"])
]


def run_project():
    print("=================================================================")
    print("       STARTING ZEPTO DATA AND AI PLATFORM AUTOMATED RUN         ")
    print("=================================================================\n")

    with open(MASTER_LOG, "w", encoding="utf-8") as master_f:
        for title, cmd in commands:
            print(f">>> Running: {title} ...")
            master_f.write(f"\n{'='*60}\n{title}\n{'='*60}\n")
            master_f.flush()

            process = subprocess.Popen(
                cmd,
                cwd=ROOT_DIR,
                env=SUBPROCESS_ENV,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace"
            )

            for line in process.stdout:
                sys.stdout.write(line)
                master_f.write(line)

            process.wait()
            if process.returncode != 0:
                print(f"[ERROR] Step '{title}' failed with exit code {process.returncode}")
                return

    print("\n=================================================================")
    print("              COMPILING FINAL COMPREHENSIVE REPORT               ")
    print("=================================================================\n")

    p1_log = ""
    p1_file = os.path.join(ROOT_DIR, "data_pipeline", "outputs", "pipeline_output.txt")
    if os.path.exists(p1_file):
        with open(p1_file, "r", encoding="utf-8") as f:
            p1_log = f.read()

    p2_log = ""
    p2_file = os.path.join(ROOT_DIR, "analytics", "outputs", "analytics_output.txt")
    if os.path.exists(p2_file):
        with open(p2_file, "r", encoding="utf-8") as f:
            p2_log = f.read()

    p3_log = ""
    p3_file = os.path.join(ROOT_DIR, "support_assistant", "outputs", "rag_output.json")
    if os.path.exists(p3_file):
        with open(p3_file, "r", encoding="utf-8") as f:
            p3_log = f.read()

    with open(FINAL_MD, "w", encoding="utf-8") as f:
        f.write("# Zepto Data & AI Platform - Master Execution & Output Summary\n\n")
        f.write("This document aggregates all terminal outputs, database query results, machine learning metric comparisons, and generated visualization charts.\n\n")

        f.write("## 1. Module 1: Data Pipeline Execution Outputs\n\n")
        f.write("```text\n")
        f.write(p1_log)
        f.write("\n```\n\n")

        f.write("## 2. Module 2: Analytics & Predictive Modeling Outputs\n\n")
        f.write("### Visual Chart Artifacts\n\n")
        f.write("#### Univariate Distributions (Age and Fare)\n")
        f.write("![Age Fare](../analytics/charts/age_fare_distributions.png)\n\n")
        f.write("#### 6x6 Feature Correlation Heatmap\n")
        f.write("![Correlation](../analytics/charts/correlation_heatmap.png)\n\n")
        f.write("#### Multivariate Story Plots\n")
        f.write("| Survival by Class and Sex | Fare by Class and Survival |\n")
        f.write("|---|---|\n")
        f.write("| ![Chart 1](../analytics/charts/chart1_survival_sex_pclass.png) | ![Chart 2](../analytics/charts/chart2_fare_by_class_survival.png) |\n")
        f.write("| Age vs Fare Scatter | Family Size Survival |\n")
        f.write("| ![Chart 3](../analytics/charts/chart3_age_vs_fare_scatter.png) | ![Chart 4](../analytics/charts/chart4_family_size_survival.png) |\n\n")
        f.write("#### Decision Tree Plot\n")
        f.write("![Decision Tree](../analytics/charts/decision_tree.png)\n\n")
        f.write("#### Fare Regression Residual Plot\n")
        f.write("![Residual Plot](../analytics/charts/fare_regression_residuals.png)\n\n")
        f.write("### Raw Analytics Statistics and Metrics Log\n\n")
        f.write("```text\n")
        f.write(p2_log)
        f.write("\n```\n\n")

        f.write("## 3. Module 3: GenAI Support Assistant Outputs\n\n")
        f.write("```json\n")
        f.write(p3_log)
        f.write("\n```\n\n")

    print("[SUCCESS] All outputs and images unified in:")
    print(f" -> Master Log File: {MASTER_LOG}")
    print(f" -> Consolidated Markdown Report: {FINAL_MD}\n")


if __name__ == "__main__":
    run_project()