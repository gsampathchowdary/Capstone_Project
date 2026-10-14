import os
import json
from rag_graph import build_support_graph

BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rag_output.json")

def test_and_save():
    print("=========================================================")
    print("      RUNNING SUPPORT ASSISTANT INGESTION AND TEST       ")
    print("=========================================================")

    app = build_support_graph()

    test_queries = [
        "What is the delivery fee for orders below INR 149?",
        "What is the capital of India?"
    ]

    all_results = []

    for query in test_queries:
        print(f"\nProcessing query: {query}")
        result = app.invoke({
            "query": query,
            "intent": "",
            "retrieved_chunks": [],
            "final_output": {}
        })
        
        output = {
            "query": query,
            "intent": result["intent"],
            "response": result["final_output"]
        }
        all_results.append(output)
        print(f"Result -> {json.dumps(output, indent=2)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OUTPUT SAVED] Support assistant test outputs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    test_and_save()