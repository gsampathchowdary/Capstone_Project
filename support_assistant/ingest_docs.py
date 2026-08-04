import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

def run_ingestion():
    print("[INGESTION] Reading Zepto policy corpus files...")
    doc_files = sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt")))
    
    if not doc_files:
        raise FileNotFoundError(f"No document files found in '{DOCS_DIR}'.")

    documents = []
    for file_path in doc_files:
        doc_id = os.path.basename(file_path).replace(".txt", "")
        loader = TextLoader(file_path, encoding="utf-8")
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["doc_id"] = doc_id
            doc.metadata["source"] = doc_id
            documents.append(doc)

    print(f"[INGESTION] Loaded {len(documents)} policy documents.")

    print("[INGESTION] Generating embeddings using sentence-transformers/all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store embeddings in persistent ChromaDB collection
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="zepto_policies"
    )
    
    print(f"[SUCCESS] Vector store populated and persisted at '{CHROMA_DIR}'.")

if __name__ == "__main__":
    run_ingestion()