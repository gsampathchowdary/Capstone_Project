import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class ZeptoPolicyRAGEngine:
    """
    Domain-grounded RAG Assistant using LangChain, ChromaDB, 
    and Hugging Face all-MiniLM-L6-v2 embeddings with Precision@3 evaluation.
    """
    def __init__(self, persist_dir="support_assistant/outputs/chroma_db"):
        self.persist_dir = persist_dir
        
        print("Initializing Hugging Face Sentence-Transformers embeddings (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 500-token recursive chunking with 10% overlap (50 tokens)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = None

    def ingest_policy_documents(self, raw_documents: list[dict]):
        """
        Ingests unstructured text documents, splits them into precise chunks, 
        and indexes them into ChromaDB.
        """
        docs = [
            Document(page_content=item["text"], metadata={"doc_id": item["id"], "category": item["category"]})
            for item in raw_documents
        ]
        
        chunks = self.text_splitter.split_documents(docs)
        print(f"Split {len(raw_documents)} documents into {len(chunks)} optimized chunks.")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Vector store successfully persisted at {self.persist_dir}")
        return len(chunks)

    def retrieve_relevant_context(self, query: str, k: int = 3):
        """
        Retrieves top-k relevant policy chunks using cosine similarity.
        """
        if not self.vector_store:
            self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def evaluate_retrieval_precision_at_k(self, evaluation_queries: list[dict], k: int = 3) -> float:
        """
        Evaluates retrieval quality, targeting high Precision@3 for policy search queries.
        """
        precisions = []
        for item in evaluation_queries:
            results = self.retrieve_relevant_context(item["query"], k=k)
            retrieved_ids = [doc.metadata.get("doc_id") for doc, _ in results]
            
            hits = sum(1 for doc_id in retrieved_ids if doc_id in item["relevant_ids"])
            precisions.append(hits / k)
            
        mean_precision = round(float(sum(precisions) / len(precisions)), 4) if precisions else 0.0
        print(f"RAG Evaluation Complete | Retrieval Precision@{k}: {mean_precision * 100}%")
        return mean_precision

if __name__ == "__main__":
    # Example execution test for the support assistant RAG engine
    rag = ZeptoPolicyRAGEngine()
    
    sample_docs = [
        {"id": "doc_01", "category": "refunds", "text": "Zepto refund policy states that items returned due to damage or spoilage within 24 hours are eligible for full instant wallet credit."},
        {"id": "doc_02", "category": "delivery", "text": "Delivery guarantees 10-minute grocery drops across metropolitan hubs. Late deliveries trigger a delivery fee waiver coupon."},
        {"id": "doc_03", "category": "subscriptions", "text": "Zepto Pass subscribers enjoy free delivery on all orders above 99 rupees and exclusive discounts on dairy products."}
    ]
    
    rag.ingest_policy_documents(sample_docs)
    
    eval_set = [
        {"query": "What is the refund process for damaged grocery items?", "relevant_ids": ["doc_01"]}
    ]
    score = rag.evaluate_retrieval_precision_at_k(eval_set, k=3)