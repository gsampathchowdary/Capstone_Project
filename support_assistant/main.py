import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_graph import build_support_graph, SupportResponse

app = FastAPI(
    title="Zepto GenAI Policy Support Assistant",
    description="Grounded RAG service for Zepto customer support policy queries.",
    version="1.0.0"
)

# Initialize compiled LangGraph app
graph_app = build_support_graph()

class AskRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "online", "service": "Zepto Support Assistant"}

@app.post("/ask", response_model=SupportResponse)
def ask_policy_question(request: AskRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    initial_state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "final_output": {}
    }

    try:
        final_state = graph_app.invoke(initial_state)
        output_dict = final_state.get("final_output", {})
        return SupportResponse(**output_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)