from fastapi import FastAPI, Query
from app.rag.engine import RAGEngine

app = FastAPI()

rag_engine = RAGEngine()

@app.get("/ask")
def query(question:str= Query(..., description= "User question")):
    """
    Return an LLM generated answer for the user question, grounded on the document provided.
    """
    try:
        print("hello here")
        answer= rag_engine.generate_answer(question)
        print("hello")
        return {"question": question, "answer": answer}

    except Exception as e:
        return {
            "question": question,
            "answer": None,
            "error": str(e)
        }