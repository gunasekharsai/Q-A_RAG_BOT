from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_graph import rag_app

app = FastAPI(title="RAG Support Bot")

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    result = rag_app.invoke({"question": q.question})
    return {"answer": result["answer"]}