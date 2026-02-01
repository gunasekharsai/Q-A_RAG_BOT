from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app.config import OPENAI_API_KEY

# Load vector DB
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)

vectorstore = FAISS.load_local(
    "data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

prompt = ChatPromptTemplate.from_template("""
You are a customer support assistant.
Answer ONLY using the context below.
If the question asks for a summary,
summarize ONLY the information present in the context.
Do not add new facts.                                          
If the answer is not present, say:
"I don't know based on the provided information."

Context:
{context}

Question:
{question}
""")

class RAGState(TypedDict):
    question: str
    documents: List[Document]
    answer: str

def retrieve(state: RAGState):
    docs =  docs = retriever.invoke(state["question"])
    return {"documents": docs}

def has_docs(state: RAGState):
    return len(state["documents"]) > 0

def generate(state: RAGState):
    context = "\n\n".join(d.page_content for d in state["documents"])
    response = llm.invoke(
        prompt.format(context=context, question=state["question"])
    )
    return {"answer": response.content}

def fallback(state: RAGState):
    return {"answer": "I don't know based on the provided information."}

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("fallback", fallback)

graph.set_entry_point("retrieve")
graph.add_conditional_edges(
    "retrieve",
    has_docs,
    {True: "generate", False: "fallback"}
)

graph.add_edge("generate", END)
graph.add_edge("fallback", END)

rag_app = graph.compile()