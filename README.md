# Q&A Support Bot (RAG)

A **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain**, **LangGraph**, **FAISS**, and **FastAPI**.  
This bot answers user questions **only using documents you provide**, preventing hallucinations and ensuring grounded responses.

## 📝 Features

- Crawl and ingest documents from websites or local sources
- Clean and chunk text for vector embedding
- Store embeddings in **FAISS vector database**
- Retrieve relevant context using semantic search
- Generate answers with **OpenAI LLMs** grounded in retrieved context
- FastAPI endpoint to serve the bot via HTTP
- Prevents hallucinations; answers only from context

## ⚙️ Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3.13 | Language |
| FastAPI | REST API backend |
| LangChain | RAG workflow (LLM, retriever, prompt) |
| LangGraph | RAG graph orchestration |
| FAISS | Vector database for embeddings |
| OpenAI LLM | Generate answers |
| BeautifulSoup & Playwright | Website crawling |
| python-dotenv | Load secrets from `.env` |

## 📂 Project Structure
Rag_project/
├─ app/
│  ├─ api.py            # FastAPI endpoints
│  ├─ ingest.py         # Crawling, cleaning, embeddings
│  ├─ rag_graph.py      # LangGraph pipeline
│  ├─ config.py         # Load .env variables
│  └─ init.py
├─ data/
│  └─ faiss_index/      # Saved FAISS index
├─ .env                 # Contains OPENAI_API_KEY (never commit!)
├─ .gitignore           # .env is ignored
├─ requirements.txt
└─ README.md


## ⚡ Setup & Installation

1. **Clone the repo**

```bash
git clone https://github.com/your-username/Q-A_RAG_BOT.git
cd Q-A_RAG_BOT
```
2.	Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

3.	Install dependencies
```bash
  pip install -r requirements.txt
```
4.	Create .env file
 ```bash
  touch .env
```

Add your OpenAI key:
```bash
  OPENAI_API_KEY=sk-your-new-key-here
```
runig the code
```bash
  # run ingestion properly
python -m app.ingest

python main.py
```



Here are we Screen shots 

<img width="1470" height="258" alt="Screenshot 2026-02-01 at 6 53 18 PM" src="https://github.com/user-attachments/assets/b8b48067-6ae3-41e8-87dc-838750a6be76" />
<img width="1470" height="74" alt="Screenshot 2026-02-01 at 6 53 41 PM" src="https://github.com/user-attachments/assets/14809b5e-1542-4020-9aca-106a11b42978" />

