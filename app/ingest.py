from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.config import OPENAI_API_KEY

def ingest_site(url: str):
    print("🔍 Loading website...")
    loader = WebBaseLoader(url)
    docs = loader.load()

    print("✂️ Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    print("🧠 Generating embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("💾 Saving FAISS index...")
    vectorstore.save_local("data/faiss_index")

    print("✅ Ingestion complete")

if __name__ == "__main__":
    ingest_site("https://en.wikipedia.org/wiki/Swami_Vivekananda")