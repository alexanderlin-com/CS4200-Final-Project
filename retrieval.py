# retrieval.py
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = os.environ["PINECONE_INDEX_NAME"]
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ["OPENAI_API_KEY"],
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever(
    search_type="mmr",  # better diversity than plain similarity
    search_kwargs={"k": 6, "fetch_k": 12, "lambda_mult": 0.5},
)

def debug_query(q: str):
    print(f"\n[QUERY] {q}\n")
    docs = retriever.invoke(q)
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        print(f"--- Result {i} ---")
        print(d.page_content[:500].strip(), "...")
        print("[META]", meta)
        print()

if __name__ == "__main__":
    while True:
        try:
            q = input("Ask about 'Me and the Boys' (blank to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        debug_query(q)
