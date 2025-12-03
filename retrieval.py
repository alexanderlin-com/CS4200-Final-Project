# retrieval.py
import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

REQUIRED_VARS = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"]
missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
if missing:
    raise SystemExit(f"[FATAL] Missing env vars: {', '.join(missing)}")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise SystemExit(f"[FATAL] Index '{PINECONE_INDEX_NAME}' does not exist.")

index = pc.Index(PINECONE_INDEX_NAME)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# retriever for “normal” semantic queries
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.5},
)

def debug_query(q: str):
    print(f"\n[QUERY] {q}\n")
    docs = retriever.invoke(q)

    # sort by story order if available
    docs_sorted = sorted(
        docs,
        key=lambda d: (d.metadata or {}).get("chunk_index", 0),
    )

    for i, d in enumerate(docs_sorted, 1):
        meta = d.metadata or {}
        idx = meta.get("chunk_index", "?")
        filename = meta.get("filename", meta.get("source", "unknown"))
        print(f"--- RESULT {i} (chunk_index={idx}, file={filename}) ---")
        print(d.page_content[:600].replace("\n", " ") + "...\n")

if __name__ == "__main__":
    print("Story retrieval debug. Ctrl+C or empty line to quit.")
    while True:
        try:
            q = input("\nAsk something about 'Me and the Boys': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break
        if not q:
            print("[INFO] Exiting.")
            break
        debug_query(q)
