# deingestion.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

REQUIRED_VARS = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
if missing:
    raise SystemExit(f"[FATAL] Missing env vars: {', '.join(missing)}")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    raise SystemExit(f"[FATAL] Index '{PINECONE_INDEX_NAME}' does not exist.")

index = pc.Index(PINECONE_INDEX_NAME)

print(f"[WARN] You are about to delete ALL vectors from index: '{PINECONE_INDEX_NAME}'")
print("The index itself will remain, but all stored embeddings/documents will be gone.\n")

confirm = input("Type 'DELETE' to confirm: ").strip()

if confirm != "DELETE":
    print("[INFO] Aborted. No vectors were deleted.")
    raise SystemExit(0)

print("[INFO] Deleting all vectors in the index...")
# This deletes all vectors but keeps the index
index.delete(delete_all=True)

print("[INFO] Deletion request sent. All vectors in the index have been removed.")
