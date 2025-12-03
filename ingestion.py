# ingestion.py
import os
import time
import uuid

from dotenv import load_dotenv
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# ENV + CONFIG
# ---------------------------------------------------------
load_dotenv()

REQUIRED_VARS = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"]
missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
if missing:
    raise SystemExit(f"[FATAL] Missing env vars: {', '.join(missing)}")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

BASE_DIR = "documents"
STORY_FOLDER = os.path.join(BASE_DIR, "pdfs")  # put story + notes here
BATCH_SIZE = 100

# ---------------------------------------------------------
# PINECONE SETUP
# ---------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"[INFO] Creating index '{PINECONE_INDEX_NAME}' ...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,  # text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("[INFO] Waiting for index to be ready...")
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ---------------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------------
if not os.path.isdir(STORY_FOLDER):
    raise SystemExit(f"[FATAL] Story directory not found: {STORY_FOLDER}")

print(f"[INFO] Loading documents from {STORY_FOLDER}")
raw_docs = []

# PDFs
pdf_loader = DirectoryLoader(
    STORY_FOLDER,
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)
raw_docs.extend(pdf_loader.load())

# Markdown
md_loader = DirectoryLoader(
    STORY_FOLDER,
    glob="*.md",
    loader_cls=TextLoader,
)
raw_docs.extend(md_loader.load())

# Plain text
txt_loader = DirectoryLoader(
    STORY_FOLDER,
    glob="*.txt",
    loader_cls=TextLoader,
)
raw_docs.extend(txt_loader.load())

if not raw_docs:
    raise SystemExit("[FATAL] No supported files found in documents/pdfs/")

print(f"[INFO] Loaded {len(raw_docs)} source documents.")

# ---------------------------------------------------------
# SPLIT + METADATA
# ---------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

print("[INFO] Splitting into chunks...")
docs = splitter.split_documents(raw_docs)
print(f"[INFO] Produced {len(docs)} chunks.")

# assign ordered chunk indices + standard metadata
for i, d in enumerate(docs):
    d.metadata.setdefault("source_category", "me_and_the_boys")
    filename = os.path.basename(d.metadata.get("source", ""))
    d.metadata["filename"] = filename
    d.metadata["chunk_index"] = i  # global story order

# quick sanity peek
print("\n[DEBUG] First 3 chunks:\n")
for i, d in enumerate(docs[:3], 1):
    print(f"--- CHUNK {i} (index={d.metadata.get('chunk_index')}) ---")
    print(d.page_content[:600].replace("\n", " ") + "...\n")

print("\n[DEBUG] Last 3 chunks:\n")
for i, d in enumerate(docs[-3:], 1):
    idx = d.metadata.get("chunk_index")
    print(f"--- CHUNK {len(docs) - 3 + i} (index={idx}) ---")
    print(d.page_content[:600].replace("\n", " ") + "...\n")

# ---------------------------------------------------------
# INGEST
# ---------------------------------------------------------
print("[INFO] Ingesting into Pinecone...")

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

for batch in tqdm(batched(docs, BATCH_SIZE), total=(len(docs) // BATCH_SIZE) + 1):
    ids = [str(uuid.uuid4()) for _ in batch]
    vector_store.add_documents(documents=batch, ids=ids)

print("[INFO] Ingestion complete. Lore uploaded.")
