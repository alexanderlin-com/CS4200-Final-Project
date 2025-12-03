# ingestion.py
import os
import time
import uuid
from dotenv import load_dotenv
from tqdm import tqdm

from pinecone import Pinecone, ServerlessSpec

# LangChain loaders & embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# LOAD ENVIRONMENT
# ---------------------------------------------------------
load_dotenv()

REQUIRED_VARS = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"]
missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
if missing:
    raise SystemExit(f"[FATAL] Missing env vars: {', '.join(missing)}")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# ---------------------------------------------------------
# INITIALIZE PINECONE
# ---------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"[INFO] Creating index '{PINECONE_INDEX_NAME}' ...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=3072,       # text-embedding-3-large
        metric="cosine",
        spec=ServerlessSpec(
            cloud=os.environ.get("PINECONE_CLOUD", "aws"),
            region=os.environ.get("PINECONE_REGION", "us-east-1"),
        ),
    )

    # Wait for index to become ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("[INFO] Waiting for index to be ready...")
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY,
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

# ---------------------------------------------------------
# LOAD DOCUMENTS
# ---------------------------------------------------------
BASE_DIR = "documents"
STORY_FOLDER = os.path.join(BASE_DIR, "pdfs")

if not os.path.isdir(STORY_FOLDER):
    raise SystemExit(f"[FATAL] Story directory not found: {STORY_FOLDER}")

print(f"[INFO] Loading documents from {STORY_FOLDER}")

raw_docs = []

# --- Load PDFs ---
pdf_loader = DirectoryLoader(
    STORY_FOLDER,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
raw_docs.extend(pdf_loader.load())

# --- Load Markdown ---
md_loader = DirectoryLoader(
    STORY_FOLDER,
    glob="*.md",
    loader_cls=TextLoader
)
raw_docs.extend(md_loader.load())

# --- Load TXT ---
txt_loader = DirectoryLoader(
    STORY_FOLDER,
    glob="*.txt",
    loader_cls=TextLoader
)
raw_docs.extend(txt_loader.load())

if not raw_docs:
    raise SystemExit("[FATAL] No supported files found in documents/pdfs/")

print(f"[INFO] Loaded {len(raw_docs)} source files.")

# Add metadata
for d in raw_docs:
    d.metadata.setdefault("source_category", "me_and_the_boys")
    d.metadata.setdefault("filename", os.path.basename(d.metadata.get("source", "")))

# ---------------------------------------------------------
# SPLIT DOCUMENTS
# ---------------------------------------------------------
print("[INFO] Splitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

docs = splitter.split_documents(raw_docs)
print(f"[INFO] Produced {len(docs)} chunks.")

# ---------------------------------------------------------
# INGEST INTO PINECONE
# ---------------------------------------------------------
print("[INFO] Starting vector ingestion...")

BATCH_SIZE = 100

def batch(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

for b in tqdm(batch(docs, BATCH_SIZE), total=(len(docs) // BATCH_SIZE) + 1):
    ids = [str(uuid.uuid4()) for _ in b]
    vector_store.add_documents(documents=b, ids=ids)

print("[INFO] Ingestion complete. Your lore is now immortal.")
