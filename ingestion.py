# import basics
import os
import time
from dotenv import load_dotenv
import uuid  # Look, something new for you to learn.
from tqdm import tqdm # To see your progress, so you don't get impatient.

# import pinecone
from pinecone import Pinecone, ServerlessSpec

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

#documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Pre-flight Check ---
print("Running pre-flight checks...")
required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.environ.get(var)]

if missing_vars:
    print(f"FATAL: Missing required environment variables: {', '.join(missing_vars)}")
    print("Fix your .env file before you waste my time again.")
    exit()

print("All checks passed. Let's do this.")
print("----------------------------------------------------")

# --- Database and Embeddings Setup ---
# (The rest of your script follows here)
# --- Database and Embeddings Setup ---
# (Your existing setup code is fine, if a little verbose)
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")

if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' not found. Creating it... might take a minute.")
    pc.create_index(
        name=index_name,
        dimension=3072,  # Match your OpenAI model's dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# (Keep all the imports and the Pinecone setup from the previous script)

# --- Document Loading and Splitting with METADATA ---
print("Loading documents from subfolders and adding metadata...")

# Define your source directories
base_dir = "documents"
source_categories = ["pdfs"]
all_raw_documents = []

for category in source_categories:
    folder_path = os.path.join(base_dir, category)
    
    if not os.path.isdir(folder_path):
        print(f"WARNING: Directory not found, skipping: '{folder_path}'")
        continue

    print(f"--> Loading files from '{folder_path}'")
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load()

    # This is the important part. Pay attention.
    # We're adding the source category to each document's metadata.
    for doc in docs:
        doc.metadata['source_category'] = category

    all_raw_documents.extend(docs)
    print(f"    Loaded {len(docs)} documents from '{category}'.")

if not all_raw_documents:
    print("FATAL: No documents found in any of the source directories. Exiting.")
    exit()

print("\n----------------------------------------------------")
print(f"Total documents loaded: {len(all_raw_documents)}")

print("Splitting all documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
documents = text_splitter.split_documents(all_raw_documents)
print(f"Total text chunks created: {len(documents)}")
print("----------------------------------------------------")


# --- Batching and Uploading ---
# (The rest of the script for batching and uploading stays EXACTLY the same as before)
# It will now process the 'documents' list which contains chunks from all folders,
# each with its own metadata.

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

BATCH_SIZE = 100 

print(f"Starting ingestion in batches of {BATCH_SIZE}...")

# ... (the rest of the tqdm loop for batching) ...
# for batch in tqdm(batch_generator(documents, BATCH_SIZE), ...):
#     ...

for batch in tqdm(batch_generator(documents, BATCH_SIZE), total=(len(documents) // BATCH_SIZE) + 1):
    # Create REAL unique IDs for this batch
    ids = [str(uuid.uuid4()) for _ in batch]
    
    # Add documents to the vector store
    vector_store.add_documents(documents=batch, ids=ids)

print("\nIngestion complete. You're welcome.")