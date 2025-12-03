# import basics
import os
import time
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Get the index name from environment variables
index_name = os.environ.get("PINECONE_INDEX_NAME")

print(f"Attempting to delete index '{index_name}'...")

# Check if the index exists before trying to delete it
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name in existing_indexes:
    pc.delete_index(index_name)
    # Wait for the index to be fully deleted
    while index_name in [index_info["name"] for index_info in pc.list_indexes()]:
        time.sleep(1) # Wait for 1 second
    print(f"Index '{index_name}' deleted successfully.")
else:
    print(f"Index '{index_name}' does not exist. Nothing to delete.")