from qdrant_client import QdrantClient
import os
import dotenv

dotenv.load_dotenv()

COLLECTION_NAME = "favorite_books"

client = QdrantClient(
    url=os.getenv("QDRANT_HOST") or "http://localhost",
    port=6333,
    api_key=os.getenv("QDRANT_API_KEY")
)

try:
    collections = client.get_collections().collections
    if any(c.name == COLLECTION_NAME for c in collections):
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"Deleted collection '{COLLECTION_NAME}'.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist.")
except Exception as e:
    print(f"Error deleting collection '{COLLECTION_NAME}': {e}")

