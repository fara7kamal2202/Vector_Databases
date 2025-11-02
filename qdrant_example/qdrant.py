from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import dotenv
import os

dotenv.load_dotenv()

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10,
        separator="\n",
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


client = QdrantClient(url=os.getenv('QDRANT_HOST'),
                      port=6333, api_key=os.getenv("QDRANT_API_KEY"))


# client.create_collection(collection_name="favorite_books",
#                          vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))

embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vector_store = Qdrant(
    client=client,
    collection_name="favorite_books",
    embeddings=embedding)

with open('books.txt', encoding='utf-8') as f:
    raw_text = f.read()


chunks = get_chunks(raw_text)
vector_store.add_texts(texts=chunks)



