import time
import pandas as pd
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import dotenv
import os
dotenv.load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

df1 = pd.read_csv('news.txt')
df1['title'] = df1['title'].fillna('')

model = SentenceTransformer('average_word_embeddings_komninos')
print("Generating embeddings for titles")
encode_titles = model.encode(df1['title'].tolist(), show_progress_bar=True, device='cpu')

pc.create_index(name="semantic-search-index",
                dimension=300,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})
index = pc.Index(name="semantic-search-index")

for i in range(0, 20):
    ids = str(df1.index[i])
    vecs = df1.title.iloc[i]
    metadata = {
        'symbol': df1.symbol.iloc[i],
        'publishedDate': df1.publishedDate.iloc[i],
        'title': df1.title.iloc[i],
        'site': df1.site.iloc[i],
    }

upsert_response = index.upsert(vectors=[(ids, vecs, metadata)])
print(f"Upserted vector ID: {ids}")
encoded_phrase = model.encode('food cost').tolist()

top_k = 4
result = index.query(
    top_k=top_k,
    vector=encoded_phrase,
    include_metadata=True)

print(result)