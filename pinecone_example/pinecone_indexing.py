from pinecone import Pinecone
import numpy as np
import pandas as pd
import dotenv
import os
import itertools
dotenv.load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "quickstart-py"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )



index = pc.Index(name=index_name)
index.upsert(
    vectors=[
        {
            "id": "BMW",
            "values": np.random.rand(1024).tolist(),
            "metadata": {"source": "Bayerische Motoren Werke AG"}
        }
    ]
)


index.upsert(
    vectors=[
        {
            "id": "VW",
            "values": np.random.rand(1024).tolist(),
            "metadata": {"source": "Volkswagen AG"}
        }
    ]
)

car_df = pd.DataFrame({'id': ['Mercedes', 'Tesla'], 'vectors': [np.random.rand(1024).tolist(), np.random.rand(1024).tolist()], 'metadata': [{'source': 'Mercedes-Benz Group AG'}, {'source': 'Audi AG'}]})
index.upsert(zip(car_df['id'], car_df['vectors'], car_df['metadata']))

res = index.query(
        top_k=3,
        include_values=True,
        include_metadata=True,
        vector=np.random.rand(1024).tolist())

for brand in res['matches']:
    print(f"ID: {brand['id']}, Score: {brand['score']}, Source: {brand['metadata']['source']}")


def chunks(iterable, size):
    it = iter(iterable)
    chunk = list(itertools.islice(it, size))
    while chunk:
        yield chunk
        chunk = list(itertools.islice(it, size))


vector_dimension = 1024
vector_count = 200
batch_size = 25

sample_date = map(
    lambda x: {
        "id": f"vec_{x}",
        "values": np.random.rand(vector_dimension).tolist(),
        "metadata": {"source": f"Sample vector {x}"}
    },
    range(vector_count))

for chunk in chunks(sample_date, batch_size):
    index.upsert(vectors=chunk)

