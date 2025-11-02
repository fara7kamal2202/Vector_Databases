import faiss
import numpy as np

dimension = 64
num_vector = 1000
query_vector = np.random.random((1, dimension)).astype('float32')
data_vectors = np.random.random((num_vector, dimension)).astype('float32')

index = faiss.IndexFlatL2(dimension)
index.add(data_vectors)

k = 5 # number of nearest neighbors to retrieve
distances, indices = index.search(query_vector, k)
print(f"Query Vector: {query_vector}")
print(f"Indices of Nearest Neighbors: {indices}")
print(f"Distances to Nearest Neighbors: {distances}")

query_vector_2 = np.array([[10] * dimension], dtype='float32')
data_vectors_2 = np.random.normal(loc=10.0, scale=1.0, size=(num_vector, dimension)).astype('float32')
index = faiss.IndexFlatL2(dimension)
index.add(data_vectors_2)
distances_2, indices_2 = index.search(query_vector_2, k)
print(f"Query Vector 2: {query_vector_2}")
print(f"Indices of Nearest Neighbors for Query Vector 2: {indices_2}")
print(f"Distances to Nearest Neighbors for Query Vector 2: {distances_2}")
