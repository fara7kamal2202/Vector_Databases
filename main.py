import os
import sqlite3
from openai import OpenAI
import numpy as np


import dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model="text-embedding-ada-002"):
    text.replace(("\n"), " ")
    r = client.embeddings.create(input=text, model=model)
    return r

def deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)
def main():
    conn = sqlite3.connect("embeddings.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Stocks (
    stock_code INTEGER PRIMARY KEY,
    stock_name TEXT NOT NULL
    )
    """)
    cursor.execute("DELETE FROM Stocks;")
    cursor.execute("DELETE FROM Vectors;")
    cursor.execute("INSERT INTO Stocks (stock_name) VALUES (?);", ("Tesla",))
    cursor.execute("INSERT INTO Stocks (stock_name) VALUES (?);", ("Microsoft",))
    cursor.execute("SELECT * FROM Stocks")
    rows = cursor.fetchall()
    print(rows)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Vectors (
    vector_id INTEGER PRIMARY KEY,
    vector_info BLOB NOT NULL
    )
    """)
    vect_tesla_embedded = np.array([1.3, 3.5, 2.2, 0.9])
    vect_microsoft_embedded = np.array([0.5, 1.5, 4.2, 2.9])

    cursor.execute("INSERT INTO Vectors (vector_info) VALUES (?);", (sqlite3.Binary(vect_tesla_embedded.tobytes()),))
    cursor.execute("INSERT INTO Vectors (vector_info) VALUES (?);", (sqlite3.Binary(vect_microsoft_embedded.tobytes()),))
    cursor.execute("SELECT * FROM Vectors")
    rows = cursor.fetchall()
    vectors = []
    for row in rows:
        vector = np.frombuffer(row[1], dtype=np.float64)
        print(vector)
        vectors.append(vector)
    print(vectors)

    q_vector = np.array([2.5, 1.2, 3.5, 5.5])
    cursor.execute("SELECT vector_info FROM Vectors ORDER BY abs(vector_info - ?) ASC;", (sqlite3.Binary(q_vector.tobytes()),))
    res = cursor.fetchone()
    print(deserialize_vector(res[0]))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    main()
