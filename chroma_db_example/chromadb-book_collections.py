import chromadb
import numpy as np


client = chromadb.Client()

book_collection = client.create_collection(name="favorite_books")
book_collection.add(
    documents=[
        "The Great Gatsby by F. Scott Fitzgerald",
        "To Kill a Mockingbird by Harper Lee",
        "1984 by George Orwell",
        "Crime and Punishment by Fyodor Dostoevsky",
        "Pride and Prejudice by Jane Austen"
    ],
    metadatas=[
        {"author": "F. Scott Fitzgerald", "year": 1925},
        {"author": "Harper Lee", "year": 1960},
        {"author": "George Orwell", "year": 1949},
        {"author": "Fyodor Dostoevsky", "year": 1866},
        {"author": "Jane Austen", "year": 1813}
    ],
    ids=["book1", "book2", "book3", "book4", "book5"])


print(book_collection.query(
        query_texts=["A dystopian novel about totalitarianism"],
        n_results=1))
the_great_GATSBY = book_collection.get(ids=["book1"])
print(the_great_GATSBY)


results = book_collection.query(
    query_texts=["A novel about racial injustice in the Deep South"],
    where={"year": 1960},
    n_results=1
)
print(results)