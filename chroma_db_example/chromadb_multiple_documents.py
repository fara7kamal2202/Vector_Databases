from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma.vectorstores import Chroma
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
import dotenv

dotenv.load_dotenv()

def process_llm_response(response):
    print("Answer:", response['result'])
    print("\nSource Documents:")
    for doc in response['source_documents']:
        print(f"- {doc.metadata['source']}")


loader = DirectoryLoader('articles/', glob='**/*.txt', loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

persist_directory = 'vector_db'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory)
vectordb = None
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)

retriever = vectordb.as_retriever(search_kwargs={"k": 2})
gpt_5_nano_llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0)


qa_chain = RetrievalQA.from_chain_type(
    llm=gpt_5_nano_llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
query = "what is the news about databricks?"
result = qa_chain.invoke(query)
process_llm_response(result)


