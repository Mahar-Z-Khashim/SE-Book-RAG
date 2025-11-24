#Libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

#Load PDF
loader = PyPDFLoader("https://mlsu.ac.in/econtents/16_EBOOK-7th_ed_software_engineering_a_practitioners_approach_by_roger_s._pressman_.pdf")
docs = loader.load()

#Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
book_chunks = text_splitter.split_documents(docs)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Persist directory for the vector database
persist_directory = "chroma_SE_RAG"
os.makedirs(persist_directory, exist_ok=True)

#Create and persist vector database
vectordb = Chroma.from_documents(
    documents=book_chunks,
    embedding=embeddings,
    persist_directory=persist_directory
    )
print("Vector database created with", vectordb._collection.count(), "documents.")