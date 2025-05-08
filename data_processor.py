import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.document_loaders import TextLoader
os.environ["COHERE_API_KEY"] = os.environ.get("COHERE_API_KEY")
class DataProcessor:
    def __init__(self, data_dir="documents", persist_dir="vectorstore"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
    def prepare_documents(self):
        documents = []
        for file in os.listdir(self.data_dir):
            loader = TextLoader(f"{self.data_dir}/{file}")
            documents.extend(loader.load())
        
        chunks = self.text_splitter.split_documents(documents)
        embeddings = CohereEmbeddings(user_agent="split",model="embed-english-light-v3.0")
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(self.persist_dir)
        return len(chunks)
    

dp= DataProcessor()
dp.prepare_documents()