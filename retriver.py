from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
import os
os.environ["COHERE_API_KEY"] = os.environ.get("COHERE_API_KEY")
class VectorRetriever:
    def __init__(self, persist_dir="vectorstore"):
        embeddings = CohereEmbeddings(user_agent="split",model="embed-english-light-v3.0")
        self.db = FAISS.load_local(persist_dir, embeddings,allow_dangerous_deserialization=True)
        
    def retrieve(self, query, k=3):
        return self.db.similarity_search(query, k=k)