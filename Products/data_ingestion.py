import os
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from Products.data_converter import DataConverter
from Products.config import Config

class DataIngestor:
    def __init__(self):
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        # Set paths for FAISS index and document store
        self.folder_path = f"faiss_store/run"
        self.index_file = os.path.join(self.folder_path, "faiss_index")
        self.store_file = os.path.join(self.folder_path, "docstore.pkl")

        os.makedirs(self.folder_path, exist_ok=True)

        self.vstore = None

    def ingest(self):
        
        print("Converting and embedding documents...")
        docs = DataConverter("data/flipkart_product_review.csv").convert()

        self.vstore = FAISS.from_documents(docs, self.embedding)

        print(f"FAISS index saved to {self.folder_path}")

        return self.vstore
