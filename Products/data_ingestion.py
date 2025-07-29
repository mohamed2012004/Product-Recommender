import os
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from data_converter import DataConverter
from config import Config

class DataIngestor:
    def __init__(self):
        self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

        # Set paths for FAISS index and document store
        self.folder_path = f"faiss_store/run"
        self.index_file = os.path.join(self.folder_path, "faiss_index")
        self.store_file = os.path.join(self.folder_path, "docstore.pkl")

        os.makedirs(self.folder_path, exist_ok=True)

        self.vstore = None

    def ingest(self, load_existing=True):
        if load_existing and os.path.exists(self.index_file) and os.path.exists(self.store_file):
            print(f"Loading existing FAISS index from {self.folder_path}")
            self.vstore = FAISS.load_local(
                folder_path=self.folder_path,
                embeddings=self.embedding,
                index_name="faiss_index"
            )
            return self.vstore

        print("Converting and embedding documents...")
        docs = DataConverter("data/flipkart_product_review.csv").convert()

        self.vstore = FAISS.from_documents(docs, self.embedding)

        self.vstore.save_local(self.folder_path)
        print(f"FAISS index saved to {self.folder_path}")

        return self.vstore

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.ingest(load_existing=False)
    print("✅ Data ingestion completed successfully.")
