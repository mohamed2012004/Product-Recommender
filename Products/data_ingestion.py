import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from Products.data_converter import DataConverter
from Products.config import Config
from langchain.embeddings.base import Embeddings


class DummyEmbedding(Embeddings):
    def embed_documents(self, texts):
        return [np.zeros(768) for _ in texts]

    def embed_query(self, text):
        return np.zeros(768)


class DataIngestor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(Config.EMBEDDING_MODEL)
        self.model.eval()

        self.folder_path = "faiss_store/run"
        os.makedirs(self.folder_path, exist_ok=True)

        self.vstore = None

    def mean_pooling(self, token_embeddings):
        """Mean pooling and L2 normalize the embedding."""
        pooled = token_embeddings.mean(dim=0).numpy()
        return pooled / np.linalg.norm(pooled)

    def chunk_embeddings(self, text, metadata=None, chunk_size=500):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False, return_attention_mask=True)
        input_ids = inputs["input_ids"].squeeze(0)

        if len(input_ids) > self.tokenizer.model_max_length:
            print(f"[Warning] Text too long ({len(input_ids)} tokens), skipping or truncating...")
            input_ids = input_ids[:self.tokenizer.model_max_length]
            inputs["input_ids"] = input_ids.unsqueeze(0)
            inputs["attention_mask"] = torch.ones_like(input_ids).unsqueeze(0)
        else:
            print(f"[Info] Token length OK: {len(input_ids)} tokens")

        with torch.no_grad():
            outputs = self.model(**inputs)

        token_embeddings = outputs.last_hidden_state.squeeze(0)
        token_ids = inputs["input_ids"].squeeze(0)

        chunks = []
        for i in range(0, token_embeddings.size(0), chunk_size):
            chunk_embed = token_embeddings[i:i + chunk_size]
            chunk_ids = token_ids[i:i + chunk_size]

            if chunk_embed.size(0) == 0:
                continue

            vector = self.mean_pooling(chunk_embed)
            text_chunk = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)

            doc = Document(page_content=text_chunk, metadata=metadata)
            chunks.append((vector, doc))

        return chunks

    def ingest(self):
        if os.path.exists(os.path.join(self.folder_path, "index.faiss")):
            print("Loading existing FAISS index...")
            self.vstore = FAISS.load_local(
                folder_path=self.folder_path,
                embeddings=DummyEmbedding(),
                index_name="index",
                  allow_dangerous_deserialization=True
            )
            return self.vstore

        print("FAISS index not found. Creating new one...")

        print("Loading and converting documents...")
        docs = DataConverter("data/flipkart_product_review.csv").convert()

        all_vectors = []
        all_docs = []

        print("Embedding and chunking...")
        for doc in docs:
            text = doc.page_content
            metadata = doc.metadata
            chunks = self.chunk_embeddings(text, metadata)

            for vec, chunk_doc in chunks:
                all_vectors.append(vec)
                all_docs.append(chunk_doc)

        print("Saving into FAISS vector store...")

        text_embeddings = [(doc.page_content, vec) for doc, vec in zip(all_docs, all_vectors)]

        self.vstore = FAISS.from_embeddings(
            text_embeddings=text_embeddings,
            embedding=DummyEmbedding()
        )

        self.vstore.save_local(self.folder_path)
        print("FAISS index saved to disk ")

        return self.vstore
