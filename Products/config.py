import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    RAG_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"