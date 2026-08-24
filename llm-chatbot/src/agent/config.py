"""Configuration settings for the LLM Chatbot."""

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


# Centralized configuration loader for environment variables.
# Provides a single place to access all required API keys and settings.
class Settings:
    # API keys
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")

    # Vectorstore settings
    DOC_PATH: str = "data/TT_Visa_FAQ.pdf"
    VECTORSTORE_DIR: str = "data/embeddings"
    # 1500 keeps every FAQ page (max 1013 chars) as a single chunk instead
    # of cutting mid-answer — see eval/compare_chunk_sizes.py for why.
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 0
    # 4 recovers a recall@3 regression measured at CHUNK_SIZE=1500 (see the
    # README's Retrieval Evaluation section) — recall@4 matches the old
    # chunk_size=500 config exactly, and k>4 adds no further recovery.
    RETRIEVAL_K: int = 4

    # Model settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    CHAT_MODEL: str = "google_genai:gemini-2.5-flash-lite"


settings = Settings()
