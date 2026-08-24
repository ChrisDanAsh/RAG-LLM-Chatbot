"""Builds a vectorstore from a PDF document."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import settings


def build_vectorstore(file_path: str, persist_dir: str = "data/embeddings"):
    """
         Loads a PDF file, splits it into chunks, embeds them, and stores them in a vectorstore.

    Args:
        file_path (str): Path to the PDF file to embed.
        persist_dir (str): Directory to optionally save embeddings.

    Returns:
        InMemoryVectorStore: The constructed vectorstore containing embedded chunks.
    """

    # 1. Load PDF from file path
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    if not docs:
        raise ValueError("Failed to pdf — no documents returned.")

    # print(f"Loaded {len(docs)} documents.")

    # 2. Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # 4. Create vectorstore
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    # document_ids = vector_store.add_documents(documents=all_splits)
    # print(f"Created vectorstore with {len(document_ids)} documents.")

    return vector_store


# Global vectorstore instance
_vectorstore = None


def initialize_vectorstore():
    """
    Builds the vectorstore at startup.
    Should be called once during application initialization.
    """
    global _vectorstore

    _vectorstore = build_vectorstore(
        file_path=settings.DOC_PATH,
        persist_dir=settings.VECTORSTORE_DIR,
    )


def get_vectorstore():
    """
    Returns the pre-initialized vectorstore instance.
    Raises error if not initialized.
    """
    if _vectorstore is None:
        raise RuntimeError(
            "Vectorstore not initialized. Call initialize_vectorstore() first."
        )
    return _vectorstore
