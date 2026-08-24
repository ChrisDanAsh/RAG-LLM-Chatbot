"""Tools for the agent to use."""

from langchain.tools import tool
from .config import settings
from .vectorstore import get_vectorstore


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information from vectorstore to help answer a query."""

    vectorstore = get_vectorstore()  # Load or create the vectorstore

    retrieved_docs = vectorstore.similarity_search(query, k=settings.RETRIEVAL_K)

    serialized = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return serialized, retrieved_docs
