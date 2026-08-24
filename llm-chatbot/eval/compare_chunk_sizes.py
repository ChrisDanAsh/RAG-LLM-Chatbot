"""Compare retrieval quality across chunk sizes for a handful of questions.

Builds a separate in-memory vectorstore per chunk_size, runs the same
questions against each, and prints the retrieved chunks side by side so you
can see *why* a chunk size helps or hurts, not just the aggregate score.

Usage:
    python eval/compare_chunk_sizes.py
    python eval/compare_chunk_sizes.py --chunk-sizes 500 1000 1500
    python eval/compare_chunk_sizes.py --question "how much does it cost?"

Finding from running this against data/TT_Visa_FAQ.pdf:
- The default question below ("how many documents can I upload and can I
  pay by credit card") targets page 4 (0-indexed), the PDF's longest page
  at 1013 characters — "What do I need to apply for a Visa?", a bulleted
  list covering photo, documents, and payment.
- At chunk_size=500 that page gets cut into ~3 fragments; at
  chunk_size=1500 it survives as one chunk.
- Result: at 500, top-3 results are dominated by two disconnected
  fragments of the SAME page, both missing the question heading that
  anchors them; at 1500, one chunk returns the complete, self-contained
  answer.
- The other two default questions are milder cases that happen to land on
  pages short enough to survive a 500-char cut mostly intact — kept here
  so you can see that not every question is affected equally.
"""

import argparse
import sys
from pathlib import Path

# This script lives in eval/, but the `agent` package lives in src/agent.
# Add src/ to the import path so `from agent.config import settings` works
# without needing the package pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config import settings

# A couple of questions chosen to stress vocabulary/structure, not just
# repeat the FAQ headings verbatim. Override with --question on the CLI.
# See the module docstring above for what each one demonstrates.
DEFAULT_QUESTIONS = [
    "how many documents can I upload and can I pay by credit card",
    "what image formats are accepted?",
    "how much does it cost?",
]

TOP_K = 3  # how many chunks to retrieve and display per question
SNIPPET_LEN = 220  # characters of chunk text to print, so output stays readable


def build_vectorstore(docs, embeddings, chunk_size, chunk_overlap):
    """Split `docs` into chunks of the given size/overlap and embed them.

    This mirrors agent/vectorstore.py's build_vectorstore, but takes
    chunk_size/overlap as parameters (and a pre-loaded embeddings model)
    so we can build several vectorstores in one run without re-downloading
    the embedding model or re-parsing the PDF each time.
    """
    # RecursiveCharacterTextSplitter tries to cut on paragraph/sentence/word
    # boundaries where possible, only falling back to a hard character cut
    # when a boundary can't be found within chunk_size.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # records each chunk's offset in the source page
    )
    chunks = splitter.split_documents(docs)

    # InMemoryVectorStore embeds every chunk immediately via `embeddings`
    # and keeps the vectors in memory for cosine-similarity search below.
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=chunks)
    return vector_store, len(chunks)


def run(chunk_sizes, chunk_overlap, questions):
    print(f"Loading {settings.DOC_PATH} ...")
    # PyPDFLoader returns one LangChain Document per PDF page, each with
    # metadata["page"] set — that page number is what we print below so we
    # can tell which chunk came from which FAQ topic.
    docs = PyPDFLoader(settings.DOC_PATH).load()

    print(f"Loading embedding model {settings.EMBEDDING_MODEL} ...")
    # Load the embedding model once and reuse it across every chunk-size
    # variant — the model itself doesn't change, only how the text is cut.
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # Build one vectorstore per requested chunk size up front, so the
    # question loop below just does lookups (fast) instead of rebuilding
    # embeddings for every question.
    stores = {}
    for cs in chunk_sizes:
        vs, n_chunks = build_vectorstore(docs, embeddings, cs, chunk_overlap)
        stores[cs] = vs
        print(f"chunk_size={cs:5d} overlap={chunk_overlap:3d} -> {n_chunks} chunks")

    for question in questions:
        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)

        for cs in chunk_sizes:
            print(f"\n--- chunk_size={cs} (top {TOP_K}) ---")
            # similarity_search_with_score embeds the question with the same
            # model, ranks chunks by cosine similarity, and returns
            # (Document, score) pairs sorted best-first — higher score
            # (closer to 1.0) means more similar.
            results = stores[cs].similarity_search_with_score(question, k=TOP_K)
            for rank, (doc, score) in enumerate(results, start=1):
                page = doc.metadata.get("page")
                # Collapse whitespace/newlines so each chunk prints as one
                # readable line instead of the PDF's ragged line breaks.
                snippet = " ".join(doc.page_content.split())[:SNIPPET_LEN]
                print(f"  [{rank}] score={score:.4f} page={page}")
                print(f"      {snippet}...")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[500, 1500],
        help="Chunk sizes to compare (default: 500 1500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)",
    )
    parser.add_argument(
        "--question",
        action="append",
        dest="questions",
        help="Question to test (repeatable). Defaults to a small built-in set.",
    )
    args = parser.parse_args()

    # --question can be passed multiple times; if it's never passed at all,
    # argparse leaves `questions` as None, so fall back to the defaults.
    questions = args.questions if args.questions else DEFAULT_QUESTIONS
    run(args.chunk_sizes, args.chunk_overlap, questions)


if __name__ == "__main__":
    main()
