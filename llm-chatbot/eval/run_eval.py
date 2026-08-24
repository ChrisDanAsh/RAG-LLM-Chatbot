"""Retrieval evaluation harness: recall@k and MRR over eval/questions.yaml.

Runs the hand-written question set (see questions.yaml for the labelling
scheme) against one or more chunk_size configs and reports recall@k / MRR
per bucket and overall, so you can compare configs before/after a change
without re-labelling anything (labels are by PDF page number, which is
stable across chunk sizes).

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --chunk-sizes 500 1500
    python eval/run_eval.py --chunk-sizes 500 750 1000 1500 --k 1 3 6

See the "Retrieval Evaluation" section of the project README for the
recorded 500-vs-1500 results and what they do/don't show.
"""

import argparse
import sys
from pathlib import Path

# eval/ is not an installed package - add it (and src/) to the import path
# so this script can be run directly with `python eval/run_eval.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from agent.config import settings
from compare_chunk_sizes import build_vectorstore  # reuse the same chunking logic

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def score_question(vectorstore, question, gold_pages, max_k):
    """Run one query and compute recall@k (for each k up to max_k) and MRR.

    Returns a dict with per-k recall (None for out-of-scope questions,
    where gold_pages is empty and recall isn't a meaningful metric),
    reciprocal rank, and the top-1 similarity score (useful even for
    out-of-scope questions as a "how confidently did it retrieve nothing
    relevant" diagnostic).
    """
    # similarity_search_with_score returns (Document, score) sorted
    # best-first; pull `max_k` results once and slice per-k below instead
    # of re-querying for every k value.
    results = vectorstore.similarity_search_with_score(question, k=max_k)
    retrieved_pages = [doc.metadata.get("page") for doc, _score in results]
    top_score = results[0][1] if results else None

    gold = set(gold_pages)

    recall_at_k = {}
    if gold:
        for k in range(1, max_k + 1):
            hit_pages = gold & set(retrieved_pages[:k])
            recall_at_k[k] = len(hit_pages) / len(gold)
    else:
        # No gold pages -> recall is undefined, not zero. Keep it out of
        # the averages rather than silently dragging them down.
        for k in range(1, max_k + 1):
            recall_at_k[k] = None

    # Reciprocal rank: 1/rank of the first retrieved chunk whose page is
    # gold, or 0.0 if none of the top max_k chunks hit. Same "undefined
    # for out-of-scope" treatment as recall.
    reciprocal_rank = None
    if gold:
        reciprocal_rank = 0.0
        for rank, page in enumerate(retrieved_pages, start=1):
            if page in gold:
                reciprocal_rank = 1.0 / rank
                break

    return {
        "recall_at_k": recall_at_k,
        "rr": reciprocal_rank,
        "top_score": top_score,
        "retrieved_pages": retrieved_pages,
    }


def evaluate_config(vectorstore, questions, ks):
    max_k = max(ks)
    per_question = []
    for q in questions:
        result = score_question(vectorstore, q["question"], q["gold_pages"], max_k)
        per_question.append({**q, **result})
    return per_question


def summarize(per_question, ks, bucket=None):
    """Average recall@k and MRR across questions that have gold labels.

    Pass `bucket` to restrict to one bucket's questions; None summarizes
    everything. Out-of-scope questions (empty gold_pages) are excluded
    from these averages since recall/MRR aren't defined for them.
    """
    rows = [q for q in per_question if bucket is None or q["bucket"] == bucket]
    labelled = [q for q in rows if q["gold_pages"]]

    summary = {"n_questions": len(rows), "n_labelled": len(labelled)}
    for k in ks:
        values = [q["recall_at_k"][k] for q in labelled]
        summary[f"recall@{k}"] = sum(values) / len(values) if values else None
    rr_values = [q["rr"] for q in labelled]
    summary["mrr"] = sum(rr_values) / len(rr_values) if rr_values else None

    # Mean top-1 score, tracked separately for out-of-scope questions
    # (labelled vs unlabelled) since that's the diagnostic they're for.
    unlabelled = [q for q in rows if not q["gold_pages"]]
    if unlabelled:
        summary["oos_mean_top_score"] = sum(q["top_score"] for q in unlabelled) / len(unlabelled)

    return summary


def print_summary_table(config_label, per_question, ks):
    print(f"\n--- {config_label} ---")
    buckets = sorted(set(q["bucket"] for q in per_question))
    header = ["bucket", "n"] + [f"recall@{k}" for k in ks] + ["mrr"]
    print("  " + " | ".join(f"{h:>10}" for h in header))

    for bucket in buckets + [None]:  # None = overall, printed last
        label = "OVERALL" if bucket is None else bucket
        s = summarize(per_question, ks, bucket=bucket)
        row = [label, str(s["n_labelled"])]
        for k in ks:
            v = s[f"recall@{k}"]
            row.append("n/a" if v is None else f"{v:.2f}")
        row.append("n/a" if s["mrr"] is None else f"{s['mrr']:.2f}")
        print("  " + " | ".join(f"{c:>10}" for c in row))

    oos_summary = summarize(per_question, ks, bucket="out_of_scope")
    if "oos_mean_top_score" in oos_summary:
        in_scope = [q for q in per_question if q["gold_pages"]]
        in_scope_mean = sum(q["top_score"] for q in in_scope) / len(in_scope)
        print(
            f"  out_of_scope mean top-1 score: {oos_summary['oos_mean_top_score']:.4f}"
            f"  (in-scope mean top-1 score: {in_scope_mean:.4f})"
        )


def print_misses(config_label, per_question, k):
    """List every labelled question that didn't hit its gold page(s) by rank k."""
    misses = [
        q for q in per_question
        if q["gold_pages"] and q["recall_at_k"][k] < 1.0
    ]
    if not misses:
        print(f"\n  [{config_label}] no misses at k={k}.")
        return
    print(f"\n  [{config_label}] misses at k={k}:")
    for q in misses:
        print(
            f"    {q['id']:>6} recall@{k}={q['recall_at_k'][k]:.2f}  "
            f"gold={q['gold_pages']}  retrieved={q['retrieved_pages'][:k]}  "
            f"\"{q['question']}\""
        )


def run(chunk_sizes, chunk_overlap, ks):
    questions = load_questions()
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH.name}")

    print(f"Loading {settings.DOC_PATH} ...")
    docs = PyPDFLoader(settings.DOC_PATH).load()

    print(f"Loading embedding model {settings.EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    results_by_config = {}
    for cs in chunk_sizes:
        vs, n_chunks = build_vectorstore(docs, embeddings, cs, chunk_overlap)
        label = f"chunk_size={cs} overlap={chunk_overlap} ({n_chunks} chunks)"
        per_question = evaluate_config(vs, questions, ks)
        results_by_config[label] = per_question
        print_summary_table(label, per_question, ks)

    # Show what's actually failing at the strictest k, per config, so a
    # regression is traceable to a specific question rather than just a
    # dropped aggregate number.
    print("\n" + "=" * 100)
    print(f"MISSES AT k={min(ks)}")
    print("=" * 100)
    for label, per_question in results_by_config.items():
        print_misses(label, per_question, min(ks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=[500, 1500],
        help="Chunk sizes to evaluate and compare (default: 500 1500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 3, 6],
        dest="ks",
        help="k values to report recall@k for (default: 1 3 6)",
    )
    args = parser.parse_args()
    run(args.chunk_sizes, args.chunk_overlap, sorted(args.ks))


if __name__ == "__main__":
    main()
