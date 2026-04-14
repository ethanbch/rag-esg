from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from rouge_score import rouge_scorer

from config import ALBERT_API_KEY, EMBEDDING_MODEL, RERANK_MODEL
from rag_app.service import DEFAULT_CHAT_MODEL, RagRuntimeConfig, RagService


def _load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON list of QA samples.")

    samples: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{index} must be an object.")

        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"Dataset item #{index} is missing a question.")

        samples.append(item)

    return samples


def _resolve_reference_answer(sample: dict[str, Any]) -> str:
    for key in ("reference_answer", "ground_truth", "answer"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        "Each sample must provide one of: reference_answer, ground_truth, answer."
    )


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": getattr(chunk, "chunk_id", ""),
        "collection_name": getattr(chunk, "collection_name", ""),
        "distance": getattr(chunk, "distance", None),
        "score": getattr(chunk, "score", None),
    }


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    model: str,
    embedding_model: str,
    reranker_backend: str,
    reranker_model: str,
    n_results_per_collection: int,
    max_chunks: int,
    system_prompt: str | None,
    collections: list[str] | None,
) -> None:
    api_key = (ALBERT_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ALBERT_API_KEY is required for evaluation.")

    samples = _load_dataset(dataset_path)

    rag_service = RagService(
        RagRuntimeConfig(
            api_key=api_key,
            model=model,
            embedding_model=embedding_model,
            reranker_backend=reranker_backend,
            reranker_model=reranker_model,
        )
    )

    available_collections = rag_service.list_collections()
    if not available_collections:
        raise RuntimeError("No collections available for evaluation.")

    available_collections_set = set(available_collections)

    if collections:
        missing_forced_collections = [
            collection
            for collection in collections
            if collection not in available_collections_set
        ]
        if missing_forced_collections:
            raise RuntimeError(
                "Unknown collection(s) passed with --collections: "
                f"{', '.join(missing_forced_collections)}. "
                "Available collections are: "
                f"{', '.join(available_collections)}"
            )

    default_collections = collections or available_collections

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows: list[dict[str, Any]] = []

    print(
        f"Evaluating {len(samples)} sample(s) with model={model}, reranker={reranker_backend}, reranker_model={reranker_model}"
    )

    for index, sample in enumerate(samples, start=1):
        question = str(sample["question"]).strip()
        reference_answer = _resolve_reference_answer(sample)

        sample_collections_raw = sample.get("collections")
        if isinstance(sample_collections_raw, list):
            sample_collections = [
                str(collection).strip()
                for collection in sample_collections_raw
                if str(collection).strip()
            ]
        else:
            sample_collections = []

        missing_sample_collections = [
            collection
            for collection in sample_collections
            if collection not in available_collections_set
        ]
        valid_sample_collections = [
            collection
            for collection in sample_collections
            if collection in available_collections_set
        ]

        if missing_sample_collections:
            print(
                f"[{index}/{len(samples)}] Missing collection(s) in dataset sample: "
                f"{', '.join(missing_sample_collections)}. They will be ignored."
            )

        if valid_sample_collections:
            active_collections = valid_sample_collections
        elif sample_collections:
            active_collections = default_collections
            print(
                f"[{index}/{len(samples)}] No valid collection left for this sample; "
                f"falling back to default collections: {', '.join(default_collections)}"
            )
        else:
            active_collections = default_collections

        rag_answer = rag_service.ask(
            question=question,
            collection_names=active_collections,
            n_results_per_collection=n_results_per_collection,
            max_chunks=max_chunks,
            system_prompt=system_prompt,
            reranker_backend=reranker_backend,
        )

        rouge_scores = scorer.score(reference_answer, rag_answer.answer)
        row = {
            "sample_index": index,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": rag_answer.answer,
            "collections": active_collections,
            "rouge1_f1": rouge_scores["rouge1"].fmeasure,
            "rouge2_f1": rouge_scores["rouge2"].fmeasure,
            "rougeL_f1": rouge_scores["rougeL"].fmeasure,
            "retrieved_chunk_count": len(rag_answer.chunks),
            "retrieved_chunks": [
                _serialize_chunk(chunk) for chunk in rag_answer.chunks
            ],
        }
        rows.append(row)

        print(
            f"[{index}/{len(samples)}] ROUGE-L F1={row['rougeL_f1']:.4f} | chunks={row['retrieved_chunk_count']}"
        )

    summary = {
        "samples": len(rows),
        "avg_rouge1_f1": mean(row["rouge1_f1"] for row in rows) if rows else 0.0,
        "avg_rouge2_f1": mean(row["rouge2_f1"] for row in rows) if rows else 0.0,
        "avg_rougeL_f1": mean(row["rougeL_f1"] for row in rows) if rows else 0.0,
        "model": model,
        "embedding_model": embedding_model,
        "reranker_backend": reranker_backend,
        "reranker_model": reranker_model,
        "n_results_per_collection": n_results_per_collection,
        "max_chunks": max_chunks,
    }

    output = {
        "summary": summary,
        "results": rows,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\nEvaluation completed.")
    print(json.dumps(summary, indent=2))
    print(f"Detailed report written to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers with ROUGE metrics."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/qa_dataset.example.json"),
        help="Path to JSON dataset with questions and reference answers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/last_eval_report.json"),
        help="Path to write the evaluation report.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CHAT_MODEL,
        help="Chat model used for generation.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help="Embedding model used for retrieval and reranking.",
    )
    parser.add_argument(
        "--reranker-backend",
        type=str,
        default="api",
        choices=["api", "none", "cosine"],
        help="Reranker backend used post-retrieval.",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=RERANK_MODEL,
        help="Reranker model used by API backend.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Chunks retrieved per collection before reranking.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=8,
        help="Max chunks kept in final context.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt override for generation.",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        default=None,
        help="Optional list of collection names to force for all samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        model=args.model,
        embedding_model=args.embedding_model,
        reranker_backend=args.reranker_backend,
        reranker_model=args.reranker_model,
        n_results_per_collection=args.n_results,
        max_chunks=args.max_chunks,
        system_prompt=args.system_prompt,
        collections=args.collections,
    )


if __name__ == "__main__":
    main()
