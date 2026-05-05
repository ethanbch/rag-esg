from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from rouge_score import rouge_scorer

from config import ALBERT_API_KEY, EMBEDDING_MODEL, RERANK_BACKEND, RERANK_MODEL
from rag_app.service import DEFAULT_CHAT_MODEL, RagRuntimeConfig, RagService

DEFAULT_DATASET = Path("evaluation/rag_evaluation_dataset.csv")
DEFAULT_OUTPUT = Path("evaluation/last_eval_report.json")
DEFAULT_TOP_K = 3
DEFAULT_MAX_CHUNKS = 8


def _load_dataset_csv(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    samples: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Dataset CSV must include a header row.")

        for index, row in enumerate(reader, start=1):
            question = str(row.get("question", "")).strip()
            if not question:
                raise ValueError(f"Dataset row #{index} is missing a question.")
            samples.append(row)

    return samples


def _resolve_reference_answer(sample: dict[str, Any]) -> str:
    for key in ("ground_truth", "reference_answer", "answer"):
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


def run_evaluation(dataset_path: Path, output_path: Path) -> None:
    api_key = (ALBERT_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ALBERT_API_KEY is required for evaluation.")

    samples = _load_dataset_csv(dataset_path)

    rag_service = RagService(
        RagRuntimeConfig(
            api_key=api_key,
            model=DEFAULT_CHAT_MODEL,
            embedding_model=EMBEDDING_MODEL,
            reranker_backend=RERANK_BACKEND,
            reranker_model=RERANK_MODEL,
        )
    )

    available_collections = rag_service.list_collections()
    if not available_collections:
        raise RuntimeError("No collections available for evaluation.")

    default_collections = available_collections

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows: list[dict[str, Any]] = []

    print(
        f"Evaluating {len(samples)} sample(s) with model={DEFAULT_CHAT_MODEL}, reranker={RERANK_BACKEND}, reranker_model={RERANK_MODEL}"
    )

    for index, sample in enumerate(samples, start=1):
        question = str(sample["question"]).strip()
        reference_answer = _resolve_reference_answer(sample)
        rag_answer = rag_service.ask(
            question=question,
            collection_names=default_collections,
            n_results_per_collection=DEFAULT_TOP_K,
            max_chunks=DEFAULT_MAX_CHUNKS,
            system_prompt=None,
            reranker_backend=RERANK_BACKEND,
        )

        rouge_scores = scorer.score(reference_answer, rag_answer.answer)
        row = {
            "sample_index": index,
            "company": str(sample.get("company", "")).strip(),
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": rag_answer.answer,
            "topic": str(sample.get("topic", "")).strip(),
            "source_pdf": str(sample.get("source_pdf", "")).strip(),
            "collections": default_collections,
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
        "model": DEFAULT_CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_backend": RERANK_BACKEND,
        "reranker_model": RERANK_MODEL,
        "n_results_per_collection": DEFAULT_TOP_K,
        "max_chunks": DEFAULT_MAX_CHUNKS,
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
        description="Evaluate RAG answers with ROUGE metrics from CSV."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to CSV dataset with questions and reference answers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
