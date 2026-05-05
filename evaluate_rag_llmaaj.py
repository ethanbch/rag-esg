from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from config import ALBERT_API_KEY, EMBEDDING_MODEL, RERANK_BACKEND, RERANK_MODEL
from rag_app.albert_client import AlbertApiConfig, AlbertChatClient
from rag_app.service import DEFAULT_CHAT_MODEL, RagRuntimeConfig, RagService

DEFAULT_DATASET = Path("evaluation/rag_evaluation_dataset.csv")
DEFAULT_OUTPUT = Path("evaluation/last_eval_report_llm.json")
DEFAULT_TOP_K = 5
DEFAULT_MAX_CHUNKS = 12
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_CHAT_MODEL)


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
        "Each sample must provide one of: ground_truth, reference_answer, answer."
    )


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    return {
        "chunk_id": getattr(chunk, "chunk_id", ""),
        "collection_name": getattr(chunk, "collection_name", ""),
        "distance": getattr(chunk, "distance", None),
        "score": getattr(chunk, "score", None),
    }


def _judge_answer(
    judge: AlbertChatClient,
    *,
    question: str,
    reference_answer: str,
    generated_answer: str,
) -> dict[str, Any]:
    system = (
        "You are a strict evaluator for RAG answers. "
        "Score only against the reference answer. "
        "Return JSON only with keys: score (1-5), verdict (correct|partial|incorrect), rationale."
    )
    user = (
        "Question:\n"
        f"{question}\n\n"
        "Reference answer:\n"
        f"{reference_answer}\n\n"
        "Generated answer:\n"
        f"{generated_answer}\n\n"
        "Reply with JSON only."
    )

    raw = judge.complete(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
    )

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"score": None, "verdict": "unparsed", "rationale": raw.strip()}

    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"score": None, "verdict": "unparsed", "rationale": raw.strip()}

    score = payload.get("score")
    verdict = str(payload.get("verdict", "")).strip().lower()
    rationale = str(payload.get("rationale", "")).strip()

    return {
        "score": score,
        "verdict": verdict,
        "rationale": rationale,
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

    judge = AlbertChatClient(
        api_config=AlbertApiConfig(
            api_key=api_key, base_url=rag_service._config.base_url
        ),
        model=DEFAULT_JUDGE_MODEL,
    )

    print(
        f"Evaluating {len(samples)} sample(s) with model={DEFAULT_CHAT_MODEL}, judge={DEFAULT_JUDGE_MODEL}"
    )

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        question = str(sample["question"]).strip()
        reference_answer = _resolve_reference_answer(sample)

        rag_answer = rag_service.ask(
            question=question,
            collection_names=available_collections,
            n_results_per_collection=DEFAULT_TOP_K,
            max_chunks=DEFAULT_MAX_CHUNKS,
            system_prompt=None,
            reranker_backend=RERANK_BACKEND,
        )

        judgment = _judge_answer(
            judge,
            question=question,
            reference_answer=reference_answer,
            generated_answer=rag_answer.answer,
        )

        row = {
            "sample_index": index,
            "company": str(sample.get("company", "")).strip(),
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": rag_answer.answer,
            "topic": str(sample.get("topic", "")).strip(),
            "source_pdf": str(sample.get("source_pdf", "")).strip(),
            "collections": available_collections,
            "judge_score": judgment.get("score"),
            "judge_verdict": judgment.get("verdict"),
            "judge_rationale": judgment.get("rationale"),
            "retrieved_chunk_count": len(rag_answer.chunks),
            "retrieved_chunks": [
                _serialize_chunk(chunk) for chunk in rag_answer.chunks
            ],
        }
        rows.append(row)

        print(
            f"[{index}/{len(samples)}] Judge score={row['judge_score']} | chunks={row['retrieved_chunk_count']}"
        )

    scored_rows = [
        row for row in rows if isinstance(row.get("judge_score"), (int, float))
    ]
    avg_score = mean(row["judge_score"] for row in scored_rows) if scored_rows else 0.0

    summary = {
        "samples": len(rows),
        "avg_judge_score": avg_score,
        "model": DEFAULT_CHAT_MODEL,
        "judge_model": DEFAULT_JUDGE_MODEL,
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
        description="Evaluate RAG answers with an LLM-as-a-judge from CSV."
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
