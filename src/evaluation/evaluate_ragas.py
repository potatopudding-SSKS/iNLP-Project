"""Run RAGAS evaluation on BioASQ-style outputs.

This is a separate evaluator focused only on RAGAS metrics so you can test any
prediction file without mixing it into the classical BioASQ metrics script.

Example:
  python src/evaluation/evaluate_ragas.py \
    --gold datasets/train/train.json \
    --pred outputs/rag_baseline_answers.json \
    --out outputs/rag_baseline_ragas.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

try:
    from evaluate import build_ragas_rows, load_json
except ModuleNotFoundError:
    from src.evaluation.evaluate import build_ragas_rows, load_json


DEFAULT_MODEL = "gemini-2.0-flash"


def align_rows(gold_path: str, pred_path: str) -> list[dict[str, Any]]:
    gold_data = load_json(gold_path)
    pred_data = load_json(pred_path)

    gold_questions = {q["id"]: q for q in gold_data.get("questions", [])}
    pred_by_id = {p["id"]: p for p in pred_data if "id" in p}

    common_ids = sorted(set(gold_questions.keys()) & set(pred_by_id.keys()))
    aligned = [(gold_questions[qid], pred_by_id[qid]) for qid in common_ids]
    base_rows = build_ragas_rows(aligned)

    # Include both legacy and newer column names to maximize compatibility
    # across ragas versions and metrics.
    rows: list[dict[str, Any]] = []
    for row in base_rows:
        rows.append(
            {
                "user_input": row["user_input"],
                "response": row["response"],
                "reference": row["reference"],
                "retrieved_contexts": row["retrieved_contexts"],
                "question": row["user_input"],
                "answer": row["response"],
                "ground_truth": row["reference"],
                "contexts": row["retrieved_contexts"],
            }
        )
    return rows


def build_google_llm(model_name: str):
    from ragas.llms import llm_factory

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY before running RAGAS with Google models."
        )

    from litellm import OpenAI as LiteLLMClient

    client = LiteLLMClient(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    return llm_factory(
        model_name,
        provider="google",
        client=client,
        adapter="litellm",
    )


def build_openai_llm(model_name: str):
    from openai import OpenAI
    from ragas.llms import llm_factory

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running RAGAS with OpenAI.")

    client = OpenAI(api_key=api_key)
    return llm_factory(model_name, client=client)


def maybe_build_embeddings(provider: str):
    if provider != "google":
        return None

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        from ragas.embeddings import GoogleEmbeddings
    except Exception:
        return None

    try:
        from google import genai  # type: ignore

        client = genai.Client(api_key=api_key)
        return GoogleEmbeddings(client=client, model="gemini-embedding-001")
    except Exception:
        try:
            return GoogleEmbeddings(model="gemini-embedding-001")
        except Exception:
            return None


def instantiate_metric(metric_obj: Any, llm: Any, embeddings: Any) -> Any:
    if not callable(metric_obj):
        return metric_obj

    attempts = []
    if llm is not None and embeddings is not None:
        attempts.append({"llm": llm, "embeddings": embeddings})
    if llm is not None:
        attempts.append({"llm": llm})
    if embeddings is not None:
        attempts.append({"embeddings": embeddings})
    attempts.append({})

    for kwargs in attempts:
        try:
            return metric_obj(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue

    # Last fallback: return as-is if the imported object is already ready.
    return metric_obj


def resolve_metrics(metric_names: list[str], llm: Any, embeddings: Any) -> list[Any]:
    from ragas import metrics as ragas_metrics

    aliases = {
        "faithfulness": ["Faithfulness", "faithfulness"],
        "answer_relevancy": ["AnswerRelevancy", "answer_relevancy", "answer_relevance"],
        "response_relevancy": ["ResponseRelevancy", "response_relevancy", "response_relevance"],
        "context_precision": ["ContextPrecision", "context_precision", "context_relevance"],
        "context_recall": ["ContextRecall", "context_recall"],
        "answer_correctness": ["AnswerCorrectness", "answer_correctness"],
    }

    chosen: list[Any] = []
    for metric_name in metric_names:
        candidates = aliases.get(metric_name, [metric_name])
        found = None
        for candidate in candidates:
            if hasattr(ragas_metrics, candidate):
                found = getattr(ragas_metrics, candidate)
                break
        if found is None:
            raise RuntimeError(f"Unsupported or unavailable RAGAS metric: {metric_name}")
        chosen.append(instantiate_metric(found, llm, embeddings))
    return chosen


def extract_scores(result: Any, metric_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        for name in df.columns:
            series = df[name]
            vals = [
                float(v)
                for v in series.tolist()
                if isinstance(v, (int, float)) and not math.isnan(float(v))
            ]
            if vals:
                out[name] = float(sum(vals) / len(vals))

    if not out and hasattr(result, "keys") and hasattr(result, "get"):
        for name in result.keys():
            val = result.get(name)
            if isinstance(val, (int, float)) and not math.isnan(float(val)):
                out[str(name)] = float(val)

    # Keep the report keyed by the requested names when possible.
    normalized: dict[str, float] = {}
    for requested in metric_names:
        for key, value in out.items():
            key_norm = key.lower().replace(" ", "_")
            req_norm = requested.lower().replace(" ", "_")
            if key_norm == req_norm or req_norm in key_norm or key_norm in req_norm:
                normalized[requested] = value
                break

    return normalized or out


def run_ragas(
    gold_path: str,
    pred_path: str,
    provider: str,
    model_name: str,
    metric_names: list[str],
) -> dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate

    rows = align_rows(gold_path, pred_path)
    if not rows:
        raise RuntimeError("No aligned rows found between gold and prediction files.")

    if provider == "google":
        llm = build_google_llm(model_name)
        embeddings = maybe_build_embeddings(provider)
    elif provider == "openai":
        llm = build_openai_llm(model_name)
        embeddings = None
    else:
        raise RuntimeError(f"Unsupported provider: {provider}")

    metrics = resolve_metrics(metric_names, llm, embeddings)
    dataset = Dataset.from_list(rows)
    result = ragas_evaluate(dataset, metrics=metrics)
    scores = extract_scores(result, metric_names)

    note = None
    if len(scores) < len(metric_names):
        missing = [m for m in metric_names if m not in scores]
        note = (
            "Some RAGAS metrics returned no valid numeric values. "
            f"Missing or invalid metrics: {', '.join(missing)}."
        )

    return {
        "provider": provider,
        "model": model_name,
        "num_rows": len(rows),
        "metrics_requested": metric_names,
        "metrics": {name: scores.get(name) for name in metric_names},
        "note": note,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS-only evaluation on BioASQ outputs")
    parser.add_argument("--gold", required=True, help="Path to gold BioASQ JSON")
    parser.add_argument("--pred", required=True, help="Path to predictions JSON")
    parser.add_argument(
        "--out",
        default="outputs/ragas_report.json",
        help="Path to write RAGAS report JSON",
    )
    parser.add_argument(
        "--provider",
        choices=["google", "openai"],
        default="google",
        help="LLM provider for RAGAS judge model",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Judge model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,context_precision,context_recall",
        help="Comma-separated RAGAS metrics to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    report = run_ragas(
        gold_path=args.gold,
        pred_path=args.pred,
        provider=args.provider,
        model_name=args.model,
        metric_names=metric_names,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved RAGAS report to: {out_path}")
    for key, val in report.get("metrics", {}).items():
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")


if __name__ == "__main__":
    main()
