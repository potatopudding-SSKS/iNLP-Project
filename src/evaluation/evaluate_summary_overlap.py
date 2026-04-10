"""Evaluate BioASQ summary answers using ROUGE overlap metrics.

This script only evaluates questions of type "summary". It aligns prediction
rows to gold questions by BioASQ question id and scores generated answers
against the gold ideal answer text.

Example:
  python src/evaluation/evaluate_summary_overlap.py \
    --gold datasets/train/train.json \
    --pred outputs/rag_baseline_answers.json \
    --out outputs/rag_baseline_summary_overlap.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(safe_str(v) for v in value)
    return str(value)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    return re.findall(r"\b\w+\b", text)


def get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(reference: list[str], prediction: list[str], n: int) -> dict[str, float]:
    ref_ngrams = get_ngrams(reference, n)
    pred_ngrams = get_ngrams(prediction, n)
    overlap = sum(min(count, pred_ngrams.get(ngram, 0)) for ngram, count in ref_ngrams.items())

    ref_total = sum(ref_ngrams.values())
    pred_total = sum(pred_ngrams.values())

    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(reference: list[str], prediction: list[str]) -> dict[str, float]:
    lcs = lcs_length(reference, prediction)
    precision = lcs / len(prediction) if prediction else 0.0
    recall = lcs / len(reference) if reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_metric(rows: list[dict[str, float]], key: str) -> float | None:
    if not rows:
        return None
    vals = [row[key] for row in rows]
    return sum(vals) / len(vals) if vals else None


def evaluate_summary_overlap(gold_path: str, pred_path: str) -> dict[str, Any]:
    gold_data = load_json(gold_path)
    pred_data = load_json(pred_path)

    gold_questions = {
        q["id"]: q
        for q in gold_data.get("questions", [])
        if safe_str(q.get("type")).lower() == "summary"
    }
    pred_by_id = {p["id"]: p for p in pred_data if "id" in p}

    common_ids = sorted(set(gold_questions.keys()) & set(pred_by_id.keys()))

    rouge1_rows: list[dict[str, float]] = []
    rouge2_rows: list[dict[str, float]] = []
    rougel_rows: list[dict[str, float]] = []

    for qid in common_ids:
        gold_q = gold_questions[qid]
        pred_q = pred_by_id[qid]

        reference = tokenize(safe_str(gold_q.get("ideal_answer")))
        prediction = tokenize(safe_str(pred_q.get("generated_answer")))

        rouge1_rows.append(rouge_n(reference, prediction, 1))
        rouge2_rows.append(rouge_n(reference, prediction, 2))
        rougel_rows.append(rouge_l(reference, prediction))

    report = {
        "num_gold_summary_questions": len(gold_questions),
        "num_pred_answers": len(pred_by_id),
        "num_aligned_summary": len(common_ids),
        "num_missing_summary_predictions": len(set(gold_questions.keys()) - set(pred_by_id.keys())),
        "metrics": {
            "rouge_1_precision": mean_metric(rouge1_rows, "precision"),
            "rouge_1_recall": mean_metric(rouge1_rows, "recall"),
            "rouge_1_f1": mean_metric(rouge1_rows, "f1"),
            "rouge_2_precision": mean_metric(rouge2_rows, "precision"),
            "rouge_2_recall": mean_metric(rouge2_rows, "recall"),
            "rouge_2_f1": mean_metric(rouge2_rows, "f1"),
            "rouge_l_precision": mean_metric(rougel_rows, "precision"),
            "rouge_l_recall": mean_metric(rougel_rows, "recall"),
            "rouge_l_f1": mean_metric(rougel_rows, "f1"),
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate only summary answers with BLEU and ROUGE")
    parser.add_argument("--gold", required=True, help="Path to gold BioASQ JSON")
    parser.add_argument("--pred", required=True, help="Path to generated answers JSON")
    parser.add_argument(
        "--out",
        default="outputs/summary_overlap_report.json",
        help="Path to write evaluation JSON report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_summary_overlap(args.gold, args.pred)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved summary overlap report to: {out_path}")
    for key, val in report.get("metrics", {}).items():
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")


if __name__ == "__main__":
    main()
