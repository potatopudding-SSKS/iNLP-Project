"""Evaluate one shared prediction file against all BioASQ test gold splits.

This is useful when a single prediction file contains answers for the combined
test set, while the gold annotations are stored in multiple files such as
13B1_golden.json, 13B2_golden.json, etc.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from evaluate import (
        compute_macro_f1_yesno,
        factoid_rr,
        list_question_f1,
        load_json,
        normalize_yesno,
        safe_str,
    )
except ModuleNotFoundError:
    from src.evaluation.evaluate import (
        compute_macro_f1_yesno,
        factoid_rr,
        list_question_f1,
        load_json,
        normalize_yesno,
        safe_str,
    )


def evaluate_in_memory(gold_questions: list[dict[str, Any]], pred_data: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = {q["id"]: q for q in gold_questions}
    pred_by_id = {p["id"]: p for p in pred_data if "id" in p}

    common_ids = sorted(set(gold_by_id.keys()) & set(pred_by_id.keys()))

    yes_gold: list[str] = []
    yes_pred: list[str] = []
    list_scores: list[float] = []
    factoid_rrs: list[float] = []

    counts = defaultdict(int)
    skipped = defaultdict(int)

    for qid in common_ids:
        gold_q = gold_by_id[qid]
        pred_q = pred_by_id[qid]
        qtype = safe_str(gold_q.get("type")).lower()
        counts[qtype] += 1

        pred_answer = pred_q.get("generated_answer")
        exact = gold_q.get("exact_answer")

        if qtype == "yesno":
            g = normalize_yesno(safe_str(exact))
            p = normalize_yesno(safe_str(pred_answer))
            if g is None:
                skipped["yesno_missing_gold"] += 1
                continue
            if p is None:
                p = "unknown"
            yes_gold.append(g)
            yes_pred.append(p)
        elif qtype == "list":
            list_scores.append(list_question_f1(pred_answer, exact))
        elif qtype == "factoid":
            factoid_rrs.append(factoid_rr(pred_answer, exact))

    return {
        "num_gold_questions": len(gold_by_id),
        "num_pred_answers": len(pred_by_id),
        "num_aligned": len(common_ids),
        "num_missing_predictions": len(set(gold_by_id.keys()) - set(pred_by_id.keys())),
        "counts_by_type_aligned": dict(counts),
        "skipped": dict(skipped),
        "metrics": {
            "list_f1": (sum(list_scores) / len(list_scores)) if list_scores else None,
            "yesno_macro_f1": compute_macro_f1_yesno(yes_gold, yes_pred) if yes_gold else None,
            "factoid_mrr": (sum(factoid_rrs) / len(factoid_rrs)) if factoid_rrs else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one prediction file against all test gold splits")
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to shared predictions JSON",
    )
    parser.add_argument(
        "--gold-files",
        nargs="+",
        default=[
            "datasets/test/13B1_golden.json",
            "datasets/test/13B2_golden.json",
            "datasets/test/13B3_golden.json",
            "datasets/test/13B4_golden.json",
        ],
        help="Gold split files to evaluate against",
    )
    parser.add_argument(
        "--out",
        default="outputs/test_bundle_eval.json",
        help="Path to write combined evaluation JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_data = load_json(args.pred)
    if not isinstance(pred_data, list):
        raise ValueError("Prediction file must contain a list of answer objects.")

    split_reports: dict[str, Any] = {}
    merged_questions: list[dict[str, Any]] = []

    for gold_file in args.gold_files:
        gold_data = load_json(gold_file)
        questions = gold_data.get("questions", [])
        split_name = Path(gold_file).stem.replace("_golden", "")
        split_reports[split_name] = evaluate_in_memory(questions, pred_data)
        merged_questions.extend(questions)

    overall_report = evaluate_in_memory(merged_questions, pred_data)
    report = {
        "pred_file": args.pred,
        "gold_files": args.gold_files,
        "overall": overall_report,
        "splits": split_reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved test-bundle evaluation to: {out_path}")
    print("\nOverall metrics:")
    for key, val in overall_report.get("metrics", {}).items():
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")

    print("\nPer-split metrics:")
    for split_name, split_report in split_reports.items():
        print(f"{split_name}:")
        for key, val in split_report.get("metrics", {}).items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
