"""Evaluate generated BioASQ-style answers against train.json gold annotations.

Supported metrics:
- List questions: entity-level F1 score
- Yes/No questions: macro F1
- Factoid questions: mean reciprocal rank (MRR)
- Summary/RAG metrics (optional via RAGAS):
  faithfulness, answer relevance, context relevance

Usage example:
  python src/evaluation/evaluate.py \
	  --gold datasets/train/train.json \
	  --pred outputs/baseline_answers.json \
	  --out outputs/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def normalize_text(text: str) -> str:
	text = text.lower().strip()
	text = re.sub(r"\s+", " ", text)
	text = re.sub(r"[^a-z0-9\-\s]", "", text)
	return text.strip()


def safe_str(value: Any) -> str:
	if value is None:
		return ""
	if isinstance(value, str):
		return value
	if isinstance(value, list):
		return " ".join(safe_str(v) for v in value)
	return str(value)


def normalize_yesno(text: str) -> str | None:
	t = normalize_text(text)
	if not t:
		return None
	if t.startswith("yes"):
		return "yes"
	if t.startswith("no"):
		return "no"
	yes_hit = re.search(r"\b(yes|true|correct)\b", t)
	no_hit = re.search(r"\b(no|false|incorrect)\b", t)
	if yes_hit and not no_hit:
		return "yes"
	if no_hit and not yes_hit:
		return "no"
	return None


def extract_candidate_items(text: str) -> list[str]:
	"""Extract a ranked list of candidate entities from free text."""
	if not text:
		return []

	raw = text.replace("\n", " ").strip()
	# Drop a leading explanatory clause before ':' if present.
	if ":" in raw:
		left, right = raw.split(":", 1)
		if len(left.split()) <= 12 and len(right.split()) >= 2:
			raw = right

	raw = re.sub(r"\band\b", ",", raw, flags=re.IGNORECASE)
	raw = re.sub(r"\bor\b", ",", raw, flags=re.IGNORECASE)
	parts = re.split(r"[,;|]", raw)
	cleaned: list[str] = []
	for p in parts:
		c = p.strip(" .\t\n\r\"'()[]")
		c = re.sub(r"\s+", " ", c)
		if c:
			cleaned.append(c)
	return cleaned


def canonicalize_gold_exact(exact_answer: Any) -> list[list[str]]:
	"""Return gold aliases as list-of-lists, regardless of original shape."""
	if exact_answer is None:
		return []
	if isinstance(exact_answer, str):
		return [[exact_answer]]
	if isinstance(exact_answer, list):
		if not exact_answer:
			return []
		# Factoid/list may be ["a"] or [["a", "b"], ["c"]]
		if all(isinstance(x, str) for x in exact_answer):
			return [[x] for x in exact_answer]
		groups: list[list[str]] = []
		for x in exact_answer:
			if isinstance(x, str):
				groups.append([x])
			elif isinstance(x, list):
				groups.append([safe_str(v) for v in x if safe_str(v).strip()])
		return groups
	return [[safe_str(exact_answer)]]


def match_alias(candidate: str, aliases: list[str]) -> bool:
	c = normalize_text(candidate)
	if not c:
		return False
	for a in aliases:
		aa = normalize_text(a)
		if not aa:
			continue
		if c == aa:
			return True
		# Allow contained match to handle explanatory spans.
		if re.search(rf"\b{re.escape(aa)}\b", c):
			return True
	return False


def list_question_f1(pred_answer: Any, gold_exact: Any) -> float:
	pred_items = extract_candidate_items(safe_str(pred_answer))
	pred_items_norm = []
	seen = set()
	for p in pred_items:
		pn = normalize_text(p)
		if pn and pn not in seen:
			seen.add(pn)
			pred_items_norm.append(p)

	gold_groups = canonicalize_gold_exact(gold_exact)
	if not gold_groups:
		return 0.0

	matched_gold = set()
	matched_pred = set()
	for pi, p in enumerate(pred_items_norm):
		for gi, aliases in enumerate(gold_groups):
			if gi in matched_gold:
				continue
			if match_alias(p, aliases):
				matched_gold.add(gi)
				matched_pred.add(pi)
				break

	tp = len(matched_gold)
	fp = max(0, len(pred_items_norm) - len(matched_pred))
	fn = max(0, len(gold_groups) - len(matched_gold))

	precision = tp / (tp + fp) if (tp + fp) else 0.0
	recall = tp / (tp + fn) if (tp + fn) else 0.0
	if precision + recall == 0:
		return 0.0
	return 2 * precision * recall / (precision + recall)


def factoid_rr(pred_answer: Any, gold_exact: Any) -> float:
	pred_candidates = extract_candidate_items(safe_str(pred_answer))
	gold_groups = canonicalize_gold_exact(gold_exact)

	if not pred_candidates or not gold_groups:
		return 0.0

	# A hit is when a candidate matches any alias in any gold group.
	for rank, cand in enumerate(pred_candidates, start=1):
		if any(match_alias(cand, aliases) for aliases in gold_groups):
			return 1.0 / rank
	return 0.0


def compute_macro_f1_yesno(gold: list[str], pred: list[str]) -> float:
	labels = ["yes", "no"]
	f1s: list[float] = []
	for label in labels:
		tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
		fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
		fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
		precision = tp / (tp + fp) if (tp + fp) else 0.0
		recall = tp / (tp + fn) if (tp + fn) else 0.0
		if precision + recall == 0:
			f1s.append(0.0)
		else:
			f1s.append(2 * precision * recall / (precision + recall))
	return sum(f1s) / len(f1s)


def parse_pmids_from_pred(item: dict[str, Any]) -> set[str]:
	pmids = item.get("pmids") or item.get("context_pmids") or []
	parsed: set[str] = set()
	for p in pmids:
		s = safe_str(p).strip()
		if not s:
			continue
		# Accept plain pmids or pubmed URLs.
		m = re.search(r"(\d{5,9})", s)
		if m:
			parsed.add(m.group(1))
	return parsed


def snippet_pmid(snippet: dict[str, Any]) -> str | None:
	doc = safe_str(snippet.get("document"))
	m = re.search(r"(\d{5,9})", doc)
	return m.group(1) if m else None


def reference_text(question: dict[str, Any]) -> str:
	exact = question.get("exact_answer")
	ideal = question.get("ideal_answer")
	if ideal:
		return safe_str(ideal)
	return safe_str(exact)


def build_ragas_rows(
	aligned: list[tuple[dict[str, Any], dict[str, Any]]]
) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	for gold_q, pred_q in aligned:
		snippets = gold_q.get("snippets") or []
		pred_pmids = parse_pmids_from_pred(pred_q)

		contexts: list[str] = []
		if snippets:
			if pred_pmids:
				for s in snippets:
					pmid = snippet_pmid(s)
					if pmid and pmid in pred_pmids:
						txt = safe_str(s.get("text")).strip()
						if txt:
							contexts.append(txt)
			if not contexts:
				contexts = [safe_str(s.get("text")).strip() for s in snippets if safe_str(s.get("text")).strip()]

		rows.append(
			{
				"user_input": safe_str(gold_q.get("body")),
				"response": safe_str(pred_q.get("generated_answer")),
				"reference": reference_text(gold_q),
				"retrieved_contexts": contexts,
			}
		)
	return rows


def try_ragas(rows: list[dict[str, Any]]) -> tuple[dict[str, float], str | None]:
	"""Return ({metric: score}, error_message)."""
	if not rows:
		return {}, "No aligned rows available for RAGAS evaluation."

	try:
		from datasets import Dataset
		from ragas import evaluate as ragas_evaluate  # type: ignore[import-not-found]
		from ragas import metrics as ragas_metrics  # type: ignore[import-not-found]
	except Exception as exc:  # pragma: no cover - depends on env
		return {}, f"RAGAS not available: {exc}"

	metric_candidates = {
		"faithfulness": ["faithfulness"],
		"answer_relevance": ["answer_relevancy", "answer_relevance"],
		"context_relevance": ["context_relevancy", "context_relevance", "context_precision"],
	}

	chosen = {}
	for out_name, names in metric_candidates.items():
		metric_obj = None
		for name in names:
			if hasattr(ragas_metrics, name):
				metric_obj = getattr(ragas_metrics, name)
				break
		if metric_obj is not None:
			chosen[out_name] = metric_obj

	if not chosen:
		return {}, "RAGAS imported, but expected metrics are unavailable in this installed version."

	try:
		ds = Dataset.from_list(rows)
		result = ragas_evaluate(ds, metrics=list(chosen.values()))
	except Exception as exc:  # pragma: no cover - depends on env and LLM config
		return {}, f"RAGAS evaluation failed: {exc}"

	# Try robust extraction across ragas versions.
	out: dict[str, float] = {}
	if hasattr(result, "to_pandas"):
		df = result.to_pandas()
		for out_name, metric_obj in chosen.items():
			metric_name = getattr(metric_obj, "name", None)
			if metric_name and metric_name in df.columns:
				vals = [v for v in df[metric_name].tolist() if isinstance(v, (float, int)) and not math.isnan(v)]
				if vals:
					out[out_name] = float(sum(vals) / len(vals))
	if not out:
		# Fallback if result behaves like dict.
		for out_name, metric_obj in chosen.items():
			metric_name = getattr(metric_obj, "name", None)
			if metric_name and hasattr(result, "get"):
				val = result.get(metric_name)
				if isinstance(val, (float, int)):
					out[out_name] = float(val)

	return out, None if out else "RAGAS ran but no metric values could be extracted."


def evaluate(gold_path: str, pred_path: str) -> dict[str, Any]:
	gold_data = load_json(gold_path)
	pred_data = load_json(pred_path)

	gold_questions = {q["id"]: q for q in gold_data.get("questions", [])}
	pred_by_id = {p["id"]: p for p in pred_data if "id" in p}

	common_ids = sorted(set(gold_questions.keys()) & set(pred_by_id.keys()))

	yes_gold: list[str] = []
	yes_pred: list[str] = []
	list_scores: list[float] = []
	factoid_rrs: list[float] = []

	counts = defaultdict(int)
	skipped = defaultdict(int)
	aligned_for_ragas: list[tuple[dict[str, Any], dict[str, Any]]] = []

	for qid in common_ids:
		gold_q = gold_questions[qid]
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
				# Keep invalid pred as opposite class by using a sentinel,
				# which counts as error for both binary one-vs-rest F1 computations.
				p = "unknown"
			yes_gold.append(g)
			yes_pred.append(p)
		elif qtype == "list":
			score = list_question_f1(pred_answer, exact)
			list_scores.append(score)
		elif qtype == "factoid":
			rr = factoid_rr(pred_answer, exact)
			factoid_rrs.append(rr)

		aligned_for_ragas.append((gold_q, pred_q))

	report: dict[str, Any] = {
		"num_gold_questions": len(gold_questions),
		"num_pred_answers": len(pred_by_id),
		"num_aligned": len(common_ids),
		"num_missing_predictions": len(set(gold_questions.keys()) - set(pred_by_id.keys())),
		"counts_by_type_aligned": dict(counts),
		"skipped": dict(skipped),
		"metrics": {
			"list_f1": (sum(list_scores) / len(list_scores)) if list_scores else None,
			"yesno_macro_f1": compute_macro_f1_yesno(yes_gold, yes_pred) if yes_gold else None,
			"factoid_mrr": (sum(factoid_rrs) / len(factoid_rrs)) if factoid_rrs else None,
		},
	}

	ragas_rows = build_ragas_rows(aligned_for_ragas)
	ragas_scores, ragas_error = try_ragas(ragas_rows)
	if ragas_scores:
		report["metrics"].update(ragas_scores)
	if ragas_error:
		report["ragas_note"] = ragas_error

	return report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate generated BioASQ-style answers")
	parser.add_argument(
		"--gold",
		default="datasets/train/train.json",
		help="Path to gold train.json",
	)
	parser.add_argument(
		"--pred",
		required=True,
		help="Path to generated answers JSON (id/type/generated_answer[/pmids])",
	)
	parser.add_argument(
		"--out",
		default="outputs/evaluation_report.json",
		help="Path to write evaluation JSON report",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	report = evaluate(args.gold, args.pred)

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(report, f, indent=2, ensure_ascii=False)

	print(f"Saved evaluation report to: {out_path}")
	for key, val in report.get("metrics", {}).items():
		if isinstance(val, float):
			print(f"{key}: {val:.4f}")
		else:
			print(f"{key}: {val}")
	if report.get("ragas_note"):
		print(f"RAGAS note: {report['ragas_note']}")


if __name__ == "__main__":
	main()
