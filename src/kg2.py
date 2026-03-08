from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from itertools import combinations
from typing import Any

import requests
from Bio import Entrez, Medline
from tqdm import tqdm

_ENTREZ_BATCH = 200
_ENTREZ_DELAY = 0.34
_OLLAMA_URL = "http://localhost:11434/api/generate"


def _parse_pmid(url: str) -> int:
	return int(url.rstrip("/").split("/")[-1])


def _normalize_term(term: str) -> str:
	cleaned = re.sub(r"\s+", " ", term.strip().lower())
	cleaned = cleaned.strip(" .,:;!?()[]{}\"'")
	return cleaned


def _dedupe_keep_order(items: list[str]) -> list[str]:
	seen: set[str] = set()
	out: list[str] = []
	for item in items:
		if item not in seen:
			seen.add(item)
			out.append(item)
	return out


def load_questions(train_file: str) -> list[dict[str, Any]]:
	with open(train_file, "r", encoding="utf-8") as f:
		payload = json.load(f)
	return payload["questions"]


def collect_unique_pmids(questions: list[dict[str, Any]]) -> list[int]:
	pmids: set[int] = set()
	for q in questions:
		for url in q.get("documents", []):
			pmids.add(_parse_pmid(url))
	return sorted(pmids)


def _fetch_single_pmid(pmid: int) -> dict[str, str]:
	handle = Entrez.efetch(db="pubmed", id=str(pmid), rettype="medline", retmode="text")
	records = list(Medline.parse(handle))
	if not records:
		return {"title": "Title not available", "abstract": ""}
	record = records[0]
	return {
		"title": record.get("TI", "Title not available"),
		"abstract": record.get("AB", ""),
	}


def fetch_pubmed_records(pmids: list[int]) -> dict[int, dict[str, str]]:
	records: dict[int, dict[str, str]] = {}

	for start in tqdm(range(0, len(pmids), _ENTREZ_BATCH), desc="Fetching abstracts"):
		batch = pmids[start : start + _ENTREZ_BATCH]
		try:
			handle = Entrez.efetch(
				db="pubmed",
				id=",".join(str(p) for p in batch),
				rettype="medline",
				retmode="text",
			)
			for record in Medline.parse(handle):
				try:
					pmid = int(record["PMID"])
				except (KeyError, ValueError):
					continue
				records[pmid] = {
					"title": record.get("TI", "Title not available"),
					"abstract": record.get("AB", ""),
				}
		except Exception:
			# Batch fetch can fail intermittently; fall back to individual calls for resilience.
			for pmid in batch:
				try:
					records[pmid] = _fetch_single_pmid(pmid)
				except Exception:
					records[pmid] = {"title": "Title not available", "abstract": ""}
				time.sleep(_ENTREZ_DELAY)
		time.sleep(_ENTREZ_DELAY)

	return records


def _extract_json_list(text: str) -> list[str]:
	matches = re.findall(r"\[[\s\S]*?\]", text)
	for candidate in matches:
		try:
			parsed = json.loads(candidate)
			if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
				return parsed
		except json.JSONDecodeError:
			continue
	return []


def _fallback_parse_lines(text: str) -> list[str]:
	terms: list[str] = []
	for line in text.splitlines():
		token = line.strip()
		token = re.sub(r"^[\-\*\d\.\)\s]+", "", token).strip()
		if token:
			terms.append(token)
	return terms


def extract_terms_with_gemma(
	abstract: str,
	*,
	model: str,
	top_k: int,
	timeout: int,
) -> list[str]:
	prompt = (
		"Extract the most important biomedical terms from the abstract. "
		f"Return ONLY a JSON array of up to {top_k} concise noun-phrase terms. "
		"No explanations, no markdown, no numbering.\n\n"
		f"Abstract:\n{abstract}"
	)

	payload = {
		"model": model,
		"prompt": prompt,
		"stream": False,
		"options": {"temperature": 0},
	}

	response = requests.post(_OLLAMA_URL, json=payload, timeout=timeout)
	response.raise_for_status()
	data = response.json()
	raw = data.get("response", "")

	terms = _extract_json_list(raw)
	if not terms:
		terms = _fallback_parse_lines(raw)

	terms = [_normalize_term(t) for t in terms]
	terms = [t for t in terms if len(t) >= 2]
	return _dedupe_keep_order(terms)[:top_k]


def build_term_graph(
	question_docs: list[list[int]],
	records: dict[int, dict[str, str]],
	doc_terms: dict[int, list[str]],
) -> dict[str, Any]:
	node_docs: dict[str, set[int]] = defaultdict(set)
	edge_docs: dict[tuple[str, str], set[int]] = defaultdict(set)

	for pmid, terms in doc_terms.items():
		unique_terms = sorted(set(terms))
		for term in unique_terms:
			node_docs[term].add(pmid)
		for left, right in combinations(unique_terms, 2):
			edge_docs[(left, right)].add(pmid)

	nodes = {
		term: {"document_count": len(pmids)} for term, pmids in sorted(node_docs.items())
	}

	edges = []
	for (left, right), pmids in sorted(edge_docs.items()):
		edges.append(
			{
				"source": left,
				"target": right,
				"weight": len(pmids),
				"documents": sorted(pmids),
			}
		)

	documents: dict[str, Any] = {}
	for pmid, rec in sorted(records.items()):
		if pmid in doc_terms:
			documents[str(pmid)] = {
				"title": rec.get("title", "Title not available"),
				"abstract": rec.get("abstract", ""),
				"terms": doc_terms[pmid],
			}

	return {
		"meta": {
			"total_documents": len(doc_terms),
			"total_nodes": len(nodes),
			"total_edges": len(edges),
			"edge_definition": "Undirected co-occurrence within the same document",
		},
		"nodes": nodes,
		"edges": edges,
		"documents": documents,
		"question_documents": question_docs,
	}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
	if args.email:
		Entrez.email = args.email
	elif os.getenv("ENTREZ_EMAIL"):
		Entrez.email = os.getenv("ENTREZ_EMAIL")
	else:
		raise ValueError("Set --email or ENTREZ_EMAIL for NCBI Entrez requests.")

	api_key = args.api_key or os.getenv("NCBI_API_KEY")
	if api_key:
		Entrez.api_key = api_key

	questions = load_questions(args.train_file)

	question_docs: list[list[int]] = []
	for q in questions:
		pmids = [_parse_pmid(url) for url in q.get("documents", [])]
		question_docs.append(pmids)

	unique_pmids = collect_unique_pmids(questions)
	if args.limit_docs is not None:
		unique_pmids = unique_pmids[: args.limit_docs]

	records = fetch_pubmed_records(unique_pmids)

	doc_terms: dict[int, list[str]] = {}
	for pmid in tqdm(unique_pmids, desc="Extracting key terms"):
		abstract = records.get(pmid, {}).get("abstract", "").strip()
		if not abstract:
			continue
		try:
			terms = extract_terms_with_gemma(
				abstract,
				model=args.model,
				top_k=args.top_k,
				timeout=args.timeout,
			)
		except Exception:
			terms = []
		if terms:
			doc_terms[pmid] = terms

	graph = build_term_graph(question_docs, records, doc_terms)

	os.makedirs(os.path.dirname(args.output), exist_ok=True)
	with open(args.output, "w", encoding="utf-8") as f:
		json.dump(graph, f, ensure_ascii=False, indent=2)

	return graph


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build a term-level knowledge graph from BioASQ train.json using Gemma."
	)
	parser.add_argument("--train-file", default="datasets/train/train.json")
	parser.add_argument("--output", default="knowledge_graph/term_kg.json")
	parser.add_argument("--email", default=None)
	parser.add_argument("--api-key", default=None)
	parser.add_argument("--model", default="gemma2:2b")
	parser.add_argument("--top-k", type=int, default=8)
	parser.add_argument("--timeout", type=int, default=120)
	parser.add_argument("--limit-docs", type=int, default=None)
	return parser.parse_args()


if __name__ == "__main__":
	arguments = parse_args()
	graph_out = run_pipeline(arguments)
	print(
		f"Built graph: {graph_out['meta']['total_nodes']} nodes, "
		f"{graph_out['meta']['total_edges']} edges, "
		f"{graph_out['meta']['total_documents']} documents"
	)
