from Bio import Entrez, Medline
import json, time, sys, os
from collections import defaultdict
import re
from urllib.error import HTTPError, URLError
from http.client import IncompleteRead, RemoteDisconnected
from tqdm import tqdm
import pickle

Entrez.email = "kssaisankalp.davey@research.iiit.ac.in"
_ENTREZ_BATCH = 200   # max IDs per efetch request
_ENTREZ_DELAY = 0.34  # ~3 req/s without API key
_ENTREZ_RETRIES = 4
_ENTREZ_BACKOFF = 1.5


def _question_words(text: str) -> set[str]:
    """Lowercased word tokens from a question body."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _entrez_efetch_with_retries(id_value: str):
    """Open an Entrez handle with bounded retries for transient network failures."""
    last_exc = None
    for attempt in range(_ENTREZ_RETRIES):
        try:
            return Entrez.efetch(db="pubmed", id=id_value, rettype="medline", retmode="text")
        except (RemoteDisconnected, IncompleteRead, URLError, HTTPError, OSError) as exc:
            last_exc = exc
            sleep_s = _ENTREZ_DELAY + (_ENTREZ_BACKOFF ** attempt)
            time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise RuntimeError("Entrez efetch failed with unknown error")


def build_dataset_index(pmids: set) -> dict:
    """Batch-fetch title+abstract for all PMIDs from Entrez (200 IDs per request)."""
    index: dict[int, dict] = {}
    pmid_list = list(pmids)
    for start in tqdm(range(0, len(pmid_list), _ENTREZ_BATCH), desc="Fetching from Entrez"):
        batch = pmid_list[start : start + _ENTREZ_BATCH]
        try:
            handle = _entrez_efetch_with_retries(",".join(str(p) for p in batch))
            for record in Medline.parse(handle):
                try:
                    pmid = int(record["PMID"])
                except (KeyError, ValueError):
                    continue
                index[pmid] = {
                    "title": record.get("TI", "Title not available"),
                    "abstract": record.get("AB", ""),
                }
        except Exception as e:
            print(f"  Warning: batch fetch failed ({e}); retrying PMIDs individually.")
            for pmid in batch:
                if pmid in index:
                    continue
                index[pmid] = fetch_from_entrez(str(pmid))
                time.sleep(_ENTREZ_DELAY)
        time.sleep(_ENTREZ_DELAY)
    return index


def fetch_from_entrez(pmid: str) -> dict:
    """Fetch title and abstract for a single paper from the Entrez API (fallback)."""
    try:
        handle = _entrez_efetch_with_retries(pmid)
        records = list(Medline.parse(handle))
        if not records:
            return {"title": "Title not available", "abstract": "Abstract not available"}
        record = records[0]
        return {
            "title": record.get("TI", "Title not available"),
            "abstract": record.get("AB", "Abstract not available"),
        }
    except Exception as e:
        # Keep pipeline running even if one PMID is temporarily unavailable.
        print(f"  Warning: single PMID fetch failed for {pmid} ({e}).")
        return {"title": "Title not available", "abstract": "Abstract not available"}


def get_paper_data(pmid: int, dataset_index: dict) -> dict:
    """Return paper data from the batch index, falling back to a single Entrez call."""
    entry = dataset_index.get(pmid)
    if entry and entry.get("abstract"):
        return entry
    # Missing or abstract-less — fetch individually.
    result = fetch_from_entrez(str(pmid))
    time.sleep(_ENTREZ_DELAY)
    return result


def create_tag(file: str = "datasets/train/train.json") -> tuple:
    with open(file, "r", encoding="utf-8") as f:
        dct = json.load(f)["questions"]

    # Collect every PMID referenced across all questions.
    all_pmids: set[int] = set()
    for d in dct:
        for url in d["documents"]:
            all_pmids.add(int(url.rstrip("/").split("/")[-1]))

    # Build a local index for those PMIDs from the ncbi/pubmed dataset.
    dataset_index = build_dataset_index(all_pmids)

    tag: dict[int, dict] = {}
    adj: dict[int, set] = defaultdict(set)
    edge_question_words: dict[tuple[int, int], set[str]] = defaultdict(set)

    for d in tqdm(dct, desc="Processing questions"):
        ids: list[int] = []
        seen: set[int] = set()
        q_words = _question_words(d.get("body", ""))
        for url in d["documents"]:
            pmid = int(url.rstrip("/").split("/")[-1])
            if pmid not in seen:
                seen.add(pmid)
                ids.append(pmid)
            if pmid not in tag:
                tag[pmid] = get_paper_data(pmid, dataset_index)

        # Build co-citation edges between every pair of PMIDs in this question.
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                u, v = ids[i], ids[j]
                adj[u].add(v)
                adj[v].add(u)
                edge_question_words[(min(u, v), max(u, v))].update(q_words)

    return dict(adj), tag, dict(edge_question_words)


if __name__ == "__main__":
    adj, tag, edge_question_words = create_tag()
    print(adj)
    with open("knowledge_graph/kg.dat", "wb+") as f:
        pickle.dump({"adjacency_list": adj,
                     "tag": tag,
                     "edge_list": edge_question_words}, f)