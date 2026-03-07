from Bio import Entrez, Medline
import json, time, sys, os
from collections import defaultdict
from tqdm import tqdm
import pickle

Entrez.email = "kssaisankalp.davey@research.iiit.ac.in"
_ENTREZ_BATCH = 200   # max IDs per efetch request
_ENTREZ_DELAY = 0.34  # ~3 req/s without API key


def build_dataset_index(pmids: set) -> dict:
    """Batch-fetch title+abstract for all PMIDs from Entrez (200 IDs per request)."""
    index: dict[int, dict] = {}
    pmid_list = list(pmids)
    for start in tqdm(range(0, len(pmid_list), _ENTREZ_BATCH), desc="Fetching from Entrez"):
        batch = pmid_list[start : start + _ENTREZ_BATCH]
        try:
            handle = Entrez.efetch(
                db="pubmed", id=",".join(str(p) for p in batch),
                rettype="medline", retmode="text"
            )
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
            print(f"  Warning: batch fetch failed ({e}); will retry individually.")
        time.sleep(_ENTREZ_DELAY)
    return index


def fetch_from_entrez(pmid: str) -> dict:
    """Fetch title and abstract for a single paper from the Entrez API (fallback)."""
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    if not records:
        return {"title": "Title not available", "abstract": "Abstract not available"}
    record = records[0]
    return {
        "title": record.get("TI", "Title not available"),
        "abstract": record.get("AB", "Abstract not available"),
    }


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

    for d in tqdm(dct, desc="Processing questions"):
        ids: list[int] = []
        seen: set[int] = set()
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

    return dict(adj), tag


if __name__ == "__main__":
    adj, tag = create_tag()
    print(adj)
    with open("knowledge_graph/kg.dat", "wb+") as f:
        pickle.dump((adj, tag), f)