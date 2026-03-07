from Bio import Entrez, Medline
import json, time
from collections import defaultdict

Entrez.email = "kssaisankalp.davey@research.iiit.ac.in"

def fetch_pubmed_data(id: str) -> dict:
    handle = Entrez.efetch(db="pubmed", id=id, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    if not records:
        return None
    record = records[0]
    title = record.get("TI", "Title not available")
    abstract = record.get("AB", "Abstract not available")
    return {
        "title": title,
        "abstract": abstract,
    }

def create_tag(file: str = "datasets/train/train.json") -> tuple:
    tag = dict()
    adj = defaultdict(set)
    with open(file, "r", encoding="utf-8") as f:
        dct = json.load(f)["questions"]
        for d in dct:
            ids = set()
            docs = d["documents"]
            for url in docs:
                id = url.split("/")[-1]
                ids.add(int(id))
                if int(id) in tag:
                    continue
                node = fetch_pubmed_data(id)
                time.sleep(0.33)
                tag[int(id)] = node
            ids = list(ids)
            for i in range(len(ids)):
                for j in range(i+1, len(ids), 1):
                    u = ids[i]
                    v = ids[j]
                    adj[u].add(v)
                    adj[v].add(u)
    return dict(adj), tag

print(create_tag()[0])