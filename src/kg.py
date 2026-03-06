from Bio import Entrez, Medline

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
        "text": f"{title}. {abstract}"
    }

def create_tag(file: str = "../datasets/train/train.json"):