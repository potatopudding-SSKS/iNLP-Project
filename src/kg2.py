"""
Knowledge Graph Builder for BioASQ Dataset

This module builds a heterogeneous knowledge graph with:
- Document nodes: PMID, abstract, MeSH terms (from HuggingFace PubMed)
- Entity nodes: name (extracted from questions via local Gemma LLM)
- Edges:
  - "related-to": Document <-> Entity (from question association)
  - Inter-entity: labeled relations (via PubMedBERT relation extraction)

Storage: graph-tool (.gt format) for efficiency
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import requests
from tqdm import tqdm

# Lazy imports for heavy dependencies
_graph_tool = None
_datasets = None
_transformers = None
_torch = None

_OLLAMA_URL = "http://localhost:11434/api/generate"


def _lazy_import_graph_tool():
    global _graph_tool
    if _graph_tool is None:
        # Import only core module to avoid GTK/matplotlib drawing dependencies
        import graph_tool
        _graph_tool = graph_tool
    return _graph_tool


def _lazy_import_datasets():
    global _datasets
    if _datasets is None:
        from datasets import load_dataset
        _datasets = load_dataset
    return _datasets


def _lazy_import_transformers():
    global _transformers, _torch
    if _transformers is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        _transformers = (AutoModelForSeq2SeqLM, AutoTokenizer)
        _torch = torch
    return _transformers, _torch


# ============================================================================
# Utility Functions
# ============================================================================

def _parse_pmid(url: str) -> int:
    """Extract PMID integer from PubMed URL."""
    return int(url.rstrip("/").split("/")[-1])


def _normalize_text(text: str) -> str:
    """Normalize whitespace and strip punctuation."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return cleaned.strip(" .,:;!?()[]{}\"'")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    """Deduplicate list while preserving order."""
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def _extract_json_list(text: str) -> list[str]:
    """Extract JSON array from LLM response text."""
    for match in re.findall(r"\[[\s\S]*?\]", text):
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return parsed
        except json.JSONDecodeError:
            continue
    return []


def _fallback_parse_lines(text: str) -> list[str]:
    """Fallback parser for non-JSON LLM output."""
    terms = []
    for line in text.splitlines():
        token = re.sub(r"^[\-\*\d\.\)\s]+", "", line.strip()).strip()
        if token:
            terms.append(token)
    return terms


# ============================================================================
# Data Loading
# ============================================================================

def load_questions(train_file: str) -> list[dict[str, Any]]:
    """Load questions from BioASQ train.json."""
    with open(train_file, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def collect_pmid_to_questions(questions: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Map each PMID to the question indices it appears in."""
    pmid_to_q: dict[int, list[int]] = {}
    for q_idx, q in enumerate(questions):
        for url in q.get("documents", []):
            pmid = _parse_pmid(url)
            pmid_to_q.setdefault(pmid, []).append(q_idx)
    return pmid_to_q


def fetch_pubmed_via_entrez(
    pmids: set[int],
    cache_file: str | None = None,
    email: str | None = None,
    api_key: str | None = None,
    batch_size: int = 200,
) -> dict[int, dict[str, Any]]:
    """
    Fetch PubMed records via NCBI Entrez API.
    
    Returns dict[pmid] = {
        "abstract": str,
        "title": str,
        "mesh_terms": list[str]
    }
    """
    import time
    import xml.etree.ElementTree as ET
    
    # Check cache first
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached PubMed data from {cache_file}")
        with open(cache_file, "r") as f:
            cached = json.load(f)
        return {int(k): v for k, v in cached.items()}
    
    # Set up Entrez credentials
    entrez_email = email or os.getenv("ENTREZ_EMAIL")
    if not entrez_email:
        print("Warning: No email set for Entrez. Set ENTREZ_EMAIL env var for better rate limits.")
        entrez_email = "anonymous@example.com"
    
    entrez_api_key = api_key or os.getenv("NCBI_API_KEY")
    
    print(f"Fetching {len(pmids)} PubMed records via Entrez API...")
    if entrez_api_key:
        print("Using API key (10 requests/sec limit)")
        delay = 0.1
    else:
        print("No API key (3 requests/sec limit). Set NCBI_API_KEY for faster fetching.")
        delay = 0.34
    
    records: dict[int, dict[str, Any]] = {}
    pmid_list = sorted(pmids)
    
    for start in tqdm(range(0, len(pmid_list), batch_size), desc="Fetching from Entrez"):
        batch = pmid_list[start:start + batch_size]
        
        # Build Entrez efetch URL
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(str(p) for p in batch),
            "rettype": "xml",
            "retmode": "xml",
            "email": entrez_email,
        }
        if entrez_api_key:
            params["api_key"] = entrez_api_key
        
        try:
            response = requests.get(base_url, params=params, timeout=60)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Get PMID
                    pmid_elem = article.find(".//PMID")
                    if pmid_elem is None:
                        continue
                    pmid = int(pmid_elem.text)
                    
                    # Get title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    
                    # Get abstract (may have multiple AbstractText elements)
                    abstract_parts = []
                    for abs_text in article.findall(".//AbstractText"):
                        if abs_text.text:
                            # Include label if present (e.g., "BACKGROUND:", "METHODS:")
                            label = abs_text.get("Label", "")
                            if label:
                                abstract_parts.append(f"{label}: {abs_text.text}")
                            else:
                                abstract_parts.append(abs_text.text)
                    abstract = " ".join(abstract_parts)
                    
                    # Get MeSH terms
                    mesh_terms = []
                    for mesh_heading in article.findall(".//MeshHeading"):
                        descriptor = mesh_heading.find("DescriptorName")
                        if descriptor is not None and descriptor.text:
                            mesh_terms.append(descriptor.text)
                    
                    records[pmid] = {
                        "abstract": abstract,
                        "title": title,
                        "mesh_terms": mesh_terms,
                    }
                    
                except (ValueError, AttributeError) as e:
                    continue
                    
        except Exception as e:
            print(f"Warning: Batch fetch failed: {e}")
            # Continue with next batch
        
        time.sleep(delay)
    
    print(f"Fetched {len(records)}/{len(pmids)} PubMed records")
    
    # Cache results
    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(records, f)
        print(f"Cached to {cache_file}")
    
    return records


# ============================================================================
# Entity Extraction (via Local Gemma)
# ============================================================================

def extract_entities_from_question(
    question: str,
    model: str = "gemma2:2b",
    top_k: int = 10,
    timeout: int = 60,
) -> list[str]:
    """
    Extract important biomedical entities from a question using local Gemma.
    """
    prompt = (
        "Extract the most important biomedical entities (diseases, genes, proteins, "
        "drugs, symptoms, anatomical terms) from this question.\n"
        f"Return ONLY a JSON array of up to {top_k} entity names.\n"
        "No explanations, no markdown.\n\n"
        f"Question: {question}"
    )
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    
    try:
        response = requests.post(_OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        raw = response.json().get("response", "")
        
        entities = _extract_json_list(raw)
        if not entities:
            entities = _fallback_parse_lines(raw)
        
        entities = [_normalize_text(e) for e in entities]
        entities = [e for e in entities if len(e) >= 2]
        return _dedupe_keep_order(entities)[:top_k]
        
    except Exception as e:
        print(f"Warning: Entity extraction failed: {e}")
        return []


# ============================================================================
# Relation Extraction (via PubMedBERT-based model)
# ============================================================================

class RelationExtractor:
    """
    Extract relations between entities using a biomedical relation extraction model.
    Uses REBEL fine-tuned on biomedical data or similar.
    """
    
    def __init__(self, model_name: str = "Babelscape/rebel-large"):
        """
        Initialize relation extractor.
        
        Note: REBEL is trained on Wikipedia, but we use it as a baseline.
        For better biomedical results, consider fine-tuned models like:
        - allenai/scibert_scivocab_cased + custom RE head
        - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
        """
        (AutoModelForSeq2SeqLM, AutoTokenizer), torch = _lazy_import_transformers()
        
        print(f"Loading relation extraction model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "xpu"
        self.model.to(self.device)
        print(f"Relation extractor on device: {self.device}")
    
    def extract_relations(
        self,
        text: str,
        entities: list[str] | None = None,
    ) -> list[tuple[str, str, str]]:
        """
        Extract (subject, relation, object) triples from text.
        
        If entities provided, filter to only relations involving those entities.
        """
        _, torch = _lazy_import_transformers()
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=3,
                num_return_sequences=1,
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Parse REBEL output format: <triplet> subj <subj> rel <obj> obj
        triples = self._parse_rebel_output(decoded)
        
        # Filter by entities if provided
        if entities:
            entity_set = {_normalize_text(e) for e in entities}
            triples = [
                (s, r, o) for s, r, o in triples
                if _normalize_text(s) in entity_set or _normalize_text(o) in entity_set
            ]
        
        return triples
    
    def _parse_rebel_output(self, text: str) -> list[tuple[str, str, str]]:
        """Parse REBEL's special token format into triples."""
        triples = []
        
        # REBEL format: <triplet> subject <subj> relation <obj> object
        triplet_pattern = r"<triplet>\s*(.+?)\s*<subj>\s*(.+?)\s*<obj>\s*(.+?)(?=<triplet>|</s>|$)"
        
        for match in re.finditer(triplet_pattern, text):
            subject = match.group(1).strip()
            relation = match.group(2).strip()
            obj = match.group(3).strip()
            relation, obj = obj, relation
            if subject and relation and obj:
                triples.append((subject, relation, obj))
        
        return triples


# ============================================================================
# Graph Construction
# ============================================================================

def build_knowledge_graph(
    questions: list[dict[str, Any]],
    pubmed_records: dict[int, dict[str, Any]],
    question_entities: dict[int, list[str]],
    entity_relations: list[tuple[str, str, str]],
    pmid_to_questions: dict[int, list[int]],
) -> Any:
    """
    Build heterogeneous knowledge graph using graph-tool.
    
    Node types:
    - document: PMID, abstract, mesh_terms
    - entity: name
    
    Edge types:
    - related-to: document <-> entity
    - [relation]: entity <-> entity (labeled)
    """
    gt = _lazy_import_graph_tool()
    
    g = gt.Graph(directed=False)
    
    # Vertex properties
    vp_type = g.new_vertex_property("string")  # "document" or "entity"
    vp_id = g.new_vertex_property("string")    # PMID or entity name
    vp_abstract = g.new_vertex_property("string")
    vp_title = g.new_vertex_property("string")
    vp_mesh = g.new_vertex_property("vector<string>")
    
    # Edge properties
    ep_type = g.new_edge_property("string")  # "related-to" or relation label
    
    # Track vertices
    doc_vertices: dict[int, Any] = {}  # pmid -> vertex
    entity_vertices: dict[str, Any] = {}  # normalized name -> vertex
    
    print("Building graph nodes...")
    
    # Add document nodes
    for pmid, record in tqdm(pubmed_records.items(), desc="Adding document nodes"):
        v = g.add_vertex()
        vp_type[v] = "document"
        vp_id[v] = str(pmid)
        vp_abstract[v] = record.get("abstract", "")
        vp_title[v] = record.get("title", "")
        vp_mesh[v] = record.get("mesh_terms", [])
        doc_vertices[pmid] = v
    
    # Collect all unique entities
    all_entities: set[str] = set()
    for entities in question_entities.values():
        all_entities.update(_normalize_text(e) for e in entities)

    # all_entities: set[str] = set()
    for e1,_,e2 in entity_relations:
        all_entities.update(_normalize_text(e) for e in [e1, e2])
    
    # Add entity nodes
    for entity_name in tqdm(sorted(all_entities), desc="Adding entity nodes"):
        v = g.add_vertex()
        vp_type[v] = "entity"
        vp_id[v] = entity_name
        vp_abstract[v] = ""
        vp_title[v] = ""
        vp_mesh[v] = []
        entity_vertices[entity_name] = v
    
    print("Building graph edges...")
    
    # Add document-entity edges ("related-to")
    for q_idx, entities in tqdm(question_entities.items(), desc="Adding doc-entity edges"):
        # Get all PMIDs for this question
        question_pmids = [
            _parse_pmid(url) 
            for url in questions[q_idx].get("documents", [])
        ]
        
        for pmid in question_pmids:
            if pmid not in doc_vertices:
                continue
            doc_v = doc_vertices[pmid]
            
            for entity in entities:
                norm_entity = _normalize_text(entity)
                if norm_entity not in entity_vertices:
                    continue
                entity_v = entity_vertices[norm_entity]
                
                # Add edge
                e = g.add_edge(doc_v, entity_v)
                ep_type[e] = "related-to"
    
    # Add entity-entity edges (from relation extraction)
    for subj, relation, obj in tqdm(entity_relations, desc="Adding entity-entity edges"):
        norm_subj = _normalize_text(subj)
        norm_obj = _normalize_text(obj)
        
        if norm_subj in entity_vertices and norm_obj in entity_vertices:
            subj_v = entity_vertices[norm_subj]
            obj_v = entity_vertices[norm_obj]
            e = g.add_edge(subj_v, obj_v)
            ep_type[e] = relation
    
    # Internalize properties
    g.vertex_properties["type"] = vp_type
    g.vertex_properties["id"] = vp_id
    g.vertex_properties["abstract"] = vp_abstract
    g.vertex_properties["title"] = vp_title
    g.vertex_properties["mesh_terms"] = vp_mesh
    g.edge_properties["relation"] = ep_type
    
    return g


# ============================================================================
# Graph Utilities
# ============================================================================

def load_graph(path: str) -> Any:
    """Load a graph-tool graph from .gt file."""
    gt = _lazy_import_graph_tool()
    return gt.load_graph(path)


def find_node_by_id(g: Any, target_id: str) -> Any | None:
    """Find a vertex by its ID (PMID or entity name)."""
    vp_id = g.vp["id"]
    target_lower = target_id.lower()
    for v in g.vertices():
        if vp_id[v] == target_id or vp_id[v] == target_lower:
            return v
    return None


def bfs(g: Any, start_id: str, max_steps: int) -> list[tuple[str, int, str]]:
    
    from collections import deque
    
    start_v = find_node_by_id(g, start_id)
    if start_v is None:
        raise ValueError(f"Node '{start_id}' not found")
    
    vp_id = g.vp["id"]
    vp_type = g.vp["type"]
    ep_type = g.ep["relation"]
    
    visited: set[int] = {int(start_v)}
    queue: deque[tuple[Any, int]] = deque([(start_v, 0)])
    result: list[tuple[str, int, str]] = [(vp_id[start_v], 0, vp_type[start_v])]
    
    while queue:
        current_v, depth = queue.popleft()
        
        if depth >= max_steps:
            continue
        
        for e in current_v.all_edges():
            neighbor = e.target() if e.source() == current_v else e.source()
            neighbor_idx = int(neighbor)
            
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                queue.append((neighbor, depth + 1))
                result.append((depth+1, vp_id[neighbor], vp_id[current_v], ep_type[e]))
    
    return result


# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """Main pipeline execution."""
    
    # Load questions
    print(f"Loading questions from {args.train_file}...")
    questions = load_questions(args.train_file)
    print(f"Loaded {len(questions)} questions")
    questions = questions[:10]
    if args.limit_questions:
        # questions = questions[:args.limit_questions]
        print(f"Limited to {len(questions)} questions")
    
    # Collect PMIDs
    pmid_to_questions = collect_pmid_to_questions(questions)
    all_pmids = set(pmid_to_questions.keys())
    print(f"Found {len(all_pmids)} unique PMIDs")
    
    # Fetch PubMed records from HuggingFace
    cache_file = os.path.join(
        os.path.dirname(args.output) or ".",
        "pubmed_cache.json"
    )
    pubmed_records = fetch_pubmed_via_entrez(
        all_pmids, 
        cache_file=cache_file,
        email=args.email,
        api_key=args.api_key,
    )
    
    # Extract entities from questions
    print("Extracting entities from questions...")
    question_entities: dict[int, list[str]] = {}
    
    for q_idx, q in enumerate(tqdm(questions, desc="Entity extraction")):
        body = q.get("body", "")
        if not body:
            continue
        
        entities = extract_entities_from_question(
            body,
            model=args.gemma_model,
            top_k=args.entity_top_k,
            timeout=args.timeout,
        )
        if entities:
            question_entities[q_idx] = entities
    
    print(f"Extracted entities for {len(question_entities)}/{len(questions)} questions")
    
    # Extract relations
    print("Extracting relations from questions...")
    all_relations: list[tuple[str, str, str]] = []
    
    if args.skip_relations:
        print("Skipping relation extraction (--skip-relations)")
    else:
        try:
            relation_extractor = RelationExtractor(model_name=args.relation_model)
            
            for q_idx, entities in tqdm(question_entities.items(), desc="Relation extraction"):
                body = questions[q_idx].get("body", "")
                relations = relation_extractor.extract_relations(body, entities)
                all_relations.extend(relations)
            
            print(f"Extracted {len(all_relations)} relations")
        except Exception as e:
            print(f"Warning: Relation extraction failed: {e}")
            print("Continuing without inter-entity relations...")
    print(len(pubmed_records))
    # Build graph
    print("Constructing knowledge graph...")
    graph = build_knowledge_graph(
        questions=questions,
        pubmed_records=pubmed_records,
        question_entities=question_entities,
        entity_relations=all_relations,
        pmid_to_questions=pmid_to_questions,
    )
    
    # Save graph
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    graph.save(args.output)
    print(f"Saved graph to {args.output}")
    
    # Print stats
    gt = _lazy_import_graph_tool()
    vp_type = graph.vertex_properties["type"]
    
    doc_count = sum(1 for v in graph.vertices() if vp_type[v] == "document")
    entity_count = sum(1 for v in graph.vertices() if vp_type[v] == "entity")
    
    print(f"\n=== Graph Statistics ===")
    print(f"Total vertices: {graph.num_vertices()}")
    print(f"  - Document nodes: {doc_count}")
    print(f"  - Entity nodes: {entity_count}")
    print(f"Total edges: {graph.num_edges()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a knowledge graph from BioASQ train.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/Output
    parser.add_argument(
        "--train-file", 
        default="datasets/train/train.json",
        help="Path to BioASQ train.json"
    )
    parser.add_argument(
        "--output", 
        default="knowledge_graph/bioasq_kg.gt",
        help="Output path for graph-tool .gt file"
    )
    
    # Model configuration
    parser.add_argument(
        "--gemma-model", 
        default="gemma2:2b",
        help="Ollama model for entity extraction"
    )
    parser.add_argument(
        "--relation-model",
        default="Babelscape/rebel-large",
        help="HuggingFace model for relation extraction"
    )
    
    # Processing options
    parser.add_argument(
        "--entity-top-k", 
        type=int, 
        default=10,
        help="Max entities to extract per question"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=60,
        help="Timeout for LLM requests (seconds)"
    )
    parser.add_argument(
        "--limit-questions", 
        type=int, 
        default=None,
        help="Limit number of questions to process (for testing)"
    )
    parser.add_argument(
        "--skip-relations",
        action="store_true",
        help="Skip relation extraction (faster, document-entity edges only)"
    )
    
    # Entrez API options
    parser.add_argument(
        "--email",
        default=None,
        help="Email for NCBI Entrez API (or set ENTREZ_EMAIL env var)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NCBI API key for faster rate limits (or set NCBI_API_KEY env var)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
