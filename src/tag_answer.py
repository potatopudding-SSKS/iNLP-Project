"""
BioASQ TAG-Guided Answer Generator
------------------------------------
A Text-Attributed Graph (TAG) reasoning pipeline for biomedical QA.

Pipeline:
  1. Load the co-citation TAG (kg.dat) built by kg.py
     - Nodes  = PubMed papers (with title + abstract)
     - Edges  = co-citation links (papers cited for the same BioASQ question)
  2. For each BioASQ question:
       a. Identify seed papers (listed in question["documents"])
       b. Perform multi-hop BFS traversal over the TAG
       c. Rank the reachable subgraph by node connectivity
       d. Serialize top-N paper abstracts as a structured evidence block
       e. Prompt Gemma with: evidence + question → graph-grounded answer
  3. Save to outputs/tag_answers.json

Format: [{"id": "<bioasq_id>", "type": "<qtype>", "generated_answer": "..."}, ...]
"""

import json
import re
import os
import time
import pickle
import argparse
from collections import deque
from pathlib import Path
from typing import Optional
import google.generativeai as genai

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
API_KEY    = "AIzaSyAMw8g9ekkGyVT0giql0p7trjnWLbVWGjg"
MODEL_NAME = "gemma-3-27b-it"
REQUEST_DELAY = 1.0
MAX_RETRIES   = 3
RETRY_DELAY   = 5.0

KG_PATH    = "knowledge_graph/kg.dat"
TRAIN_JSON = "datasets/train/train.json"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tag_answers.json")

# TAG traversal settings
HOP_DEPTH   = 2    # how many hops to expand from seed papers
MAX_CONTEXT = 5    # max papers to include in the context block
MAX_ABSTRACT_TOKENS = 200   # truncate each abstract to ~200 words

# ──────────────────────────────────────────────
#  Prompt templates
# ──────────────────────────────────────────────
SYSTEM_CONTEXT = (
    "You are a biomedical expert. The following text passages are excerpts from "
    "PubMed research articles retrieved via TAG-guided graph traversal. "
    "Use them as your primary source of evidence to construct your answer. "
    "Do not invent facts not supported by the provided passages.\n\n"
    "Graph-retrieved evidence:\n"
    "{context}\n\n"
    "---\n"
)

PROMPTS = {
    "yesno": (
        SYSTEM_CONTEXT +
        "Answer the following yes/no biomedical question with a single, complete "
        "natural-language sentence. Begin with 'Yes' or 'No', then provide a brief "
        "biomedical justification drawn from the context above (maximum 60 words).\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "factoid": (
        SYSTEM_CONTEXT +
        "Answer the following factoid biomedical question with exactly one complete "
        "natural-language sentence (maximum 40 words). Name the specific entity or "
        "entities asked for, grounded in the context above. Do NOT use bullet points.\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "list": (
        SYSTEM_CONTEXT +
        "Answer the following list biomedical question with a single fluent "
        "natural-language sentence (maximum 80 words) that enumerates all relevant "
        "entities found in the context above. Use commas and 'and' to separate items. "
        "Do NOT use bullet points or numbered lists.\n\n"
        "Question: {question}\n\nAnswer:"
    ),
    "summary": (
        SYSTEM_CONTEXT +
        "Write a concise summary answer to the following biomedical question in "
        "2 to 4 complete natural-language sentences (maximum 120 words). "
        "Be factual; synthesise information from the context above. "
        "Do NOT use headings, bullet points, or lists.\n\n"
        "Question: {question}\n\nAnswer:"
    ),
}

# ──────────────────────────────────────────────
#  Load KG
# ──────────────────────────────────────────────

def load_kg(kg_path: str) -> tuple[dict, dict]:
    """Load adjacency list and tag dict from kg.dat."""
    with open(kg_path, "rb") as f:
        adj, tag = pickle.load(f)
    # Ensure adjacency values are sets (they should be, but just in case)
    adj = {k: set(v) for k, v in adj.items()}
    return adj, tag


# ──────────────────────────────────────────────
#  Subgraph retrieval
# ──────────────────────────────────────────────

def get_subgraph_papers(
    seed_pmids: list[int],
    adj: dict[int, set],
    hop_depth: int = HOP_DEPTH,
    max_papers: int = MAX_CONTEXT,
) -> list[int]:
    """
    TAG traversal: BFS from seed papers up to `hop_depth` hops.
    Ranks reachable nodes by connectivity within the subgraph and
    returns the top `max_papers` PMIDs (seed papers always included first).
    """
    visited: set[int] = set()
    queue = deque()
    degree_in_subgraph: dict[int, int] = {}

    # Initialise BFS with seeds
    for pmid in seed_pmids:
        if pmid in adj:
            queue.append((pmid, 0))
            visited.add(pmid)

    while queue:
        node, depth = queue.popleft()
        for neighbor in adj.get(node, set()):
            degree_in_subgraph[neighbor] = degree_in_subgraph.get(neighbor, 0) + 1
            if neighbor not in visited and depth < hop_depth:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    # Always include seeds; then add top-ranked neighbors by connectivity
    result = [p for p in seed_pmids if p in adj or p in visited]
    extras = sorted(
        [p for p in visited if p not in result],
        key=lambda p: degree_in_subgraph.get(p, 0),
        reverse=True,
    )
    combined = result + extras
    return combined[:max_papers]


# ──────────────────────────────────────────────
#  Context serialization
# ──────────────────────────────────────────────

def truncate(text: str, max_words: int = MAX_ABSTRACT_TOKENS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def build_context(pmids: list[int], tag: dict) -> str:
    """Serialize selected papers into a numbered text block."""
    chunks = []
    for i, pmid in enumerate(pmids, 1):
        entry = tag.get(pmid, {})
        title    = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        if not abstract:
            continue
        chunk = f"[{i}] {title}\n{truncate(abstract)}"
        chunks.append(chunk)
    return "\n\n".join(chunks) if chunks else "No relevant context found."


# ──────────────────────────────────────────────
#  JSON loader (with repair for corrupt train.json)
# ──────────────────────────────────────────────

def load_bioasq_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    fixed = re.sub(
        r'("ideal_answer":\s*\[)([^"\[\]\n])',
        r'\1"\2',
        raw,
    )
    return json.loads(fixed)


# ──────────────────────────────────────────────
#  API helpers
# ──────────────────────────────────────────────

def init_model(api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def call_api(model, prompt: str, retries: int = MAX_RETRIES) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"  [Attempt {attempt}/{retries}] API error: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
    return ""


# ──────────────────────────────────────────────
#  Core generation loop
# ──────────────────────────────────────────────

def generate_tag_answers(
    train_path: str,
    kg_path: str,
    output_path: str,
    model,
    resume: bool = True,
    limit: Optional[int] = None,
):
    print("Loading knowledge graph...")
    adj, tag = load_kg(kg_path)
    print(f"  KG loaded: {len(adj)} nodes, {sum(len(v) for v in adj.values())//2} edges")

    questions = load_bioasq_json(train_path)["questions"]
    if limit:
        questions = questions[:limit]
    total = len(questions)
    print(f"Loaded {total} questions from {train_path}")

    # Resume logic
    results: list[dict] = []
    done_ids: set[str] = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} questions already done.")

    try:
        for idx, q in enumerate(questions, 1):
            qid   = q["id"]
            qtype = q.get("type", "summary")
            body  = q["body"]

            if qid in done_ids:
                continue

            # --- Step 1: extract seed PMIDs from question's document list ---
            seed_pmids = [
                int(url.rstrip("/").split("/")[-1])
                for url in q.get("documents", [])
            ]

            # --- Step 2: TAG multi-hop traversal ---
            selected_pmids = get_subgraph_papers(seed_pmids, adj)

            # --- Step 3: serialize subgraph evidence ---
            context = build_context(selected_pmids, tag)

            # --- Step 4: build prompt and call LLM ---
            template = PROMPTS.get(qtype, PROMPTS["summary"])
            prompt   = template.format(context=context, question=body)

            print(
                f"[{idx:>5}/{total}] type={qtype:<8} id={qid} "
                f"ctx_papers={len(selected_pmids)}  Q: {body[:60]}..."
            )
            generated = call_api(model, prompt)
            time.sleep(REQUEST_DELAY)

            results.append({
                "id": qid,
                "type": qtype,
                "generated_answer": generated,
                "context_pmids": selected_pmids,   # for attribution / ablation
            })
            done_ids.add(qid)

            # Checkpoint every 50
            if len(results) % 50 == 0:
                _save(results, output_path)
                print(f"  → Checkpoint: {len(results)} done")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving progress...")

    _save(results, output_path)
    print(f"\nDone! {len(results)} TAG-guided answers saved to {output_path}")
    return results


def _save(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="BioASQ TAG-guided answer generator")
    parser.add_argument("--train",  default=TRAIN_JSON)
    parser.add_argument("--kg",     default=KG_PATH)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--model",  default=MODEL_NAME)
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--hops",  type=int, default=HOP_DEPTH,
                        help="BFS hop depth for subgraph expansion")
    parser.add_argument("--top-k", type=int, default=MAX_CONTEXT,
                        help="Max papers to include in context")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Model  : {args.model}")
    print(f"Input  : {args.train}")
    print(f"KG     : {args.kg}")
    print(f"Output : {args.output}")
    print(f"Hops   : {args.hops}  |  Top-K papers: {args.top_k}")
    if args.limit:
        print(f"Limit  : {args.limit} questions")

    model = init_model(args.api_key, args.model)
    generate_tag_answers(
        train_path=args.train,
        kg_path=args.kg,
        output_path=args.output,
        model=model,
        resume=not args.no_resume,
        limit=args.limit,
    )
