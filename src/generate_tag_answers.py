"""
BioASQ Baseline Answer Generator using Gemma (Google Generative AI API)
-----------------------------------------------------------------------
Generates answers for all four BioASQ question types:
  - yesno   : answers "yes" or "no" (+ a brief justification)
  - factoid : short, precise factual answer
  - list    : a list of entities
  - summary : a paragraph-length synthesis

Output: outputs/baseline_answers.json
Format: [{"id": "<bioasq_id>", "type": "<qtype>", "generated_answer": "..."},  ...]

Note: train.json has a known malformed entry where some ideal_answer arrays are
      missing the opening quote. load_bioasq_json() repairs this automatically.
"""

import json
import re
import os
import time
import argparse
import tempfile
from pathlib import Path
from typing import Optional
from google import genai

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
API_KEY = "AIzaSyBGD0Yxr2sGXukGRi38Zs_XEWYFKScDktU"
MODEL_NAME = "gemma-3-27b-it"          # or "gemma-3-12b-it", "gemini-1.5-flash", etc.
REQUEST_DELAY = 1.0                    # seconds between requests (rate-limit friendly)
MAX_RETRIES = 3                        # retries on transient errors
RETRY_DELAY = 5.0                      # seconds to wait before retry

TRAIN_JSON = "datasets/train/train.json"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tag_final_answers.json")

# ──────────────────────────────────────────────
#  Prompt templates per question type
# ──────────────────────────────────────────────

PROMPTS = {
    "yesno": """You are a biomedical expert. Answer the question using ONLY the provided context.
Do not use outside knowledge. Do not refer to the context as passages, excerpts, documents, or studies.
Do not write phrases such as "Passage 1", "the passage", "the context states", or "according to the text".
Write the answer directly and fluently.

Context:
{context}

Question: {question}

Instructions:
- Answer with "Yes" or "No" followed by a brief justification.
- Use only the provided context.
- If the answer is not explicitly stated, infer the best possible answer from the context.
- Use multiple passages if needed.
- Do not mention passages, context blocks, or source formatting in the answer.
- The answer must be fluent, complete, and non-empty.

Answer:""",

    "factoid": """You are a biomedical expert. Answer the question using ONLY the provided context.
Do not use outside knowledge. Do not refer to the context as passages, excerpts, documents, or studies.
Do not write phrases such as "Passage 1", "the passage", "the context states", or "according to the text".
Write the answer directly and fluently.

Context:
{context}

Question: {question}

Instructions:
- Provide the answer in a single complete sentence.
- Name the specific entity, value, or concept being asked for.
- Use only the provided context.
- If the answer is not explicitly stated, infer the best possible answer from the context.
- Use multiple passages if needed.
- Do not mention passages, context blocks, or source formatting in the answer.
- The answer must be fluent, specific, and non-empty.

Answer:""",

    "list": """You are a biomedical expert. Answer the question using ONLY the provided context.
Do not use outside knowledge. Do not refer to the context as passages, excerpts, documents, or studies.
Do not write phrases such as "Passage 1", "the passage", "the context states", or "according to the text".
Write the answer directly and fluently.

Context:
{context}

Question: {question}

Instructions:
- Provide all relevant items in a single fluent sentence using natural enumeration.
- Use only the provided context.
- If the answer is not explicitly stated, infer the best possible answer from the context.
- Use multiple passages if needed.
- Do not mention passages, context blocks, or source formatting in the answer.
- The answer must be fluent, complete, and non-empty.

Answer:""",

    "summary": """You are a biomedical expert. Answer the question using ONLY the provided context.
Do not use outside knowledge. Do not refer to the context as passages, excerpts, documents, or studies.
Do not write phrases such as "Passage 1", "the passage", "the context states", or "according to the text".
Write the answer directly and fluently.

Context:
{context}

Question: {question}

Instructions:
- Write a concise, informative answer in 2-4 complete sentences.
- Use only the provided context.
- If the answer is not explicitly stated, infer the best possible answer from the context.
- Use multiple passages if needed.
- Do not mention passages, context blocks, or source formatting in the answer.
- The answer must be fluent, coherent, and non-empty.

Answer:"""
}


# ──────────────────────────────────────────────
#  JSON loading (with repair for corrupt train.json)
# ──────────────────────────────────────────────

def load_bioasq_json(path: str) -> dict:
    """
    Load a BioASQ JSON file, repairing a known corruption where some
    `ideal_answer` array values are missing their opening double-quote.

    Example corrupt form:    "ideal_answer": [thapi/metadata/DOID:11372"
    Repaired form:           "ideal_answer": ["thapi/metadata/DOID:11372"
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    # Fix: `ideal_answer`: [<non-quote, non-bracket char> → add missing opening "
    fixed = re.sub(
        r'("ideal_answer":\s*\[)([^"\[\]\n])',
        r'\1"\2',
        raw,
    )
    return json.loads(fixed)


# ──────────────────────────────────────────────
#  API helpers
# ──────────────────────────────────────────────

def init_client(api_key: str):
    """Initialise the Gemma/Gemini model via Google GenAI SDK."""
    client = genai.Client(api_key=api_key)
    return client


def call_api(client, model_name, prompt: str, retries: int = MAX_RETRIES) -> str:
    """Call the Gemma API with retry logic. Returns the generated text."""
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model = model_name,
                contents = [
                    prompt
                ]
            )
            return response.text.strip()
        except Exception as e:
            print(f"  [Attempt {attempt}/{retries}] API error: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
    return ""   # return empty string on complete failure


def upload_json_context(client, structure: dict):
    """Upload JSON context as a temporary file and return uploaded file metadata."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(structure, tmp, ensure_ascii=False, indent=2)
            temp_path = tmp.name
        return client.files.upload(file=temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ──────────────────────────────────────────────
#  Core generation loop
# ──────────────────────────────────────────────

def generate_answers(
    train_path: str,
    output_path: str,
    client,
    model_name: str,
    resume: bool = True,
    limit: Optional[int] = None,
    tag_evaluation: bool = False,
):
    """
    Iterate over all questions in train.json, generate answers via Gemma,
    and save incrementally to output_path.

    Parameters
    ----------
    train_path   : path to train.json
    output_path  : path for the output JSON file
    client       : initialised GenAI client
    resume       : if True, skip questions already present in output_path
    limit        : if set, process at most this many questions (for testing)
    """
    # Load dataset (with repair for known corrupt entries)
    questions = load_bioasq_json(train_path)["questions"]

    graph = None
    if tag_evaluation:
        from kg2 import load_graph
        graph = load_graph("knowledge_graph/bioasq_kg.gt")

    if limit:
        questions = questions[:limit]

    total = len(questions)
    print(f"Loaded {total} questions from {train_path}")

    # Load existing results if resuming
    results: list[dict] = []
    done_ids: set[str] = set()
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} questions already done, skipping them.")

    # Count by type for progress reporting
    type_counts = {"yesno": 0, "factoid": 0, "list": 0, "summary": 0}

    try:
        for idx, q in enumerate(questions, 1):
            qid = q["id"]
            qtype = q.get("type", "summary")
            body = q["body"]
            attached_file = None

            if qid in done_ids:
                continue  # already processed in a previous run

            # Build prompt
            prompt_template = PROMPTS.get(qtype, PROMPTS["summary"])

            if tag_evaluation:
                try:
                    from graph_traversal import process_question
                    structure = process_question(body, graph)
                    attached_file = upload_json_context(client, structure)
                except Exception as e:
                    print(f"  [TAG context] Failed to build JSON structure: {e}")

            prompt = prompt_template.format(
                context=f"[Attached JSON file: {attached_file}]" if attached_file else "No additional context.",
                question=body
            )
            # Generate
            print(f"[{idx:>5}/{total}] type={qtype:<8} id={qid}  Q: {body[:80]}...")
            generated = call_api(client, model_name, prompt)
            if attached_file is not None:
                try:
                    client.files.delete(name=attached_file.name)
                except Exception:
                    # Best-effort remote cleanup.
                    pass
            time.sleep(REQUEST_DELAY)

            # Store
            results.append({
                "id": qid,
                "type": qtype,
                "generated_answer": generated,
            })
            done_ids.add(qid)
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

            # Save incrementally every 50 questions
            if len(results) % 50 == 0:
                _save(results, output_path)
                print(f"  → Checkpoint saved ({len(results)} done so far)")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")

    # Final save
    _save(results, output_path)
    print(f"\nDone! {len(results)} answers saved to {output_path}")
    print("Breakdown:", {k: v for k, v in type_counts.items() if v > 0})
    return results


def _save(results: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate BioASQ answers using Gemma API"
    )
    parser.add_argument(
        "--train", default=TRAIN_JSON,
        help=f"Path to train.json (default: {TRAIN_JSON})"
    )
    parser.add_argument(
        "--output", default=OUTPUT_FILE,
        help=f"Output JSON path (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N questions (useful for testing)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Restart from scratch even if output file exists"
    )
    parser.add_argument(
        "--api-key", default=API_KEY,
        help="Google Generative AI API key"
    )
    parser.add_argument(
        "--tag-evaluation", action="store_true",
        help="Include create_json graph structure in the LLM prompt"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Model : {args.model}")
    print(f"Input : {args.train}")
    print(f"Output: {args.output}")
    if args.limit:
        print(f"Limit : {args.limit} questions")

    client = init_client(args.api_key)
    generate_answers(
        train_path=args.train,
        output_path=args.output,
        client=client,
        model_name=args.model,
        resume=not args.no_resume,
        limit=args.limit,
        tag_evaluation=args.tag_evaluation,
    )
