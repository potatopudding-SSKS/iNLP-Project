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
from pathlib import Path
from typing import Optional
import google.generativeai as genai

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
API_KEY = "AIzaSyAMw8g9ekkGyVT0giql0p7trjnWLbVWGjg"
MODEL_NAME = "gemma-3-27b-it"          # or "gemma-3-12b-it", "gemini-1.5-flash", etc.
REQUEST_DELAY = 1.0                    # seconds between requests (rate-limit friendly)
MAX_RETRIES = 3                        # retries on transient errors
RETRY_DELAY = 5.0                      # seconds to wait before retry

TRAIN_JSON = "datasets/train/train.json"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_answers.json")

# ──────────────────────────────────────────────
#  Prompt templates per question type
# ──────────────────────────────────────────────

PROMPTS = {
    "yesno": (
        "You are a biomedical expert. Answer the following yes/no biomedical question "
        "with a single, complete natural-language sentence. "
        "Begin your sentence with either 'Yes' or 'No', then immediately provide a "
        "brief biomedical justification in the same sentence (maximum 60 words total). "
        "Do NOT produce a bare 'yes' or 'no' — the explanation must be included.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "factoid": (
        "You are a biomedical expert. Answer the following factoid biomedical question "
        "with exactly one complete natural-language sentence (maximum 40 words). "
        "Your sentence must name the specific biomedical entity or entities being asked "
        "about and may include one brief qualifying detail (e.g. function, class, or "
        "context). Do NOT use bullet points, numbering, or lists.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "list": (
        "You are a biomedical expert. Answer the following list biomedical question "
        "with a single fluent natural-language sentence (maximum 80 words) that "
        "enumerates all the relevant biomedical entities. "
        "Write in the style of a textbook answer: present the entities as a "
        "grammatically complete sentence, using commas and 'and' to separate items. "
        "Do NOT use bullet points, numbered lists, or a bare comma-separated list "
        "without an introductory clause.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "summary": (
        "You are a biomedical expert. Write a concise, informative summary answer "
        "to the following biomedical question in 2 to 4 complete natural-language "
        "sentences (maximum 120 words). "
        "Be factual and precise; synthesise the key biomedical facts relevant to "
        "the question. Write in a style suitable for a biomedical review article — "
        "do NOT use headings, bullet points, or lists.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
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

def init_model(api_key: str, model_name: str):
    """Initialise the Gemma/Gemini model via Google GenAI SDK."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model


def call_api(model, prompt: str, retries: int = MAX_RETRIES) -> str:
    """Call the Gemma API with retry logic. Returns the generated text."""
    for attempt in range(1, retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,        # deterministic / greedy
                    max_output_tokens=256,  # cap output length
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"  [Attempt {attempt}/{retries}] API error: {e}")
            if attempt < retries:
                time.sleep(RETRY_DELAY)
    return ""   # return empty string on complete failure


# ──────────────────────────────────────────────
#  Core generation loop
# ──────────────────────────────────────────────

def generate_answers(
    train_path: str,
    output_path: str,
    model,
    resume: bool = True,
    limit: Optional[int] = None,
):
    """
    Iterate over all questions in train.json, generate answers via Gemma,
    and save incrementally to output_path.

    Parameters
    ----------
    train_path   : path to train.json
    output_path  : path for the output JSON file
    model        : initialised GenerativeModel
    resume       : if True, skip questions already present in output_path
    limit        : if set, process at most this many questions (for testing)
    """
    # Load dataset (with repair for known corrupt entries)
    questions = load_bioasq_json(train_path)["questions"]

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

            if qid in done_ids:
                continue  # already processed in a previous run

            # Build prompt
            prompt_template = PROMPTS.get(qtype, PROMPTS["summary"])
            prompt = prompt_template.format(question=body)

            # Generate
            print(f"[{idx:>5}/{total}] type={qtype:<8} id={qid}  Q: {body[:80]}...")
            generated = call_api(model, prompt)
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Model : {args.model}")
    print(f"Input : {args.train}")
    print(f"Output: {args.output}")
    if args.limit:
        print(f"Limit : {args.limit} questions")

    model = init_model(args.api_key, args.model)
    generate_answers(
        train_path=args.train,
        output_path=args.output,
        model=model,
        resume=not args.no_resume,
        limit=args.limit,
    )
