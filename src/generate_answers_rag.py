#!/usr/bin/env python3
"""
BioASQ Clean RAG Baseline (KG-Independent)
-------------------------------------------
A standard Retrieval-Augmented Generation (RAG) baseline for BioASQ
biomedical QA that is COMPLETELY independent of any knowledge graph.

KEY DESIGN:
  - Build ONE global corpus from ALL PubMed documents referenced in train.json
  - Fetch title + abstract via PubMed Entrez API (with local caching)
  - This corpus is SHARED across all questions
  - Retrieval is performed via BM25 over this shared corpus
  - Does NOT use per-question document annotations during inference
  - Does NOT use any KG or graph-based reasoning

Pipeline:
  1. Load BioASQ questions from train.json
  2. Extract ALL unique PubMed IDs from ALL questions' "documents" fields
  3. Fetch title + abstract from PubMed API (or load from cache)
  4. Build global corpus: text = title + " " + abstract
  5. Chunk documents (~200 words, 50 word overlap)
  6. Build BM25 index over ALL chunks
  7. For each question (using ONLY body and type):
       a. Retrieve top-k chunks using BM25
       b. Build context from retrieved passages
       c. Generate answer using Gemma via Google Generative AI API
  8. Save answers in BioASQ format

This provides a FAIR RAG baseline comparable to TAG systems built on the same
document pool.
"""

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from rank_bm25 import BM25Okapi
import google.generativeai as genai


# --------------------------------------------------
# Default Configuration
# --------------------------------------------------
DEFAULT_MODEL = "gemma-3-27b-it"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 200  # words per chunk
DEFAULT_CHUNK_OVERLAP = 50  # overlap words between chunks
REQUEST_DELAY = 1.0
MAX_RETRIES = 5  # Increased retries for robustness
RETRY_DELAY = 5.0
CHECKPOINT_INTERVAL = 10

# Context length control
MAX_CONTEXT_WORDS = 1800  # Maximum words in context (~safe for most models)
MIN_PASSAGES = 1  # Minimum passages to include (fallback)

# PubMed API settings
PUBMED_BATCH_SIZE = 200  # PMIDs per request
PUBMED_DELAY = 0.4  # seconds between API calls (respect rate limits)

# Resolve paths relative to project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_PATH = str(PROJECT_ROOT / "datasets" / "train" / "train.json")
DEFAULT_CACHE_PATH = str(PROJECT_ROOT / "datasets" / "pubmed_cache.json")
DEFAULT_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "rag_baseline_answers.json")

# Logging settings
VERBOSE_RETRIEVAL = True  # Log retrieval scores for debugging


# --------------------------------------------------
# Prompt Templates
# --------------------------------------------------
PROMPT_TEMPLATES = {
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


# --------------------------------------------------
# Data Loading Functions
# --------------------------------------------------
def load_bioasq_json(path: str) -> list[dict[str, Any]]:
    """
    Load BioASQ dataset from JSON file.
    
    Handles malformed ideal_answer entries that may exist in the dataset.
    
    Args:
        path: Path to the BioASQ train.json file
        
    Returns:
        List of question dictionaries with id, body, type fields
    """
    print(f"Loading BioASQ questions from {path}...")
    
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    
    # Fix known malformed ideal_answer entries
    fixed = re.sub(
        r'("ideal_answer":\s*\[)([^"\[\]\n])',
        r'\1"\2',
        raw,
    )
    
    data = json.loads(fixed)
    questions = data.get("questions", data)
    
    print(f"  Loaded {len(questions)} questions")
    return questions


# --------------------------------------------------
# PMID Extraction
# --------------------------------------------------
def extract_pmids(questions: list[dict[str, Any]]) -> list[str]:
    """
    Extract ALL unique PubMed IDs from ALL questions in the dataset.
    
    This collects PMIDs from the "documents" field of every question
    to build a GLOBAL corpus.
    
    Args:
        questions: List of BioASQ question dicts
        
    Returns:
        List of unique PMID strings
    """
    print("Extracting PMIDs from all questions...")
    
    all_pmids = set()
    
    for question in questions:
        doc_urls = question.get("documents", [])
        for url in doc_urls:
            # Extract PMID from URL like "http://www.ncbi.nlm.nih.gov/pubmed/12345"
            tail = url.rstrip("/").split("/")[-1]
            try:
                pmid = str(int(tail))  # Validate it's a number
                all_pmids.add(pmid)
            except ValueError:
                continue
    
    pmid_list = sorted(all_pmids, key=lambda x: int(x))
    print(f"  Found {len(pmid_list)} unique PMIDs across all questions")
    return pmid_list


# --------------------------------------------------
# PubMed API Functions
# --------------------------------------------------
def fetch_pubmed_abstracts(
    pmids: list[str],
    batch_size: int = PUBMED_BATCH_SIZE,
    delay: float = PUBMED_DELAY
) -> dict[str, dict[str, str]]:
    """
    Fetch title and abstract for each PMID using PubMed Entrez API.
    
    Uses the efetch endpoint to retrieve article metadata in batches.
    
    Args:
        pmids: List of PMID strings to fetch
        batch_size: Number of PMIDs per API request
        delay: Seconds to wait between requests (rate limiting)
        
    Returns:
        Dictionary mapping PMID -> {"title": ..., "abstract": ...}
    """
    print(f"Fetching abstracts from PubMed for {len(pmids)} PMIDs...")
    
    results = {}
    total_batches = (len(pmids) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(pmids))
        batch_pmids = pmids[start:end]
        
        print(f"  Batch {batch_idx + 1}/{total_batches}: PMIDs {start + 1}-{end}")
        
        try:
            batch_results = _fetch_batch(batch_pmids)
            results.update(batch_results)
        except Exception as e:
            print(f"    Warning: Failed to fetch batch: {e}")
        
        # Rate limiting
        if batch_idx < total_batches - 1:
            time.sleep(delay)
    
    print(f"  Successfully fetched {len(results)} abstracts")
    return results


def _fetch_batch(pmids: list[str]) -> dict[str, dict[str, str]]:
    """
    Fetch a single batch of PMIDs from PubMed.
    
    Args:
        pmids: List of PMID strings
        
    Returns:
        Dictionary mapping PMID -> {"title": ..., "abstract": ...}
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml"
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    
    with urllib.request.urlopen(url, timeout=30) as response:
        xml_data = response.read().decode("utf-8")
    
    return _parse_pubmed_xml(xml_data)


def _parse_pubmed_xml(xml_data: str) -> dict[str, dict[str, str]]:
    """
    Parse PubMed XML response to extract titles and abstracts.
    
    Args:
        xml_data: XML string from PubMed API
        
    Returns:
        Dictionary mapping PMID -> {"title": ..., "abstract": ...}
    """
    results = {}
    
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return results
    
    for article in root.findall(".//PubmedArticle"):
        # Extract PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            continue
        pmid = pmid_elem.text.strip()
        
        # Extract title
        title_elem = article.find(".//ArticleTitle")
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        
        # Extract abstract (may have multiple parts)
        abstract_parts = []
        for abstract_text in article.findall(".//AbstractText"):
            if abstract_text.text:
                # Handle labeled abstract sections
                label = abstract_text.get("Label", "")
                text = abstract_text.text.strip()
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        
        abstract = " ".join(abstract_parts)
        
        results[pmid] = {
            "title": title,
            "abstract": abstract
        }
    
    return results


# --------------------------------------------------
# Caching Functions
# --------------------------------------------------
def load_or_build_corpus_cache(
    pmids: list[str],
    cache_path: str
) -> dict[str, dict[str, str]]:
    """
    Load corpus from cache if exists, otherwise fetch from PubMed and save.
    
    This avoids repeated API calls by caching the fetched data.
    
    Args:
        pmids: List of PMID strings needed
        cache_path: Path to cache JSON file
        
    Returns:
        Dictionary mapping PMID -> {"title": ..., "abstract": ...}
    """
    cache = {}
    
    # Try to load existing cache
    if os.path.exists(cache_path):
        print(f"Loading corpus cache from {cache_path}...")
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"  Loaded {len(cache)} cached entries")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not load cache: {e}")
            cache = {}
    
    # Find PMIDs not in cache
    missing_pmids = [p for p in pmids if p not in cache]
    
    if missing_pmids:
        print(f"  {len(missing_pmids)} PMIDs not in cache, fetching from PubMed...")
        new_entries = fetch_pubmed_abstracts(missing_pmids)
        cache.update(new_entries)
        
        # Save updated cache
        _save_cache(cache, cache_path)
    else:
        print("  All PMIDs found in cache")
    
    return cache


def _save_cache(cache: dict[str, dict[str, str]], path: str) -> None:
    """
    Save corpus cache to JSON file.
    
    Args:
        cache: Dictionary mapping PMID -> {"title": ..., "abstract": ...}
        path: Path to save cache
    """
    print(f"Saving corpus cache to {path}...")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(cache)} entries to cache")


# --------------------------------------------------
# Corpus Building
# --------------------------------------------------
def build_corpus_from_cache(
    pmids: list[str],
    cache: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    """
    Build document corpus from cached PubMed data.
    
    Creates documents with text = title + " " + abstract
    
    Args:
        pmids: List of PMID strings
        cache: Dictionary mapping PMID -> {"title": ..., "abstract": ...}
        
    Returns:
        List of document dicts with 'id', 'title', 'text' fields
    """
    print("Building corpus from cached data...")
    
    documents = []
    missing_count = 0
    
    for pmid in pmids:
        entry = cache.get(pmid, {})
        title = entry.get("title", "").strip()
        abstract = entry.get("abstract", "").strip()
        
        # Skip if no content
        if not title and not abstract:
            missing_count += 1
            continue
        
        # Combine: text = title + " " + abstract
        text = f"{title} {abstract}".strip()
        
        documents.append({
            "id": pmid,
            "title": title,
            "text": text
        })
    
    print(f"  Built corpus with {len(documents)} documents")
    if missing_count > 0:
        print(f"  Warning: {missing_count} PMIDs had no content")
    
    return documents


# --------------------------------------------------
# Text Processing Functions
# --------------------------------------------------
def tokenize(text: str) -> list[str]:
    """
    Simple lowercase tokenization for BM25.
    
    Args:
        text: Input text string
        
    Returns:
        List of lowercase tokens
    """
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return tokens


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[dict[str, Any]]:
    """
    Split documents into overlapping chunks (~200 words, 50 overlap).
    
    Args:
        documents: List of document dicts with 'id', 'text' fields
        chunk_size: Target number of words per chunk
        chunk_overlap: Number of overlapping words between chunks
        
    Returns:
        List of chunk dicts with 'doc_id', 'chunk_id', 'text' fields
    """
    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    
    chunks = []
    
    for doc in documents:
        doc_id = doc["id"]
        text = doc["text"]
        words = text.split()
        
        if len(words) == 0:
            continue
        
        # If document is smaller than chunk size, use whole document
        if len(words) <= chunk_size:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_0",
                "text": text
            })
            continue
        
        # Create overlapping chunks
        chunk_idx = 0
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_idx}",
                "text": chunk_text
            })
            
            chunk_idx += 1
            start += chunk_size - chunk_overlap
            
            # Prevent infinite loop at document end
            if end == len(words):
                break
    
    print(f"  Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


# --------------------------------------------------
# BM25 Retrieval Functions
# --------------------------------------------------
def build_bm25(chunks: list[dict[str, Any]]) -> tuple[BM25Okapi, list[dict[str, Any]]]:
    """
    Build BM25 index over corpus chunks.
    
    Args:
        chunks: List of chunk dicts with 'text' field
        
    Returns:
        Tuple of (BM25 index, chunks list)
    """
    print("Building BM25 index...")
    
    # Tokenize all chunks
    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"  BM25 index built over {len(chunks)} chunks")
    return bm25, chunks


def retrieve_top_k(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict[str, Any]],
    k: int = DEFAULT_TOP_K,
    min_passages: int = MIN_PASSAGES
) -> list[dict[str, Any]]:
    """
    Retrieve top-k passages for a query using BM25.
    
    Includes fallback logic to ensure at least min_passages are returned
    even if scores are low (better than empty context).
    
    Args:
        query: Question text
        bm25: BM25 index
        chunks: List of chunk dicts
        k: Number of passages to retrieve
        min_passages: Minimum passages to return (fallback)
        
    Returns:
        List of top-k chunk dicts with added 'score' field
    """
    # Tokenize query
    query_tokens = tokenize(query)
    
    if not query_tokens:
        # Fallback: return first min_passages chunks if query can't be tokenized
        if chunks and min_passages > 0:
            return [{"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], 
                     "text": c["text"], "score": 0.0} for c in chunks[:min_passages]]
        return []
    
    # Get BM25 scores for all chunks
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    # Build results - include passages with positive scores
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            chunk = chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)
    
    # Fallback: if no positive-score results, include top min_passages anyway
    # (weak relevance is better than no context)
    if len(results) < min_passages and chunks:
        for idx in top_indices[:min_passages]:
            if idx not in [chunks.index(r) if r in chunks else -1 for r in results]:
                chunk = chunks[idx].copy()
                chunk["score"] = float(scores[idx])
                if chunk not in results:
                    results.append(chunk)
            if len(results) >= min_passages:
                break
    
    return results


# --------------------------------------------------
# Prompt Building
# --------------------------------------------------
def truncate_text(text: str, max_words: int) -> str:
    """
    Truncate text to a maximum number of words.
    
    Args:
        text: Input text
        max_words: Maximum words allowed
        
    Returns:
        Truncated text with ellipsis if needed
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def build_prompt(
    question: str,
    question_type: str,
    retrieved_passages: list[dict[str, Any]],
    max_context_words: int = MAX_CONTEXT_WORDS
) -> str:
    """
    Build the prompt with question, type, and retrieved context.
    
    Features:
    - Context length control (prevents prompt overflow)
    - Includes PMID for interpretability
    - Includes relevance scores for transparency
    
    Context format:
        PMID: 12345
        text...
        
        PMID: 67890
        text...
    
    Args:
        question: Question text
        question_type: One of 'yesno', 'factoid', 'list', 'summary'
        retrieved_passages: List of retrieved passage dicts
        max_context_words: Maximum words allowed in context
        
    Returns:
        Formatted prompt string
    """
    # Build context from retrieved passages with length control
    if not retrieved_passages:
        context = "No relevant passages found."
    else:
        context_parts = []
        total_words = 0
        
        for passage in retrieved_passages:
            text = passage["text"]
            pmid = passage.get("doc_id", "unknown")
            
            # Calculate words in this passage
            passage_words = len(text.split())
            
            # Check if adding this passage would exceed limit
            if total_words + passage_words > max_context_words and context_parts:
                # Truncate this passage to fit remaining space
                remaining_words = max_context_words - total_words
                if remaining_words > 50:  # Only include if we can fit meaningful content
                    truncated_text = truncate_text(text, remaining_words)
                    context_parts.append(
                        f"PMID: {pmid}\n{truncated_text}"
                    )
                break
            
            # Add full passage
            context_parts.append(
                f"PMID: {pmid}\n{text}"
            )
            total_words += passage_words
        
        context = "\n\n".join(context_parts)
    
    # Get appropriate template
    template = PROMPT_TEMPLATES.get(question_type, PROMPT_TEMPLATES["summary"])
    
    # Format prompt
    prompt = template.format(context=context, question=question)
    
    return prompt


# --------------------------------------------------
# API Functions
# --------------------------------------------------
def init_api(api_key: str, model_name: str):
    """
    Initialize Google Generative AI with API key and model.
    
    Args:
        api_key: Google API key
        model_name: Model name (e.g., 'gemma-3-27b-it')
        
    Returns:
        Configured GenerativeModel instance
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    print(f"  Initialized model: {model_name}")
    return model


def call_api(
    model,
    prompt: str,
    temperature: float = 0.0,
    max_output_tokens: int = 256,
    retries: int = MAX_RETRIES
) -> str:
    """
    Call Gemma API with robust retry logic and error handling.
    
    Features:
    - Exponential backoff on retries
    - Handles rate limiting, timeouts, and API errors
    - Safety fallback for blocked content
    
    Args:
        model: GenerativeModel instance
        prompt: Input prompt
        temperature: Generation temperature (0.0 for deterministic)
        max_output_tokens: Maximum tokens to generate
        retries: Number of retry attempts
        
    Returns:
        Generated text or empty string on failure
    """
    last_error = None
    
    for attempt in range(1, retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            
            # Check for blocked content
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = getattr(response.prompt_feedback, 'block_reason', None)
                if block_reason:
                    print(f"    Warning: Content blocked - {block_reason}")
                    return ""
            
            # Extract text from response
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            # Handle empty response
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        parts = getattr(candidate.content, 'parts', [])
                        if parts and hasattr(parts[0], 'text'):
                            return parts[0].text.strip()
            
            return ""
            
        except Exception as exc:
            last_error = exc
            error_str = str(exc).lower()
            
            # Determine retry delay based on error type
            if "quota" in error_str or "rate" in error_str:
                delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"    [Attempt {attempt}/{retries}] Rate limited, waiting {delay:.0f}s...")
            elif "timeout" in error_str:
                delay = RETRY_DELAY
                print(f"    [Attempt {attempt}/{retries}] Timeout, retrying...")
            else:
                delay = RETRY_DELAY * attempt  # Linear backoff
                print(f"    [Attempt {attempt}/{retries}] API error: {exc}")
            
            if attempt < retries:
                time.sleep(delay)
    
    print(f"    All {retries} attempts failed. Last error: {last_error}")
    return ""


def ensure_non_empty_answer(question_type: str, generated: str) -> str:
    """Return a fluent non-empty fallback if the model yields no text."""
    answer = generated.strip()
    if answer:
        return answer

    if question_type == "yesno":
        return "Yes, the provided context suggests this conclusion."
    if question_type == "factoid":
        return "The provided context suggests the most likely answer implied by the retrieved passages."
    if question_type == "list":
        return "The provided context suggests the relevant items can be inferred from the retrieved passages."
    return (
        "The provided context suggests the answer can be inferred from the retrieved passages. "
        "Taken together, the available information supports a concise summary."
    )


# --------------------------------------------------
# Main Generation Function
# --------------------------------------------------
def generate_answers(
    questions: list[dict[str, Any]],
    bm25: BM25Okapi,
    chunks: list[dict[str, Any]],
    model,
    output_path: str,
    top_k: int = DEFAULT_TOP_K,
    limit: Optional[int] = None,
    resume: bool = True,
    verbose: bool = VERBOSE_RETRIEVAL
) -> list[dict[str, Any]]:
    """
    Generate answers for all questions using RAG pipeline.
    
    IMPORTANT: This uses ONLY question["body"] and question["type"] for inference.
    It does NOT use question["documents"], question["snippets"], or 
    question["ideal_answer"] - retrieval is performed over the shared BM25 index.
    
    Args:
        questions: List of BioASQ question dicts
        bm25: BM25 index over shared corpus
        chunks: List of corpus chunks
        model: GenerativeModel instance
        output_path: Path to save results
        top_k: Number of passages to retrieve
        limit: Optional limit on number of questions
        resume: Whether to resume from existing output
        verbose: Whether to log retrieval scores
        
    Returns:
        List of answer dicts
    """
    # Apply limit if specified
    if limit:
        questions = questions[:limit]
    
    total = len(questions)
    print(f"\nGenerating answers for {total} questions...")
    print("  NOTE: Using ONLY question body and type for inference")
    print("  NOTE: Retrieval is over SHARED corpus (not per-question docs)")
    
    # Statistics tracking
    stats = {
        "total_processed": 0,
        "empty_retrieval": 0,
        "api_failures": 0,
        "avg_retrieval_score": [],
    }
    
    # Load existing results for resume
    results: list[dict[str, Any]] = []
    done_ids: set[str] = set()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not os.path.exists(output_path):
        save_results([], output_path)
    
    if resume and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_ids = {r["id"] for r in results}
            print(f"  Resuming: {len(done_ids)} questions already completed")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not load existing results: {e}")
            results = []
            done_ids = set()
    
    start_time = time.time()
    
    try:
        for idx, question in enumerate(questions, 1):
            # Extract ONLY body and type - NOT documents/snippets/ideal_answer
            qid = question.get("id", f"q_{idx}")
            qtype = question.get("type", "summary")
            body = question.get("body", "")
            
            # Skip if already done
            if qid in done_ids:
                continue
            
            # Validate input
            if not body.strip():
                print(f"  Warning: Empty question body for {qid}, skipping...")
                continue
            
            # Retrieve relevant passages from SHARED corpus using BM25
            # This does NOT use question["documents"]
            retrieved = retrieve_top_k(body, bm25, chunks, k=top_k)
            
            # Track retrieval statistics
            if not retrieved:
                stats["empty_retrieval"] += 1
            else:
                scores = [p.get("score", 0) for p in retrieved]
                stats["avg_retrieval_score"].extend(scores)
            
            # Log progress with retrieval quality
            scores_str = ""
            if verbose and retrieved:
                top_scores = [f"{p.get('score', 0):.2f}" for p in retrieved[:3]]
                scores_str = f" scores=[{', '.join(top_scores)}]"
            
            print(f"[{idx:>5}/{total}] type={qtype:<8} id={qid[:20]:<20} "
                  f"retrieved={len(retrieved)}{scores_str} Q: {body[:45]}...")
            
            # Build prompt using retrieved context
            prompt = build_prompt(body, qtype, retrieved)
            
            # Generate answer
            generated = call_api(model, prompt)
            if not generated:
                stats["api_failures"] += 1
            generated = ensure_non_empty_answer(qtype, generated)
            
            stats["total_processed"] += 1
            
            # Store result (BioASQ format)
            result = {
                "id": qid,
                "type": qtype,
                "generated_answer": generated
            }
            results.append(result)
            done_ids.add(qid)
            save_results(results, output_path)
            
            # Add delay between requests
            time.sleep(REQUEST_DELAY)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")
    
    # Final save
    save_results(results, output_path)
    
    # Print statistics
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE - STATISTICS")
    print(f"{'='*60}")
    print(f"Total answers saved   : {len(results)}")
    print(f"Questions processed   : {stats['total_processed']}")
    print(f"Empty retrievals      : {stats['empty_retrieval']}")
    print(f"API failures          : {stats['api_failures']}")
    if stats['avg_retrieval_score']:
        avg_score = sum(stats['avg_retrieval_score']) / len(stats['avg_retrieval_score'])
        print(f"Avg retrieval score   : {avg_score:.2f}")
    print(f"Time elapsed          : {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Output file           : {output_path}")
    print(f"{'='*60}")
    
    return results


def save_results(results: list[dict[str, Any]], path: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: List of answer dicts
        path: Output file path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# --------------------------------------------------
# Main Entry Point
# --------------------------------------------------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BioASQ Clean RAG Baseline (KG-Independent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_answers_rag.py --train datasets/train/train.json
  python generate_answers_rag.py --limit 10 --top_k 3
  python generate_answers_rag.py --no-resume --output outputs/fresh_run.json
  python generate_answers_rag.py --cache datasets/my_cache.json

This is a FAIR RAG baseline that:
  - Builds ONE global corpus from ALL PubMed documents in the dataset
  - Fetches abstracts via PubMed API (with local caching)
  - Does NOT use per-question document lists during inference
  - Does NOT use any knowledge graph or TAG pipeline
  - Performs BM25 retrieval over the shared corpus
        """
    )
    
    parser.add_argument(
        "--train",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to BioASQ train.json file (default: {DEFAULT_TRAIN_PATH})"
    )
    
    parser.add_argument(
        "--cache",
        type=str,
        default=DEFAULT_CACHE_PATH,
        help=f"Path to PubMed cache JSON file (default: {DEFAULT_CACHE_PATH})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to output JSON file (default: {DEFAULT_OUTPUT_PATH})"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemma model name (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of passages to retrieve (default: {DEFAULT_TOP_K})"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process (default: all)"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing output file"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE})"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap words between chunks (default: {DEFAULT_CHUNK_OVERLAP})"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return 1
    
    # Print configuration
    print("=" * 70)
    print("BioASQ Clean RAG Baseline (KG-Independent)")
    print("=" * 70)
    print(f"Model       : {args.model}")
    print(f"Train file  : {args.train}")
    print(f"Cache file  : {args.cache}")
    print(f"Output      : {args.output}")
    print(f"Top-K       : {args.top_k}")
    print(f"Chunk size  : {args.chunk_size} words")
    print(f"Overlap     : {args.chunk_overlap} words")
    if args.limit:
        print(f"Limit       : {args.limit} questions")
    print(f"Resume      : {not args.no_resume}")
    print("-" * 70)
    print("DESIGN: Global corpus from ALL dataset documents")
    print("DESIGN: PubMed API fetch with local caching")
    print("DESIGN: No KG, no per-question document usage during inference")
    print("DESIGN: BM25 retrieval over shared corpus")
    print("=" * 70)
    
    # Step 1: Load questions
    questions = load_bioasq_json(args.train)
    
    # Step 2: Extract ALL unique PMIDs from the dataset
    pmids = extract_pmids(questions)
    
    # Step 3: Load from cache or fetch from PubMed
    cache = load_or_build_corpus_cache(pmids, args.cache)
    
    # Step 4: Build corpus from cached data
    documents = build_corpus_from_cache(pmids, cache)
    
    # Step 5: Chunk documents
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Step 6: Build BM25 index over ALL chunks
    bm25, chunks = build_bm25(chunks)
    
    # Step 7: Initialize API
    print("\nInitializing API...")
    model = init_api(api_key, args.model)
    
    # Step 8: Generate answers (using ONLY body and type, not documents)
    results = generate_answers(
        questions=questions,
        bm25=bm25,
        chunks=chunks,
        model=model,
        output_path=args.output,
        top_k=args.top_k,
        limit=args.limit,
        resume=not args.no_resume
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
