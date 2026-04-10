#!/usr/bin/env python3
"""
Run the existing RAG baseline over every BioASQ JSON file in datasets/test.

This reuses the shared-corpus BM25 pipeline from generate_answers_rag.py, but
builds the shared corpus from the train set only, then loops over all test-set
files and writes one output JSON per input file.
"""

import argparse
import os
from pathlib import Path

from generate_answers_rag import (
    DEFAULT_CACHE_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_TRAIN_PATH,
    PROJECT_ROOT,
    build_bm25,
    build_corpus_from_cache,
    chunk_documents,
    extract_pmids,
    generate_answers,
    init_api,
    load_bioasq_json,
    load_or_build_corpus_cache,
)


DEFAULT_TEST_DIR = str(PROJECT_ROOT / "datasets" / "test")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "test_rag")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BioASQ RAG baseline for every JSON file in datasets/test."
    )
    parser.add_argument(
        "--train",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help=f"Path to train JSON used to build the retrieval corpus (default: {DEFAULT_TRAIN_PATH})",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=DEFAULT_TEST_DIR,
        help=f"Directory containing test JSON files (default: {DEFAULT_TEST_DIR})",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=DEFAULT_CACHE_PATH,
        help=f"Path to PubMed cache JSON file (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated answers (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemma model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of passages to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit questions per test file (default: all)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh for each output file, ignoring any existing output",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Words per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap words between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it with: export GEMINI_API_KEY=your_api_key")
        return 1

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return 1

    test_files = sorted(test_dir.glob("*.json"))
    if not test_files:
        print(f"Error: No JSON files found in {test_dir}")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("BioASQ Test RAG Generator")
    print("=" * 70)
    print(f"Model       : {args.model}")
    print(f"Train file  : {args.train}")
    print(f"Test dir    : {test_dir}")
    print(f"Cache file  : {args.cache}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Top-K       : {args.top_k}")
    print(f"Chunk size  : {args.chunk_size} words")
    print(f"Overlap     : {args.chunk_overlap} words")
    if args.limit:
        print(f"Limit       : {args.limit} questions per file")
    print(f"Resume      : {not args.no_resume}")
    print(f"Files found : {len(test_files)}")
    print("=" * 70)

    print("\nBuilding shared retrieval corpus from train set...")
    train_questions = load_bioasq_json(args.train)
    train_pmids = extract_pmids(train_questions)
    cache = load_or_build_corpus_cache(train_pmids, args.cache)
    documents = build_corpus_from_cache(train_pmids, cache)
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    bm25, chunks = build_bm25(chunks)

    model = init_api(api_key, args.model)

    for test_file in test_files:
        output_path = Path(args.output_dir) / f"{test_file.stem}_rag_answers.json"

        print(f"\nProcessing test file: {test_file}")
        print(f"Output path         : {output_path}")
        print("-" * 70)

        questions = load_bioasq_json(str(test_file))

        generate_answers(
            questions=questions,
            bm25=bm25,
            chunks=chunks,
            model=model,
            output_path=str(output_path),
            top_k=args.top_k,
            limit=args.limit,
            resume=not args.no_resume,
        )

    print("\nAll test files processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
