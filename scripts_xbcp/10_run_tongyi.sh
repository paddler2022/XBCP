#!/bin/bash
# Tongyi-DeepResearch-30B-A3B (remote) on 3 corpora, BM25 + Qwen3-8B
# Prerequisites: vLLM at remote endpoint (no reasoning-parser, no tool-parser)
set -e
T=8
URL="https://your-tongyi-endpoint/v1"
CMD="python search_agent/tongyi_client.py --model-url $URL --num-threads $T --quiet"

# --- original ---
$CMD --searcher-type faiss --index-path "indexes/original/qwen3-embedding-8b/corpus*.pkl" --model-name Qwen/Qwen3-Embedding-8B --normalize --output-dir runs/original/tongyi_qwen3-8b
$CMD --searcher-type bm25 --index-path indexes/original/bm25 --output-dir runs/original/tongyi_bm25

# --- crosslingual ---
$CMD --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/tongyi_qwen3-8b
$CMD --searcher-type bm25 --index-path indexes/crosslingual/bm25 --output-dir runs/crosslingual/tongyi_bm25

# --- multilingual ---
$CMD --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/tongyi_qwen3-8b
$CMD --searcher-type bm25 --index-path indexes/multilingual/bm25 --output-dir runs/multilingual/tongyi_bm25
