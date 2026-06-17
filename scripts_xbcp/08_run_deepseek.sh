#!/bin/bash
# DeepSeek V4 Pro on 3 corpora, BM25 + Qwen3-Embedding-8B
set -e
CMD="python search_agent/deepseek_remote_client.py --num-threads 8"

# ── BM25 (CPU only) ──
$CMD --searcher-type bm25 --index-path indexes/original/bm25 --output-dir runs/original/deepseek_v4_pro_bm25
$CMD --searcher-type bm25 --index-path indexes/crosslingual/bm25 --output-dir runs/crosslingual/deepseek_v4_pro_bm25
$CMD --searcher-type bm25 --index-path indexes/multilingual/bm25 --output-dir runs/multilingual/deepseek_v4_pro_bm25

# ── Qwen3-Embedding-8B (needs GPU) ──
$CMD --searcher-type faiss --index-path "indexes/original/qwen3-embedding-8b/corpus*.pkl" --model-name Qwen/Qwen3-Embedding-8B --normalize --output-dir runs/original/deepseek_v4_pro_qwen3-8b
$CMD --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/deepseek_v4_pro_qwen3-8b
$CMD --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/deepseek_v4_pro_qwen3-8b
