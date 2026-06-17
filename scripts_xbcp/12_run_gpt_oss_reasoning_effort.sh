#!/bin/bash
# GPT-OSS-20B + Qwen3-Embedding-8B, reasoning-effort low & high, 3 corpora
# Prerequisites: vllm serve openai/gpt-oss-20b --port 8000 --gpu-memory-utilization 0.9
set -e
T=8

# ── Low ──
python search_agent/gpt_oss_client.py --reasoning-effort low --num-threads $T --searcher-type faiss --index-path "indexes/original/qwen3-embedding-8b/corpus*.pkl" --model-name Qwen/Qwen3-Embedding-8B --normalize --output-dir runs/original/gpt_oss_20b_re-low_qwen3-8b
python search_agent/gpt_oss_client.py --reasoning-effort low --num-threads $T --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_re-low_qwen3-8b
python search_agent/gpt_oss_client.py --reasoning-effort low --num-threads $T --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_re-low_qwen3-8b

# ── High ──
python search_agent/gpt_oss_client.py --reasoning-effort high --num-threads $T --searcher-type faiss --index-path "indexes/original/qwen3-embedding-8b/corpus*.pkl" --model-name Qwen/Qwen3-Embedding-8B --normalize --output-dir runs/original/gpt_oss_20b_re-high_qwen3-8b
python search_agent/gpt_oss_client.py --reasoning-effort high --num-threads $T --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_re-high_qwen3-8b
python search_agent/gpt_oss_client.py --reasoning-effort high --num-threads $T --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_re-high_qwen3-8b
