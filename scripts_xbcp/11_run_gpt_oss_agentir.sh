#!/bin/bash
# GPT-OSS-20B + AgentIR (reasoning-aware retrieval) + Qwen3-Embedding-8B, 3 corpora
# Prerequisites: vllm serve openai/gpt-oss-20b --port 8000 --gpu-memory-utilization 0.9
set -e
T=8
CMD="python search_agent/gpt_oss_agentir_client.py --reasoning-effort medium --num-threads $T"

$CMD --searcher-type faiss --index-path "indexes/original/qwen3-embedding-8b/corpus*.pkl" --model-name Qwen/Qwen3-Embedding-8B --normalize --output-dir runs/original/gpt_oss_20b_agentir_qwen3-8b
$CMD --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_agentir_qwen3-8b
$CMD --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_agentir_qwen3-8b
