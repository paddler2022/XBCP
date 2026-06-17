#!/bin/bash
# GPT-OSS-20B and GPT-OSS-120B with translated query + translated prompt
# on crosslingual corpus, BM25 + Qwen3-Embedding-8B
set -e
T=8
Q="data/crosslingual/queries_translated.tsv"
LM="data/crosslingual/query_lang_assignment.json"

# ── GPT-OSS-20B (local vLLM) ──
python search_agent/gpt_oss_client.py --query $Q --lang-map $LM --num-threads $T --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_qwen3-8b_tq_tp
python search_agent/gpt_oss_client.py --query $Q --lang-map $LM --num-threads $T --searcher-type bm25 --index-path indexes/crosslingual/bm25 --output-dir runs/crosslingual/gpt_oss_20b_bm25_tq_tp

# ── GPT-OSS-120B (remote) ──
python search_agent/gpt_oss_remote_client.py --query $Q --lang-map $LM --num-threads $T --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_120b_qwen3-8b_tq_tp
python search_agent/gpt_oss_remote_client.py --query $Q --lang-map $LM --num-threads $T --searcher-type bm25 --index-path indexes/crosslingual/bm25 --output-dir runs/crosslingual/gpt_oss_120b_bm25_tq_tp
