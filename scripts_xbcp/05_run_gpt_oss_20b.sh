#!/bin/bash
# Run gpt-oss-20b agent with selected retrievers on crosslingual and multilingual corpus.
# Prerequisites: vllm serve openai/gpt-oss-20b --port 8000 --gpu-memory-utilization 0.9
# set -e  # uncomment to stop on first error

T=8  # num-threads

# ── Crosslingual ──
python search_agent/gpt_oss_client.py --searcher-type bm25 --index-path indexes/crosslingual/bm25 --output-dir runs/crosslingual/gpt_oss_20b_bm25 --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-4b/corpus.pkl --model-name Qwen/Qwen3-Embedding-4B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_qwen3-4b --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_qwen3-8b --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/crosslingual/multilingual_e5-large/corpus.pkl --model-name intfloat/multilingual-e5-large --pooling mean --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_e5-large --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/crosslingual/arctic-l/corpus.pkl --model-name Snowflake/snowflake-arctic-embed-l-v2.0 --pooling cls --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_20b_arctic-l --num-threads $T

# ── Multilingual ──
python search_agent/gpt_oss_client.py --searcher-type bm25 --index-path indexes/multilingual/bm25 --output-dir runs/multilingual/gpt_oss_20b_bm25 --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-4b/corpus.pkl --model-name Qwen/Qwen3-Embedding-4B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_qwen3-4b --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_qwen3-8b --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/multilingual/multilingual_e5-large/corpus.pkl --model-name intfloat/multilingual-e5-large --pooling mean --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_e5-large --num-threads $T
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/multilingual/arctic-l/corpus.pkl --model-name Snowflake/snowflake-arctic-embed-l-v2.0 --pooling cls --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_arctic-l --num-threads $T

# ── Oracle (multilingual, retrieval on evidence docs only) ──
python search_agent/gpt_oss_client.py --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/evidence.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/evidence_docs_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_20b_oracle_qwen3-8b --num-threads $T
