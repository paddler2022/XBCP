#!/bin/bash
# GPT-OSS-120B (remote) on 3 corpora, 5 retrievers each
# Prerequisites: remote endpoint at $URL
set -e
T=8
CMD="python search_agent/gpt_oss_remote_client.py --num-threads $T"

# ── Original ──
$CMD --searcher-type faiss --index-path "indexes/original/qwen3-embedding-8b/corpus*.pkl" --model-name Qwen/Qwen3-Embedding-8B --normalize --output-dir runs/original/gpt_oss_120b_qwen3-8b
$CMD --searcher-type faiss --index-path "indexes/original/qwen3-embedding-4b/corpus*.pkl" --model-name Qwen/Qwen3-Embedding-4B --normalize --output-dir runs/original/gpt_oss_120b_qwen3-4b
$CMD --searcher-type faiss --index-path "indexes/original/arctic-l/corpus*.pkl" --model-name Snowflake/snowflake-arctic-embed-l-v2.0 --pooling cls --normalize --attn-implementation sdpa --task-prefix "query: " --output-dir runs/original/gpt_oss_120b_arctic-l
$CMD --searcher-type faiss --index-path "indexes/original/multilingual_e5-large/corpus*.pkl" --model-name intfloat/multilingual-e5-large --pooling mean --normalize --attn-implementation sdpa --task-prefix "query: " --output-dir runs/original/gpt_oss_120b_e5-large
$CMD --searcher-type bm25 --index-path indexes/original/bm25 --output-dir runs/original/gpt_oss_120b_bm25

# ── Crosslingual ──
$CMD --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_120b_qwen3-8b
$CMD --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-4b/corpus.pkl --model-name Qwen/Qwen3-Embedding-4B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/gpt_oss_120b_qwen3-4b
$CMD --searcher-type faiss --index-path indexes/crosslingual/arctic-l/corpus.pkl --model-name Snowflake/snowflake-arctic-embed-l-v2.0 --pooling cls --normalize --attn-implementation sdpa --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --task-prefix "query: " --output-dir runs/crosslingual/gpt_oss_120b_arctic-l
$CMD --searcher-type faiss --index-path indexes/crosslingual/multilingual_e5-large/corpus.pkl --model-name intfloat/multilingual-e5-large --pooling mean --normalize --attn-implementation sdpa --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --task-prefix "query: " --output-dir runs/crosslingual/gpt_oss_120b_e5-large
$CMD --searcher-type bm25 --index-path indexes/crosslingual/bm25 --output-dir runs/crosslingual/gpt_oss_120b_bm25

# ── Multilingual ──
$CMD --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_120b_qwen3-8b
$CMD --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-4b/corpus.pkl --model-name Qwen/Qwen3-Embedding-4B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/gpt_oss_120b_qwen3-4b
$CMD --searcher-type faiss --index-path indexes/multilingual/arctic-l/corpus.pkl --model-name Snowflake/snowflake-arctic-embed-l-v2.0 --pooling cls --normalize --attn-implementation sdpa --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --task-prefix "query: " --output-dir runs/multilingual/gpt_oss_120b_arctic-l
$CMD --searcher-type faiss --index-path indexes/multilingual/multilingual_e5-large/corpus.pkl --model-name intfloat/multilingual-e5-large --pooling mean --normalize --attn-implementation sdpa --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --task-prefix "query: " --output-dir runs/multilingual/gpt_oss_120b_e5-large
$CMD --searcher-type bm25 --index-path indexes/multilingual/bm25 --output-dir runs/multilingual/gpt_oss_120b_bm25
