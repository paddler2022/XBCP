#!/bin/bash
# Run Qwen3.6-35B-A3B agent with selected retrievers on crosslingual and multilingual corpus.
# Prerequisites: model served at --model-url endpoint (e.g. vLLM or SGLang)

T=8
MODEL="Qwen/Qwen3.6-35B-A3B"
URL="https://your-qwen36-endpoint/v1"

# ── Crosslingual ──
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type bm25 --index-path indexes/crosslingual/bm25 --output-dir runs/crosslingual/qwen36_35b_bm25 --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-4b/corpus.pkl --model-name Qwen/Qwen3-Embedding-4B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/qwen36_35b_qwen3-4b --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/qwen36_35b_qwen3-8b --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/crosslingual/multilingual_e5-large/corpus.pkl --model-name intfloat/multilingual-e5-large --pooling mean --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/qwen36_35b_e5-large --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/crosslingual/arctic-l/corpus.pkl --model-name Snowflake/snowflake-arctic-embed-l-v2.0 --pooling cls --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl --output-dir runs/crosslingual/qwen36_35b_arctic-l --num-threads $T

# ── Multilingual ──
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type bm25 --index-path indexes/multilingual/bm25 --output-dir runs/multilingual/qwen36_35b_bm25 --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-4b/corpus.pkl --model-name Qwen/Qwen3-Embedding-4B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/qwen36_35b_qwen3-4b --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/multilingual/qwen3-embedding-8b/corpus.pkl --model-name Qwen/Qwen3-Embedding-8B --normalize --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/qwen36_35b_qwen3-8b --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/multilingual/multilingual_e5-large/corpus.pkl --model-name intfloat/multilingual-e5-large --pooling mean --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/qwen36_35b_e5-large --num-threads $T
python search_agent/openai_remote_client.py --model $MODEL --model-url $URL --searcher-type faiss --index-path indexes/multilingual/arctic-l/corpus.pkl --model-name Snowflake/snowflake-arctic-embed-l-v2.0 --pooling cls --normalize --attn-implementation sdpa --task-prefix "query: " --dataset-name data/multilingual/multilingual_corpus_tevatron.jsonl --output-dir runs/multilingual/qwen36_35b_arctic-l --num-threads $T
