#!/bin/bash
# Build indexes for XBCP corpus (crosslingual and multilingual).
# Strategy: encode only 5040 evidence docs, then merge with original corpus.pkl
#
# Usage:
#   bash scripts_xbcp/04_build_indexes.sh crosslingual all
#   bash scripts_xbcp/04_build_indexes.sh multilingual qwen3-0.6b
#   bash scripts_xbcp/04_build_indexes.sh crosslingual bm25

set -e

VERSION="${1:?Usage: $0 <crosslingual|multilingual> <model|all>}"
TARGET="${2:-all}"

EVIDENCE_TEVATRON="data/${VERSION}/evidence_docs_tevatron.jsonl"
INDEX_BASE="indexes/${VERSION}"
QUERY_JSONL="data/original/browsecomp_plus_decrypted.jsonl"
RUNS_DIR="runs/${VERSION}"
QREL_EVIDENCE="topics-qrels/qrel_evidence.txt"
QREL_GOLDS="topics-qrels/qrel_golds.txt"
CORPUS_PYSERINI="data/${VERSION}/${VERSION}_corpus.jsonl"
CORPUS_DIR="data/${VERSION}/corpus_dir"

mkdir -p "$RUNS_DIR"

# ── Model configs ──
# Format: key|model_path|orig_dir|pooling|query_prefix|passage_prefix|query_max_len|passage_max_len|normalize|corpus_batch|query_batch|attn_impl
declare -A MODELS
MODELS=(
    ["qwen3-embedding-0.6b"]="Qwen/Qwen3-Embedding-0.6B|qwen3-embedding-0.6b|eos|Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:| |512|4096|yes|32|156|"
    ["qwen3-embedding-4b"]="Qwen/Qwen3-Embedding-4B|qwen3-embedding-4b|eos|Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:| |512|4096|yes|16|64|"
    ["qwen3-embedding-8b"]="Qwen/Qwen3-Embedding-8B|qwen3-embedding-8b|eos|Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:| |512|4096|yes|8|32|"
    ["arctic-l"]="Snowflake/snowflake-arctic-embed-l-v2.0|arctic-l|cls|query: | |512|8192|yes|8|32|sdpa"
    ["multilingual_e5-large"]="intfloat/multilingual-e5-large|multilingual_e5-large|mean|query: |passage: |512|512|yes|32|156|sdpa"
)


build_bm25() {
    echo "=== Building BM25 index for ${VERSION} ==="
    local INDEX_DIR="${INDEX_BASE}/bm25"
    if [ ! -d "$INDEX_DIR" ]; then
        mkdir -p "$CORPUS_DIR"
        if [ ! -f "$CORPUS_DIR/${VERSION}_corpus.jsonl" ]; then
            cp "$CORPUS_PYSERINI" "$CORPUS_DIR/${VERSION}_corpus.jsonl"
        fi
        python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input "$CORPUS_DIR" \
            --index "$INDEX_DIR" \
            --generator DefaultLuceneDocumentGenerator \
            --threads 16 --storePositions --storeDocvectors --storeRaw
    fi

    local TREC_FILE="$RUNS_DIR/bm25_top1000.trec"
    if [ ! -f "$TREC_FILE" ]; then
        echo "  Running BM25 retrieval..."
        python -m pyserini.search.lucene \
            --index "$INDEX_DIR" \
            --topics topics-qrels/queries.tsv \
            --output "$TREC_FILE" \
            --bm25 --hits 1000
    fi
    echo "  --- Evidence ---"
    python -m pyserini.eval.trec_eval -c -m recall.5,100,1000 -m ndcg_cut.10 "$QREL_EVIDENCE" "$TREC_FILE"
    echo "  --- Gold ---"
    python -m pyserini.eval.trec_eval -c -m recall.5,100,1000 -m ndcg_cut.10 "$QREL_GOLDS" "$TREC_FILE"
}

merge_embeddings() {
    local INDEX_DIR="$1"
    local ORIG_INDEX="$2"
    local EVIDENCE_PKL="$3"

    if [ -f "$INDEX_DIR/corpus.pkl" ]; then
        echo "  corpus.pkl already exists, skipping merge"
        return
    fi

    echo "  Merging evidence embeddings with original corpus..."
    python -c "
import pickle, glob, numpy as np

orig_files = sorted(glob.glob('${ORIG_INDEX}/corpus.pkl'))
if not orig_files:
    orig_files = sorted(glob.glob('${ORIG_INDEX}/corpus.*.pkl'))
if not orig_files:
    raise FileNotFoundError('No corpus pkl found in ${ORIG_INDEX}/')

print(f'Loading original from {len(orig_files)} files...')
orig_reps, orig_lookup = [], []
for f in orig_files:
    reps, lookup = pickle.load(open(f, 'rb'))
    orig_reps.append(reps)
    orig_lookup.extend(lookup)
orig_reps = np.concatenate(orig_reps, axis=0)
print(f'  Original: {orig_reps.shape[0]} docs, dim={orig_reps.shape[1]}')

ev_reps, ev_lookup = pickle.load(open('${EVIDENCE_PKL}', 'rb'))
print(f'  Evidence: {ev_reps.shape[0]} docs')

ev_map = {str(docid): vec for docid, vec in zip(ev_lookup, ev_reps)}
replaced = 0
for i, docid in enumerate(orig_lookup):
    if str(docid) in ev_map:
        orig_reps[i] = ev_map[str(docid)]
        replaced += 1

print(f'  Replaced {replaced} vectors')
with open('${INDEX_DIR}/corpus.pkl', 'wb') as f:
    pickle.dump((orig_reps, orig_lookup), f)
print(f'  Saved to ${INDEX_DIR}/corpus.pkl')
"
}

search_and_eval() {
    local INDEX_DIR="$1"
    local KEY="$2"

    local TREC_FILE="$RUNS_DIR/${KEY}_top1000.trec"
    if [ ! -f "$TREC_FILE" ]; then
        echo "  Searching..."
        python -m tevatron.retriever.driver.search \
            --query_reps "$INDEX_DIR/query.pkl" \
            --passage_reps "$INDEX_DIR/corpus.pkl" \
            --depth 1000 --batch_size 128 --save_text \
            --save_ranking_to "$RUNS_DIR/${KEY}_top1000.txt"
        python -m tevatron.utils.format.convert_result_to_trec \
            --input "$RUNS_DIR/${KEY}_top1000.txt" \
            --output "$TREC_FILE"
    fi

    echo "  --- Evidence ---"
    python -m pyserini.eval.trec_eval -c -m recall.5,100,1000 -m ndcg_cut.10 "$QREL_EVIDENCE" "$TREC_FILE"
    echo "  --- Gold ---"
    python -m pyserini.eval.trec_eval -c -m recall.5,100,1000 -m ndcg_cut.10 "$QREL_GOLDS" "$TREC_FILE"
}

build_dense_tevatron() {
    local KEY="$1"
    local CONFIG="${MODELS[$KEY]}"
    IFS='|' read -r MODEL_PATH ORIG_DIR POOLING QUERY_PREFIX PASSAGE_PREFIX QUERY_MAX_LEN PASSAGE_MAX_LEN NORMALIZE CORPUS_BATCH QUERY_BATCH ATTN_IMPL <<< "$CONFIG"

    local INDEX_DIR="${INDEX_BASE}/${KEY}"
    local ORIG_INDEX="indexes/original/${ORIG_DIR}"
    mkdir -p "$INDEX_DIR"

    echo "=== Building dense index: $KEY (${VERSION}) ==="

    NORMALIZE_FLAG=""
    [ "$NORMALIZE" = "yes" ] && NORMALIZE_FLAG="--normalize"

    ATTN_FLAG=""
    [ -n "$ATTN_IMPL" ] && ATTN_FLAG="--attn_implementation $ATTN_IMPL"

    # Step 1: Encode evidence docs
    local EVIDENCE_PKL="${INDEX_DIR}/evidence.pkl"
    if [ ! -f "$EVIDENCE_PKL" ]; then
        echo "  Encoding evidence docs with Tevatron..."
        python -m tevatron.retriever.driver.encode \
            --model_name_or_path "$MODEL_PATH" \
            --dataset_path "$EVIDENCE_TEVATRON" \
            --encode_output_path "$EVIDENCE_PKL" \
            --passage_max_len "$PASSAGE_MAX_LEN" \
            --pooling "$POOLING" \
            --passage_prefix "$PASSAGE_PREFIX" \
            --per_device_eval_batch_size "$CORPUS_BATCH" \
            --fp16 $NORMALIZE_FLAG $ATTN_FLAG
    fi

    # Step 2: Merge
    merge_embeddings "$INDEX_DIR" "$ORIG_INDEX" "$EVIDENCE_PKL"

    # Step 3: Encode queries (reuse from original if exists)
    if [ ! -f "$INDEX_DIR/query.pkl" ]; then
        if [ -f "indexes/original/${ORIG_DIR}/query.pkl" ]; then
            echo "  Reusing original query.pkl"
            cp "indexes/original/${ORIG_DIR}/query.pkl" "$INDEX_DIR/query.pkl"
        else
            echo "  Encoding queries..."
            python -m tevatron.retriever.driver.encode \
                --model_name_or_path "$MODEL_PATH" \
                --dataset_path "$QUERY_JSONL" \
                --encode_output_path "$INDEX_DIR/query.pkl" \
                --query_max_len "$QUERY_MAX_LEN" \
                --encode_is_query \
                --pooling "$POOLING" \
                --query_prefix "$QUERY_PREFIX" \
                --per_device_eval_batch_size "$QUERY_BATCH" \
                --fp16 $NORMALIZE_FLAG $ATTN_FLAG
        fi
    fi

    # Step 4: Search + Evaluate
    search_and_eval "$INDEX_DIR" "$KEY"
}

# ── Main dispatch ──
ALL_KEYS="qwen3-embedding-0.6b qwen3-embedding-4b qwen3-embedding-8b arctic-l multilingual_e5-large"

case "$TARGET" in
    bm25) build_bm25 ;;
    qwen3-embedding-0.6b|qwen3-embedding-4b|qwen3-embedding-8b|arctic-l|multilingual_e5-large) build_dense_tevatron "$TARGET" ;;
    all)
        build_bm25
        for key in $ALL_KEYS; do
            build_dense_tevatron "$key"
        done
        ;;
    *) echo "Usage: $0 <crosslingual|multilingual> [bm25|qwen3-embedding-0.6b|qwen3-embedding-4b|qwen3-embedding-8b|arctic-l|multilingual_e5-large|all]"; exit 1 ;;
esac

echo "=== All done for ${VERSION} ==="
