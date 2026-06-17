#!/bin/bash
# Batch eval Oracle directories under runs_latest/ using OpenRouter judge.
# Oracle only needs Accuracy (no citation, no retrieval recall).
#
# Usage:
#   bash scripts_evaluation/batch_eval_oracle_openrouter.sh
#   bash scripts_evaluation/batch_eval_oracle_openrouter.sh --force
EVAL_DIR="evals_latest"
FORCE=""
if [[ "$*" == *"--force"* ]]; then
    FORCE="1"
fi
EXTRA_ARGS="${@}"

eval_oracle() {
    local input_dir="$1"
    local count=$(ls "$input_dir"/run_*.json 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        return
    fi
    local rel_path="${input_dir#runs_latest/runs/}"
    local eval_summary="${EVAL_DIR}/${rel_path}/evaluation_summary.json"
    if [ -f "$eval_summary" ] && [ -z "$FORCE" ]; then
        echo "SKIP (done): $input_dir"
        return
    fi
    echo "=== Evaluating Oracle: $input_dir ($count runs) ==="
    python scripts_evaluation/evaluate_oracle_openrouter.py \
        --input_dir "$input_dir" \
        --eval_dir "$EVAL_DIR" \
        $EXTRA_ARGS
}

for dir in runs_latest/runs/*/*oracle*; do
    [ -d "$dir" ] && eval_oracle "$dir"
done

echo ""
echo "=== All Oracle evaluations done ==="
echo "Results in ./${EVAL_DIR}/"
