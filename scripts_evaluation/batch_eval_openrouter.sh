#!/bin/bash
# Batch eval all run directories under runs_latest/ using OpenRouter judge.
# Oracle directories are skipped (use batch_eval_oracle_openrouter.sh instead).
# Skips already-evaluated queries (resume-safe).
#
# Usage:
#   bash scripts_evaluation/batch_eval_openrouter.sh
#   bash scripts_evaluation/batch_eval_openrouter.sh --force   # re-eval everything
EVAL_DIR="evals_latest"
FORCE=""
if [[ "$*" == *"--force"* ]]; then
    FORCE="1"
fi
EXTRA_ARGS="${@}"

eval_dir() {
    local input_dir="$1"
    # Skip oracle directories
    if [[ "$input_dir" == *oracle* ]]; then
        echo "SKIP (oracle): $input_dir"
        return
    fi
    local count=$(ls "$input_dir"/run_*.json 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        return
    fi
    # Check if already fully evaluated (summary exists)
    local rel_path="${input_dir#runs_latest/runs/}"
    rel_path="${rel_path#runs_latest/reruns/}"
    local eval_summary="${EVAL_DIR}/${rel_path}/evaluation_summary.json"
    if [ -f "$eval_summary" ] && [ -z "$FORCE" ]; then
        echo "SKIP (done): $input_dir"
        return
    fi
    echo "=== Evaluating $input_dir ($count runs) ==="
    python scripts_evaluation/evaluate_with_openrouter.py \
        --input_dir "$input_dir" \
        --eval_dir "$EVAL_DIR" \
        $EXTRA_ARGS
}

# runs_latest/runs/
for dir in runs_latest/runs/*/*; do
    [ -d "$dir" ] && eval_dir "$dir"
done

# runs_latest/reruns/
for dir in runs_latest/reruns/*/*; do
    [ -d "$dir" ] && eval_dir "$dir"
done

echo ""
echo "=== All evaluations done ==="
echo "Results in ./${EVAL_DIR}/"
