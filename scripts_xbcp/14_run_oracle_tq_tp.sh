#!/bin/bash
# Oracle tq+tp: translated prompt + translated query + translated evidence
# Tests pure agent reasoning ability in target language (no retrieval, no language switching)
#
# Usage:
#   bash scripts_xbcp/26_run_oracle_tq_tp.sh              # local GPT-OSS-20B
#   bash scripts_xbcp/26_run_oracle_tq_tp.sh remote        # remote GPT-OSS-120B
T=8
Q="data/crosslingual/queries_translated.tsv"
LM="data/crosslingual/query_lang_assignment.json"

if [ "$1" = "remote" ]; then
    URL="https://your-gpt-oss-120b-endpoint/v1"
    MODEL="openai/gpt-oss-120b"
    AGENT="gpt_oss_120b"
else
    URL="http://localhost:8000/v1"
    MODEL="openai/gpt-oss-20b"
    AGENT="gpt_oss_20b"
fi

python scripts_xbcp/07_run_oracle.py --model $MODEL --model-url $URL --agent-name ${AGENT}_tq_tp --query $Q --lang-map $LM --num-threads $T --version crosslingual
