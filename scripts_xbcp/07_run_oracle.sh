#!/bin/bash
# Run Oracle mode: all evidence docs in prompt, no retrieval.

# ── GPT-OSS-20B (vllm serve openai/gpt-oss-20b --port 8000) ──
python scripts_xbcp/07_run_oracle.py --version original --num-threads 8
python scripts_xbcp/07_run_oracle.py --version crosslingual --num-threads 8
python scripts_xbcp/07_run_oracle.py --version multilingual --num-threads 8

# ── Qwen3.6-35B-A3B (remote) ──
python scripts_xbcp/07_run_oracle.py --version original --model Qwen/Qwen3.6-35B-A3B --model-url https://your-qwen36-endpoint/v1 --agent-name qwen36_35b --disable-thinking --num-threads 8
python scripts_xbcp/07_run_oracle.py --version crosslingual --model Qwen/Qwen3.6-35B-A3B --model-url https://your-qwen36-endpoint/v1 --agent-name qwen36_35b --disable-thinking --num-threads 8
python scripts_xbcp/07_run_oracle.py --version multilingual --model Qwen/Qwen3.6-35B-A3B --model-url https://your-qwen36-endpoint/v1 --agent-name qwen36_35b --disable-thinking --num-threads 8

# ── GPT-OSS-120B (remote) ──
python scripts_xbcp/07_run_oracle.py --version original --model openai/gpt-oss-120b --model-url https://your-gpt-oss-120b-endpoint/v1 --agent-name gpt_oss_120b --num-threads 8
python scripts_xbcp/07_run_oracle.py --version crosslingual --model openai/gpt-oss-120b --model-url https://your-gpt-oss-120b-endpoint/v1 --agent-name gpt_oss_120b --num-threads 8
python scripts_xbcp/07_run_oracle.py --version multilingual --model openai/gpt-oss-120b --model-url https://your-gpt-oss-120b-endpoint/v1 --agent-name gpt_oss_120b --num-threads 8
