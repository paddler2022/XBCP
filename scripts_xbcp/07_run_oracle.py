"""
Oracle evaluation: all evidence docs are provided in the prompt. No retrieval.
Agent reads evidence docs and answers the question directly.

Usage:
  python scripts_xbcp/07_run_oracle.py --version multilingual
  python scripts_xbcp/07_run_oracle.py --version crosslingual
  python scripts_xbcp/07_run_oracle.py --version multilingual --model-url http://localhost:8000/v1
"""

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import openai
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from search_agent.prompts import QUERY_TEMPLATE_ORACLE, QUERY_TEMPLATE_ORACLE_MULTILINGUAL

sys.stdout.reconfigure(encoding="utf-8")


def load_qrels(path):
    """Load qrels: query_id -> set of docids."""
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    qrels.setdefault(qid, set()).add(docid)
    return qrels


def load_corpus(path):
    """Load corpus: docid -> text. If path is None, load from HuggingFace."""
    corpus = {}
    if path is None:
        from datasets import load_dataset
        print("Loading corpus from HuggingFace...")
        ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
        for row in ds:
            corpus[str(row["docid"])] = row["text"]
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                corpus[str(d.get("docid", d.get("id", "")))] = d.get("text", d.get("contents", ""))
    return corpus


def load_queries(path):
    queries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries.append((row[0].strip(), row[1].strip()))
    return queries


def build_oracle_prompt(question, evidence_docs, language=None):
    docs_text = ""
    for i, (docid, text) in enumerate(evidence_docs):
        docs_text += f"\n--- Document {i+1} (docid: {docid}) ---\n{text}\n"
    if language and language != "English" and language in QUERY_TEMPLATE_ORACLE_MULTILINGUAL:
        return QUERY_TEMPLATE_ORACLE_MULTILINGUAL[language].format(Question=question, EvidenceDocuments=docs_text)
    return QUERY_TEMPLATE_ORACLE.format(Question=question, EvidenceDocuments=docs_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["crosslingual", "multilingual", "original"], required=True)
    parser.add_argument("--query", default="topics-qrels/queries.tsv")
    parser.add_argument("--qrel-path", default="topics-qrels/qrel_evidence.txt")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--agent-name", default="gpt_oss_20b", help="Agent name for output dir")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable thinking for Qwen3.6")
    parser.add_argument("--lang-map", type=str, default=None, help="Query language map JSON for multilingual prompt")
    args = parser.parse_args()

    if args.version == "original":
        corpus_path = None  # will load from HuggingFace
    else:
        corpus_path = f"data/{args.version}/{args.version}_corpus_tevatron.jsonl"
    output_dir = Path(f"runs/{args.version}/{args.agent_name}_oracle")
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = openai.OpenAI(base_url=args.model_url, api_key=api_key)

    lang_map = {}
    if args.lang_map:
        lang_map = json.loads(Path(args.lang_map).read_text(encoding="utf-8"))

    qrels = load_qrels(args.qrel_path)
    corpus = load_corpus(corpus_path)
    queries = load_queries(args.query)

    # Skip already processed
    processed_ids = set()
    for json_path in output_dir.glob("run_*.json"):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
                qid = meta.get("query_id")
                if qid:
                    processed_ids.add(str(qid))
        except Exception:
            continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]
    print(f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)})")

    completed_lock = threading.Lock()
    completed_count = [0]

    MAX_DOC_CHARS = 100000

    def handle_query(qid, qtext, pbar=None):
        evidence_docids = qrels.get(qid, set())
        evidence_docs = [(did, corpus[did][:MAX_DOC_CHARS]) for did in evidence_docids if did in corpus]
        total_chars = sum(len(text) for _, text in evidence_docs)

        language = lang_map.get(qid)
        prompt = build_oracle_prompt(qtext, evidence_docs, language=language)

        try:
            kwargs = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
            }
            if args.disable_thinking:
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            elif args.reasoning_effort:
                kwargs["extra_body"] = {"reasoning_effort": args.reasoning_effort}

            response = client.chat.completions.create(**kwargs)
            answer = response.choices[0].message.content

            result = {
                "query_id": qid,
                "query": qtext,
                "evidence_docids": list(evidence_docids),
                "num_evidence_docs": len(evidence_docs),
                "total_evidence_chars": total_chars,
                "prompt": prompt,
                "response": answer,
                "timestamp": datetime.now().isoformat(),
            }

            out_path = output_dir / f"run_{qid}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            with completed_lock:
                completed_count[0] += 1
                if pbar:
                    pbar.set_postfix(completed=completed_count[0])

        except Exception as e:
            print(f"[Error] Query id={qid} failed: {e}")

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Oracle queries", unit="query") as pbar:
            for qid, qtext in pbar:
                handle_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Oracle queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(handle_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]
            for _ in as_completed(futures):
                pbar.update(1)

    print(f"\nDone. {completed_count[0]} queries completed -> {output_dir}")


if __name__ == "__main__":
    main()
