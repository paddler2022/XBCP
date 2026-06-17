"""
Remote GPT-OSS-120B client via Chat Completions API (vLLM endpoint).
Based on openai_remote_client.py with gpt-oss-specific adjustments:
  - extra_body uses reasoning_effort instead of chat_template_kwargs
  - reasoning field is "reasoning" (not "reasoning_content")

Usage:
  python search_agent/gpt_oss_remote_client.py \
    --model openai/gpt-oss-120b \
    --model-url https://your-gpt-oss-120b-endpoint/v1 \
    --searcher-type faiss \
    --index-path indexes/crosslingual/qwen3-embedding-8b/corpus.pkl \
    --model-name Qwen/Qwen3-Embedding-8B --normalize \
    --dataset-name data/crosslingual/crosslingual_corpus_tevatron.jsonl \
    --output-dir runs/crosslingual/gpt_oss_120b_qwen3-8b \
    --num-threads 8
"""

from __future__ import annotations

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
from prompts import format_query
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType


def _get_reasoning(msg):
    return getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)


class SearchToolHandler:

    def __init__(self, searcher, snippet_max_tokens=None, k=5, include_get_document=True):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def execute_tool(self, tool_name, arguments):
        if tool_name == "local_knowledge_base_retrieval":
            return self._search(arguments["user_query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        raise ValueError(f"Unknown tool: {tool_name}")

    def get_tool_definitions(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "local_knowledge_base_retrieval",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_query": {"type": "string", "description": "Query to search the local knowledge base"}
                    },
                    "required": ["user_query"],
                },
            },
        }]
        if self.include_get_document:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {"type": "string", "description": "Document ID to retrieve"}
                        },
                        "required": ["docid"],
                    },
                },
            })
        return tools

    def _search(self, query):
        candidates = self.searcher.search(query, self.k)
        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                tokens = self.tokenizer.encode(cand["text"], add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    cand["snippet"] = self.tokenizer.decode(tokens[:self.snippet_max_tokens], skip_special_tokens=True)
                else:
                    cand["snippet"] = cand["text"]
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]
        return json.dumps([
            {"docid": c["docid"], "score": c.get("score"), "snippet": c["snippet"]}
            if c.get("score") is not None else {"docid": c["docid"], "snippet": c["snippet"]}
            for c in candidates
        ], indent=2)

    def _get_document(self, docid):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document '{docid}' not found"})
        return json.dumps(result, indent=2)


def run_conversation_with_tools(client, model, messages, tools, tool_handler,
                                max_iterations=100, verbose=False,
                                reasoning_effort="medium"):
    tool_usage = {}
    iteration = 1

    while iteration <= max_iterations:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                extra_body={"reasoning_effort": reasoning_effort},
            )
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            iteration += 1
            continue

        choice = response.choices[0]
        msg = choice.message

        msg_dict = msg.model_dump()
        reasoning = _get_reasoning(msg)
        if reasoning:
            msg_dict["reasoning_content"] = reasoning
        # Strip fields that vLLM doesn't accept in message history
        msg_dict.pop("reasoning", None)
        if not msg_dict.get("tool_calls"):
            msg_dict.pop("tool_calls", None)
        for k in ["refusal", "annotations", "audio", "function_call"]:
            msg_dict.pop(k, None)
        messages.append(msg_dict)

        if not msg.tool_calls:
            return messages, tool_usage, "completed"

        for tc in msg.tool_calls:
            try:
                arguments = json.loads(tc.function.arguments)
                result = tool_handler.execute_tool(tc.function.name, arguments)
                tool_usage[tc.function.name] = tool_usage.get(tc.function.name, 0) + 1
            except Exception as e:
                result = f"Error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if choice.finish_reason == "stop":
            return messages, tool_usage, "completed"

        iteration += 1

    return messages, tool_usage, "incomplete"


def _persist_response(out_dir, model, messages, tool_usage, status, query_id=None, initial_prompt=None):
    os.makedirs(out_dir, exist_ok=True)

    normalized_results = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "") or ""
            reasoning = _get_reasoning(msg) or ""

        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

        if role == "assistant" and reasoning and reasoning.strip():
            normalized_results.append({
                "type": "reasoning",
                "tool_name": None,
                "arguments": None,
                "output": [reasoning.strip()],
            })

        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    tc_id = tc.get("id")
                    tc_name = fn.get("name")
                    tc_args = fn.get("arguments")
                else:
                    tc_id = tc.id
                    tc_name = tc.function.name
                    tc_args = tc.function.arguments

                tool_output = None
                for m2 in messages:
                    m2d = m2 if isinstance(m2, dict) else m2.model_dump() if hasattr(m2, "model_dump") else {}
                    if m2d.get("role") == "tool" and m2d.get("tool_call_id") == tc_id:
                        tool_output = m2d.get("content")
                        break

                normalized_results.append({
                    "type": "tool_call",
                    "tool_name": tc_name,
                    "arguments": tc_args,
                    "output": tool_output,
                })

        elif role == "assistant" and content and content.strip():
            normalized_results.append({
                "type": "output_text",
                "tool_name": None,
                "arguments": None,
                "output": content.strip(),
            })

    normalized_tool_counts = {}
    for name, count in (tool_usage or {}).items():
        normalized_name = "search" if "retrieval" in name.lower() else name
        normalized_tool_counts[normalized_name] = normalized_tool_counts.get(normalized_name, 0) + count

    record = {
        "metadata": {"model": model, "output_dir": str(out_dir)},
        "query_id": query_id,
        "initial_prompt": initial_prompt,
        "tool_call_counts": normalized_tool_counts,
        "status": status,
        "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
        "result": normalized_results,
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(str(out_dir), f"run_{ts}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str, ensure_ascii=False)
    print("Saved response to", filename, "| tool call counts:", normalized_tool_counts)


def _process_tsv_dataset(tsv_path, client, args, tool_handler):
    dataset_path = Path(tsv_path)
    out_dir = Path(args.output_dir).expanduser().resolve()

    lang_map = {}
    if args.lang_map:
        lang_map = json.loads(Path(args.lang_map).read_text(encoding="utf-8"))

    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    # Range selection: --query-start / --query-end (takes priority)
    q_start = getattr(args, "query_start", None)
    q_end = getattr(args, "query_end", None)
    if q_start is not None or q_end is not None:
        q_start = q_start or 0
        q_end = q_end or len(queries)
        queries = queries[q_start:q_end]
        print(f"Query range [{q_start}, {q_end}): {len(queries)} queries")
    # Modulo slicing: --slice-id / --total-slices
    elif getattr(args, "total_slices", None) and args.total_slices > 1:
        slice_id = getattr(args, "slice_id", 0)
        queries = [q for i, q in enumerate(queries) if i % args.total_slices == slice_id]
        print(f"Slice {slice_id}/{args.total_slices}: {len(queries)} queries assigned")

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid = meta.get("query_id")
                    if qid:
                        processed_ids.add(str(qid))
            except Exception:
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]
    if args.max_queries and len(remaining) > args.max_queries:
        remaining = remaining[:args.max_queries]
    print(f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ...")

    completed_lock = threading.Lock()
    completed_count = [0]
    tools = tool_handler.get_tool_definitions()

    def _handle_single_query(qid, qtext, pbar=None):
        language = lang_map.get(qid)
        initial_prompt = format_query(qtext, args.query_template, language=language)
        messages = [{"role": "user", "content": initial_prompt}]

        try:
            messages, tool_usage, status = run_conversation_with_tools(
                client, args.model, messages, tools, tool_handler,
                args.max_iterations, args.verbose, args.reasoning_effort
            )
            if status == "completed":
                with completed_lock:
                    completed_count[0] += 1
                    if pbar:
                        pbar.set_postfix(completed=completed_count[0])

            _persist_response(out_dir, args.model, messages, tool_usage, status, query_id=qid, initial_prompt=initial_prompt)
        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
             tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            futures = [executor.submit(_handle_single_query, qid, qtext, pbar) for qid, qtext in remaining]
            for _ in as_completed(futures):
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Remote GPT-OSS-120B client with local retrieval tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", default="topics-qrels/queries.tsv")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--model-url", default="https://your-gpt-oss-120b-endpoint/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--output-dir", default="runs/crosslingual/gpt_oss_120b_qwen3-8b")
    parser.add_argument("--query-template", choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT", "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION"], default="QUERY_TEMPLATE_NO_GET_DOCUMENT")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--get-document", action="store_true")
    parser.add_argument("--slice-id", type=int, default=0, help="Slice index (0-based)")
    parser.add_argument("--total-slices", type=int, default=1, help="Total number of slices")
    parser.add_argument("--query-start", type=int, default=None, help="Start index of queries (0-based, inclusive)")
    parser.add_argument("--query-end", type=int, default=None, help="End index of queries (exclusive)")
    parser.add_argument("--lang-map", type=str, default=None, help="Path to query language map JSON for multilingual prompt")

    parser.add_argument("--searcher-type", choices=SearcherType.get_choices(), required=True)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--hf-home", type=str)

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.max_queries is None:
        max_q = os.environ.get("MAX_QUERIES")
        if max_q:
            args.max_queries = int(max_q)

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    client = openai.OpenAI(base_url=args.model_url, api_key=args.api_key)
    searcher = searcher_class(args)
    tool_handler = SearchToolHandler(searcher=searcher, snippet_max_tokens=args.snippet_max_tokens, k=args.k, include_get_document=args.get_document)

    if isinstance(args.query, str) and args.query.strip().lower().endswith(".tsv"):
        potential_path = Path(args.query.strip())
        if potential_path.is_file():
            _process_tsv_dataset(str(potential_path), client, args, tool_handler)
            return

    messages = [{"role": "user", "content": format_query(args.query, args.query_template)}]
    tools = tool_handler.get_tool_definitions()
    messages, tool_usage, status = run_conversation_with_tools(
        client, args.model, messages, tools, tool_handler,
        args.max_iterations, args.verbose, args.reasoning_effort
    )
    _persist_response(args.output_dir, args.model, messages, tool_usage, status)
    rprint(messages)


if __name__ == "__main__":
    main()
