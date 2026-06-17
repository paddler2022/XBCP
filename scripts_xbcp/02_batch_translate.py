"""
Step 2: Translate evidence docs using Azure OpenAI (concurrent real-time API).
New prompt: translate everything including proper nouns.

Usage:
  # V1 crosslingual
  python scripts_xbcp/02_batch_translate.py run --input data/crosslingual/evidence_docs.jsonl --workers 10
  python scripts_xbcp/02_batch_translate.py run --input data/crosslingual/evidence_docs.jsonl --workers 10 --start 0 --end 1000

  # V2 multilingual
  python scripts_xbcp/02_batch_translate.py run --input data/multilingual/evidence_docs.jsonl --workers 10

  # Merge + Stats
  python scripts_xbcp/02_batch_translate.py --output-dir data/crosslingual merge
  python scripts_xbcp/02_batch_translate.py --output-dir data/multilingual merge
  python scripts_xbcp/02_batch_translate.py --output-dir data/crosslingual stats
"""

import json
import os
import sys
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from openai import AzureOpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

AZURE_ENDPOINT = "https://your-azure-openai-endpoint/"
DEPLOYMENT_FULL = "gpt-5.4"
DEPLOYMENT_MINI = "gpt-5.4"
API_VERSION = "2025-04-01-preview"

LOW_RESOURCE_LANGS = {"Swahili", "Wolof", "Yoruba", "Zulu"}

DEFAULT_WORKERS = 10
INPUT_FILE = Path("data/crosslingual/evidence_docs.jsonl")
RESULT_DIR = Path("data/crosslingual")
MERGED_FILE = RESULT_DIR / "translated_docs.jsonl"
STATS_FILE = RESULT_DIR / "token_stats.json"
ERROR_FILE = RESULT_DIR / "translation_errors.jsonl"

_write_lock = threading.Lock()
_client = None


def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=API_VERSION,
    )


def _get_shared_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = get_client()
    return _client


def build_translation_prompt(text: str, target_lang: str) -> list[dict]:
    system_prompt = (
        f"Translate the following document completely into {target_lang}. "
        f"Translate everything including proper nouns, titles, terminology, "
        f"and metadata field names according to {target_lang} conventions. "
        f"For example, 'name:' should become the equivalent in {target_lang}, 'birth_date:' should become the equivalent in {target_lang}, etc.\n\n"
        "Rules:\n"
        f"1. Ensuring cultural appropriateness for {target_lang} speakers.\n"
        f"2. If works such as books, movies, TV shows, songs, or other literary/entertainment titles have well-known translations in {target_lang}, use those established translations.\n"
        "3. Preserve all URLs, email addresses, math formulas, and code blocks unchanged.\n"
        "4. Output only the translated document. Do not add explanations."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def _load_all_non_english(input_path: str) -> list[dict]:
    docs = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["language"] == "English":
                continue
            docs.append(doc)
    return docs


def _load_done_docids_from_file(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["docid"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def _get_range_file(start: int, end: int) -> Path:
    return RESULT_DIR / f"translated_{start:04d}_{end:04d}.jsonl"


def _get_deployment(lang: str) -> str:
    return DEPLOYMENT_FULL if lang in LOW_RESOURCE_LANGS else DEPLOYMENT_MINI


CHUNK_SIZE = 0
def _get_chunk_dir(docid) -> Path:
    return RESULT_DIR / "tmp" / str(docid)


def _get_chunk_path(docid, chunk_idx: int) -> Path:
    return _get_chunk_dir(docid) / f"chunk_{chunk_idx:04d}.txt"


def _split_by_newline(text: str, chunk_size: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        if start + chunk_size >= len(text):
            chunks.append(text[start:])
            break
        # Find last newline within chunk_size range
        end = text.rfind("\n", start, start + chunk_size)
        if end == -1 or end == start:
            end = start + chunk_size
        else:
            end += 1  # include the newline
        chunks.append(text[start:end])
        start = end
    return chunks


def _sanitize_for_filter(text: str) -> str:
    import re
    replacements = [
        (r'\bfucking\b', 'freaking'), (r'\bfucked\b', 'messed'), (r'\bfuck\b', 'darn'),
        (r'\bbloody\b', 'very'), (r'\bshit\b', 'stuff'), (r'\bbastard\b', 'rascal'),
        (r'\bbugger\b', 'fellow'), (r'\bwhore\b', 'woman'), (r'\barse\b', 'rear'),
        (r'\bpiss\b', 'go'), (r'\bcrap\b', 'stuff'), (r'\bsod\b', 'fellow'),
        (r'\bnaked\b', 'bare'), (r'\bdamn\b', 'darn'),
    ]
    for pat, repl in replacements:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def _translate_chunk(client, deployment: str, text: str, target_lang: str, max_retries: int = 3) -> tuple[str, dict]:
    import time as _t
    messages = build_translation_prompt(text, target_lang)
    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(
                model=deployment,
                messages=messages,
                max_completion_tokens=128000,
                reasoning_effort="none",
                stream=True,
                stream_options={"include_usage": True},
            )
            parts = []
            usage_data = None
            for chunk in stream:
                if chunk.usage:
                    usage_data = chunk.usage
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
            translated = "".join(parts)
            usage = {
                "prompt_tokens": usage_data.prompt_tokens if usage_data else 0,
                "completion_tokens": usage_data.completion_tokens if usage_data else 0,
                "total_tokens": usage_data.total_tokens if usage_data else 0,
            }
            return translated, usage
        except Exception as e:
            if "content_filter" in str(e):
                raise
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"      retry {attempt+1}/{max_retries} after {wait}s: {e}", flush=True)
                _t.sleep(wait)
            else:
                raise


def _translate_one(doc: dict, output_file: Path) -> dict | None:
    import time as _time
    deployment = _get_deployment(doc["language"])
    text = doc["text"]
    char_count = doc["char_count"]
    need_chunk = char_count > CHUNK_SIZE and CHUNK_SIZE > 0

    num_chunks = len(_split_by_newline(text, CHUNK_SIZE)) if need_chunk else 1
    print(f"\n  >> Start docid={doc['docid']} lang={doc['language']} model={deployment} chars={char_count} chunks={num_chunks}", flush=True)

    t0 = _time.time()
    client = _get_shared_client()

    translated_parts = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        if not need_chunk:
            translated_text, usage = _translate_chunk(client, deployment, text, doc["language"])
            translated_parts.append(translated_text)
            total_usage = usage
        else:
            text_chunks = _split_by_newline(text, CHUNK_SIZE)
            num_chunks = len(text_chunks)
            doc_tmp_dir = _get_chunk_dir(doc["docid"])
            doc_tmp_dir.mkdir(parents=True, exist_ok=True)

            for ci, chunk in enumerate(text_chunks):
                chunk_path = _get_chunk_path(doc["docid"], ci)
                if chunk_path.exists():
                    part = chunk_path.read_text(encoding="utf-8")
                    print(f"    chunk {ci+1}/{num_chunks} (cached, {len(part)} chars)", flush=True)
                else:
                    print(f"    chunk {ci+1}/{num_chunks} ({len(chunk)} chars)", flush=True)
                    part, usage = _translate_chunk(client, deployment, chunk, doc["language"])
                    chunk_path.write_text(part, encoding="utf-8")
                    for k in total_usage:
                        total_usage[k] += usage[k]
                translated_parts.append(part)
    except Exception as e:
        error_record = {
            "docid": doc["docid"],
            "language": doc["language"],
            "error": str(e),
            "chunks_done": len(translated_parts),
            "chunks_total": num_chunks,
        }
        with _write_lock:
            with open(ERROR_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_record, ensure_ascii=False) + "\n")
        return None

    translated_text = "".join(translated_parts)

    elapsed = _time.time() - t0
    print(f"  << Done  docid={doc['docid']} in {elapsed:.1f}s tokens={total_usage['total_tokens']} chunks={num_chunks}", flush=True)

    record = {
        "docid": doc["docid"],
        "language": doc["language"],
        "translated_text": translated_text,
        "usage": total_usage,
    }

    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


# ── run ───────────────────────────────────────────────────────────

def run(workers: int, input_path: str, start: int | None = None, end: int | None = None, max_chars: int | None = None):
    all_docs = _load_all_non_english(input_path)
    total = len(all_docs)

    if start is None:
        start = 0
    if end is None:
        end = total

    start = max(0, min(start, total))
    end = max(start, min(end, total))

    range_docs = all_docs[start:end]
    output_file = _get_range_file(start, end)

    already_done = _load_done_docids_from_file(output_file)
    remaining = [d for d in range_docs if d["docid"] not in already_done]

    if max_chars is not None:
        skipped = [d for d in remaining if d["char_count"] > max_chars]
        remaining = [d for d in remaining if d["char_count"] <= max_chars]
        print(f"Skipped (>{max_chars:,} chars): {len(skipped)}")

    print(f"Total non-English docs: {total}")
    print(f"Range:                  [{start}, {end})")
    print(f"Docs in range:          {len(range_docs)}")
    print(f"Already translated:     {len(already_done)}")
    print(f"This run:               {len(remaining)}")
    print(f"Output file:            {output_file}")
    print(f"Workers:                {workers}")

    if not remaining:
        print("Nothing to do.")
        return

    succeeded = 0
    failed = 0

    if tqdm:
        pbar = tqdm(total=len(remaining), desc=f"[{start}-{end}]", unit="doc")
    else:
        pbar = None

    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {pool.submit(_translate_one, doc, output_file): doc for doc in remaining}
    try:
        while pending:
            done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                if result:
                    succeeded += 1
                    if pbar:
                        pbar.set_postfix(ok=succeeded, fail=failed)
                else:
                    failed += 1
                if pbar:
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Already-written translations are saved.")
        if pbar:
            pbar.close()
        os._exit(0)

    if pbar:
        pbar.close()

    print(f"\nDone. Succeeded: {succeeded}, Failed: {failed}")
    if failed > 0:
        print(f"Failed docs logged in {ERROR_FILE}")
    print(f"Results in {output_file}")


# ── merge ─────────────────────────────────────────────────────────

def merge():
    range_files = sorted(RESULT_DIR.glob("translated_[0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9].jsonl"))

    if not range_files:
        print("No range files found.")
        return

    seen = set()
    total = 0

    with open(MERGED_FILE, "w", encoding="utf-8") as f_out:
        for rf in range_files:
            count = 0
            with open(rf, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    docid = record.get("docid")
                    if docid and docid not in seen:
                        seen.add(docid)
                        f_out.write(line)
                        count += 1
            print(f"  {rf.name}: {count} docs")
            total += count

    print(f"\nMerged {total} docs from {len(range_files)} files into {MERGED_FILE}")


# ── stats ─────────────────────────────────────────────────────────

def stats():
    if not MERGED_FILE.exists():
        print(f"{MERGED_FILE} not found. Run 'merge' first.")
        return

    total_input = total_output = total_total = count = 0
    by_lang = {}

    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            usage = record.get("usage", {})
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
            tot = usage.get("total_tokens", inp + out)
            total_input += inp
            total_output += out
            total_total += tot
            count += 1
            lang = record.get("language", "unknown")
            if lang not in by_lang:
                by_lang[lang] = {"count": 0, "input": 0, "output": 0}
            by_lang[lang]["count"] += 1
            by_lang[lang]["input"] += inp
            by_lang[lang]["output"] += out

    if count == 0:
        print("No records found.")
        return

    avg_input = total_input // count
    avg_output = total_output // count
    PRICE_INPUT = 2.50
    PRICE_OUTPUT = 15.00
    total_cost = (total_input * PRICE_INPUT + total_output * PRICE_OUTPUT) / 1_000_000

    print(f"Translated docs:      {count}")
    print(f"Total input tokens:   {total_input:,}")
    print(f"Total output tokens:  {total_output:,}")
    print(f"Avg per doc:          {avg_input:,} in / {avg_output:,} out")
    print(f"Total cost:           ${total_cost:.2f}")

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump({"translated_docs": count, "total_input_tokens": total_input,
                    "total_output_tokens": total_output, "total_cost_usd": round(total_cost, 2)}, f, indent=2)
    print(f"\nSaved to {STATS_FILE}")


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_run.add_argument("--input", default=str(INPUT_FILE))
    p_run.add_argument("--start", type=int, default=None)
    p_run.add_argument("--end", type=int, default=None)
    p_run.add_argument("--max-chars", type=int, default=None)
    p_run.add_argument("--no-chunk", action="store_true", help="Disable chunking, send full doc in one API call")

    sub.add_parser("merge")
    sub.add_parser("stats")

    args = parser.parse_args()

    global RESULT_DIR, MERGED_FILE, STATS_FILE, ERROR_FILE
    if args.output_dir:
        RESULT_DIR = Path(args.output_dir)
    elif hasattr(args, "input") and args.input != str(INPUT_FILE):
        RESULT_DIR = Path(args.input).parent
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_FILE = RESULT_DIR / "translated_docs.jsonl"
    STATS_FILE = RESULT_DIR / "token_stats.json"
    ERROR_FILE = RESULT_DIR / "translation_errors.jsonl"

    if args.command == "run":
        if hasattr(args, "no_chunk") and args.no_chunk:
            global CHUNK_SIZE
            CHUNK_SIZE = 0
        run(args.workers, args.input, args.start, args.end, args.max_chars)
    elif args.command == "merge":
        merge()
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
