"""
Translate queries.tsv into target languages based on multilingual query_lang_assignment.json.
English queries are kept as-is. Uses Azure OpenAI (GPT-5.4).

Usage:
  # Translate all
  python scripts_xbcp/14_translate_queries.py --workers 10

  # Translate a range
  python scripts_xbcp/14_translate_queries.py --workers 10 --start 0 --end 100

  # Dry run (show what would be translated)
  python scripts_xbcp/14_translate_queries.py --dry-run

Output: data/crosslingual/queries_translated.tsv (same format: qid<TAB>query)
"""

import argparse
import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-5.4"

_write_lock = threading.Lock()
_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=OPENROUTER_BASE,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


def translate_query(query: str, target_lang: str) -> str:
    client = get_client()

    messages = [
        {"role": "system", "content": (
            f"Translate the following question into {target_lang}. "
            f"Translate everything including proper nouns and titles according to {target_lang} conventions. "
            "Output only the translated question. Do not add explanations."
        )},
        {"role": "user", "content": query},
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=4096,
        temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="topics-qrels/queries.tsv")
    parser.add_argument("--lang-map", default="data/crosslingual/query_lang_assignment.json")
    parser.add_argument("--output", default="data/crosslingual/queries_translated.tsv")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lang_map = json.loads(Path(args.lang_map).read_text(encoding="utf-8"))

    queries = []
    with open(args.queries, newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                queries.append((row[0].strip(), row[1].strip()))

    # Apply range
    if args.start is not None or args.end is not None:
        queries = queries[args.start:args.end]

    # Load already-translated
    output_path = Path(args.output)
    translated = {}
    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    translated[row[0].strip()] = row[1].strip()

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in translated]
    print(f"Total: {len(queries)}, already translated: {len(translated)}, remaining: {len(remaining)}")

    if args.dry_run:
        from collections import Counter
        langs = Counter(lang_map.get(qid, "Unknown") for qid, _ in remaining)
        for lang, count in sorted(langs.items()):
            print(f"  {lang}: {count}")
        return

    def handle(qid, qtext):
        lang = lang_map.get(qid, "English")
        if lang == "English":
            return qid, qtext
        try:
            result = translate_query(qtext, lang)
            return qid, result
        except Exception as e:
            print(f"[Error] qid={qid} lang={lang}: {e}")
            return qid, None

    with ThreadPoolExecutor(max_workers=args.workers) as executor, \
         tqdm(total=len(remaining), desc="Translating") as pbar:
        futures = {executor.submit(handle, qid, qtext): qid for qid, qtext in remaining}
        for future in as_completed(futures):
            qid, result = future.result()
            if result is not None:
                with _write_lock:
                    translated[qid] = result
                    with output_path.open("a", encoding="utf-8") as f:
                        f.write(f"{qid}\t{result}\n")
            pbar.update(1)

    print(f"Done. {len(translated)} queries saved to {args.output}")


if __name__ == "__main__":
    main()
