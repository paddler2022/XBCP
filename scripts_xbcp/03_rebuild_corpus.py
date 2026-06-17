"""
Step 3: Rebuild corpus for both crosslingual and multilingual setups.

Usage:
  python scripts_xbcp/03_rebuild_corpus.py --version crosslingual
  python scripts_xbcp/03_rebuild_corpus.py --version multilingual
  python scripts_xbcp/03_rebuild_corpus.py --version all
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")


def rebuild(dirname):
    from datasets import load_dataset

    out_dir = Path(dirname)
    print(f"\n=== Rebuilding {out_dir} ===")

    # Load translations
    trans = {}
    trans_path = out_dir / "translated_docs.jsonl"
    if trans_path.exists():
        with open(trans_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                trans[d["docid"]] = d["translated_text"]
    print(f"  Translations: {len(trans)}")

    # Load lang assignment
    with open(out_dir / "lang_assignment.json", "r", encoding="utf-8") as f:
        doc_lang = json.load(f)
    evidence_docids = set(doc_lang.keys())

    # Load HF corpus
    print("  Loading HuggingFace corpus...")
    hf_corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
    print(f"  {len(hf_corpus)} docs")

    # Build corpus
    dirname = out_dir.name
    pyserini_path = out_dir / f"{dirname}_corpus.jsonl"
    tevatron_path = out_dir / f"{dirname}_corpus_tevatron.jsonl"
    lang_stats = Counter()

    with open(pyserini_path, "w", encoding="utf-8") as f_p, \
         open(tevatron_path, "w", encoding="utf-8") as f_t:
        for item in hf_corpus:
            docid = str(item["docid"])
            if docid in evidence_docids:
                lang = doc_lang[docid]
                if lang == "English":
                    text = item["text"]
                elif docid in trans and trans[docid]:
                    text = trans[docid]
                else:
                    text = item["text"]
                    lang = "English (missing)"
                lang_stats[lang] += 1
            else:
                text = item["text"]
                lang_stats["English (negative)"] += 1

            f_p.write(json.dumps({"id": docid, "contents": text}, ensure_ascii=False) + "\n")
            f_t.write(json.dumps({"docid": docid, "text": text}, ensure_ascii=False) + "\n")

    # Build evidence_docs_tevatron.jsonl
    ev_tev_path = out_dir / "evidence_docs_tevatron.jsonl"
    with open(out_dir / "evidence_docs.jsonl", "r", encoding="utf-8") as fin, \
         open(ev_tev_path, "w", encoding="utf-8") as fout:
        for line in fin:
            d = json.loads(line)
            docid = d["docid"]
            text = trans.get(docid) or d["text"]
            fout.write(json.dumps({"docid": docid, "text": text}, ensure_ascii=False) + "\n")

    total = sum(lang_stats.values())
    print(f"  Corpus: {total} docs")
    print(f"  Pyserini: {pyserini_path}")
    print(f"  Tevatron: {tevatron_path}")
    print(f"  Evidence Tevatron: {ev_tev_path}")

    with open(out_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump({"total": total, "distribution": dict(lang_stats)}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["crosslingual", "multilingual", "all"], default="all")
    args = parser.parse_args()

    if args.version in ("crosslingual", "all"):
        rebuild("data/crosslingual")
    if args.version in ("multilingual", "all"):
        rebuild("data/multilingual")

    print("\nDone.")


if __name__ == "__main__":
    main()
