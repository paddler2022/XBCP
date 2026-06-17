"""
Step 1: Prepare evidence docs and language assignments for XBCP.

Extracts the 5,040 evidence documents from the BrowseComp-Plus corpus
and assigns target translation languages under two schemes:
  - crosslingual: all evidence docs for a query share the same language (per-query-group)
  - multilingual: each evidence doc independently assigned a language (per-document, uniform)

Outputs:
  data/crosslingual/evidence_docs.jsonl
  data/crosslingual/lang_assignment.json
  data/crosslingual/query_lang_assignment.json
  data/multilingual/evidence_docs.jsonl
  data/multilingual/lang_assignment.json

Usage:
  python scripts_xbcp/01_prepare_data.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

LANGUAGES = [
    "Chinese", "English", "French", "German", "Japanese", "Korean",
    "Portuguese", "Spanish", "Swahili", "Wolof", "Yoruba", "Zulu",
]

QREL_PATH = "topics-qrels/qrel_evidence.txt"


def load_qrels(path):
    """Load qrels: query_id -> set of evidence docids."""
    qrels = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels[parts[0]].add(parts[2])
    return dict(qrels)


def extract_evidence_docs(qrels):
    """Extract evidence documents from BrowseComp-Plus HuggingFace corpus."""
    evidence_docids = set()
    for docids in qrels.values():
        evidence_docids.update(docids)
    print(f"Evidence docids: {len(evidence_docids)}")

    print("Loading BrowseComp-Plus corpus from HuggingFace...")
    corpus = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")

    docs = {}
    for item in corpus:
        docid = str(item["docid"])
        if docid in evidence_docids:
            docs[docid] = {
                "docid": docid,
                "text": item["text"],
                "url": item.get("url", ""),
                "char_count": len(item["text"]),
            }
    print(f"Extracted {len(docs)}/{len(evidence_docids)} evidence docs")
    return docs


def assign_crosslingual(qrels, docs):
    """Per-query-group: all evidence docs for a query share one language."""
    query_ids = sorted(qrels.keys())
    random.seed(42)
    random.shuffle(query_ids)

    query_lang = {}
    for i, qid in enumerate(query_ids):
        query_lang[qid] = LANGUAGES[i % len(LANGUAGES)]

    doc_lang = {}
    for qid, docids in qrels.items():
        lang = query_lang[qid]
        for docid in docids:
            if docid in docs:
                doc_lang[docid] = lang

    return doc_lang, query_lang


def assign_multilingual(docs):
    """Per-document: each doc independently assigned, uniform across languages."""
    docids = sorted(docs.keys())
    random.seed(42)
    random.shuffle(docids)

    doc_lang = {}
    for i, docid in enumerate(docids):
        doc_lang[docid] = LANGUAGES[i % len(LANGUAGES)]

    return doc_lang


def write_output(out_dir, docs, doc_lang, query_lang=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "evidence_docs.jsonl", "w", encoding="utf-8") as f:
        for docid in sorted(docs.keys()):
            doc = dict(docs[docid])
            doc["language"] = doc_lang.get(docid, "English")
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    with open(out_dir / "lang_assignment.json", "w", encoding="utf-8") as f:
        json.dump(doc_lang, f, ensure_ascii=False, indent=2)

    if query_lang:
        with open(out_dir / "query_lang_assignment.json", "w", encoding="utf-8") as f:
            json.dump(query_lang, f, ensure_ascii=False, indent=2)

    from collections import Counter
    c = Counter(doc_lang.values())
    print(f"  {out_dir}: {len(doc_lang)} docs, {len(c)} languages")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    qrels = load_qrels(QREL_PATH)
    print(f"Queries with evidence: {len(qrels)}")

    docs = extract_evidence_docs(qrels)

    print("\n=== Crosslingual (per-query-group) ===")
    cl_doc_lang, cl_query_lang = assign_crosslingual(qrels, docs)
    write_output("data/crosslingual", docs, cl_doc_lang, cl_query_lang)

    print("\n=== Multilingual (per-document, uniform) ===")
    ml_doc_lang = assign_multilingual(docs)
    write_output("data/multilingual", docs, ml_doc_lang)

    print("\nDone. Ready for 02_batch_translate.py")


if __name__ == "__main__":
    main()
