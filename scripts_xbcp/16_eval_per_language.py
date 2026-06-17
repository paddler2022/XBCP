"""
Per-language evaluation for multilingual corpus.

Usage:
  python scripts_xbcp/15_eval_per_language.py --eval-dir eval_20260518/multilingual/gpt_oss_20b_qwen3-8b
  python scripts_xbcp/15_eval_per_language.py --eval-dir eval_20260518/multilingual/gpt_oss_20b_agentir_qwen3-8b
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def calib_err(confidence, correct, beta=100):
    if len(confidence) < 2:
        return float("nan")
    beta = min(beta, max(len(confidence) // 3, 2))
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if not bins:
        return float("nan")
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins)):
        bin_confidence = confidence[bins[i][0]: bins[i][1]]
        bin_correct = correct[bins[i][0]: bins[i][1]]
        num_examples_in_bin = len(bin_confidence)
        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            cerr += num_examples_in_bin / total_examples * (difference ** 2)
    return np.sqrt(cerr) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="e.g. eval_20260518/multilingual/gpt_oss_20b_qwen3-8b")
    parser.add_argument("--lang-map", default="data/crosslingual/query_lang_assignment.json")
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    args = parser.parse_args()

    lang_map = json.loads(Path(args.lang_map).read_text(encoding="utf-8"))
    eval_dir = Path(args.eval_dir)

    per_lang = defaultdict(list)

    for f in sorted(eval_dir.glob("*_eval.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        qid = str(d["query_id"])
        lang = lang_map.get(qid, "Unknown")

        jr = d.get("judge_result", {})
        if jr.get("parse_error"):
            correct = False
        else:
            c = jr.get("correct", False)
            correct = (c is True or (isinstance(c, str) and c.lower() == "yes"))
        raw_conf = jr.get("confidence")
        try:
            confidence = float(str(raw_conf).replace("%", "")) if raw_conf is not None else None
        except (ValueError, TypeError):
            confidence = None
        recall = d.get("retrieval", {}).get("recall", 0.0)
        search = d.get("tool_call_counts", {}).get("search", 0)

        per_lang[lang].append({
            "correct": correct,
            "confidence": confidence,
            "recall": recall,
            "search": search,
        })

    is_oracle = "oracle" in str(eval_dir).lower()

    rows = []
    for lang in sorted(per_lang.keys()):
        items = per_lang[lang]
        n = len(items)
        acc = sum(1 for x in items if x["correct"]) / n * 100
        recall = sum(x["recall"] for x in items) / n * 100
        avg_search = sum(x["search"] for x in items) / n

        confs = [float(x["confidence"]) for x in items if x["confidence"] is not None]
        cors = [bool(x["correct"]) for x in items if x["confidence"] is not None]
        if len(confs) >= 6:
            ce = calib_err(np.array(confs, dtype=float) / 100.0, np.array(cors, dtype=float))
        else:
            ce = float("nan")

        rows.append((lang, n, acc, recall, ce, avg_search))

    # Total
    all_items = [x for items in per_lang.values() for x in items]
    n = len(all_items)
    acc = sum(1 for x in all_items if x["correct"]) / n * 100
    recall = sum(x["recall"] for x in all_items) / n * 100
    avg_search = sum(x["search"] for x in all_items) / n
    confs = [float(x["confidence"]) for x in all_items if x["confidence"] is not None]
    cors = [bool(x["correct"]) for x in all_items if x["confidence"] is not None]
    ce = calib_err(np.array(confs, dtype=float) / 100.0, np.array(cors, dtype=float)) if len(confs) >= 6 else float("nan")
    rows.append(("**Total**", n, acc, recall, ce, avg_search))

    if is_oracle:
        if args.format == "csv":
            print("Language,N,Accuracy (%)")
            for lang, n, acc, recall, ce, avg_s in rows:
                print(f"{lang},{n},{acc:.2f}")
        else:
            print(f"\nEval dir: {args.eval_dir}\n")
            print(f"| Language | N | Accuracy (%) |")
            print(f"|----------|--:|:------------:|")
            for lang, n, acc, recall, ce, avg_s in rows:
                print(f"| {lang} | {n} | {acc:.2f} |")
    else:
        if args.format == "csv":
            print("Language,N,Accuracy (%),Recall (%),Calibration Error (%),Avg Search Calls")
            for lang, n, acc, recall, ce, avg_s in rows:
                print(f"{lang},{n},{acc:.2f},{recall:.2f},{ce:.2f},{avg_s:.2f}")
        else:
            print(f"\nEval dir: {args.eval_dir}\n")
            print(f"| Language | N | Accuracy (%) | Recall (%) | Calibration Error (%) | Avg Search Calls |")
            print(f"|----------|--:|:------------:|:----------:|:---------------------:|:----------------:|")
            for lang, n, acc, recall, ce, avg_s in rows:
                ce_str = f"{ce:.2f}" if not np.isnan(ce) else "N/A"
                print(f"| {lang} | {n} | {acc:.2f} | {recall:.2f} | {ce_str} | {avg_s:.2f} |")


if __name__ == "__main__":
    main()
