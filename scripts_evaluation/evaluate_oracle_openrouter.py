"""
Evaluate Oracle runs using OpenRouter judge. Only computes Accuracy.

Usage:
  python scripts_evaluation/evaluate_oracle_openrouter.py \
      --input_dir runs_latest/runs/original/gpt_oss_20b_oracle \
      --eval_dir evals_new
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import openai
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from search_agent.prompts import GRADER_TEMPLATE_MULTILINGUAL


def load_ground_truth(jsonl_path: Path) -> Dict[str, Dict[str, str]]:
    gt = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            gt[str(obj["query_id"])] = {"question": obj["query"], "answer": obj["answer"]}
    return gt


def create_judge_prompt(question, response, correct_answer):
    return GRADER_TEMPLATE_MULTILINGUAL.format(
        question=question, response=response, correct_answer=correct_answer,
    )


def call_judge(client, prompt, model, max_output_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_output_tokens,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_judge_response(judge_response):
    result = {"extracted_final_answer": None, "reasoning": None, "correct": None,
              "confidence": None, "parse_error": False}
    if not judge_response:
        result["parse_error"] = True
        return result

    for pattern in [r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
                    r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
                    r"extracted_final_answer:\s*(.*?)(?=\n|$)"]:
        m = re.search(pattern, judge_response, re.IGNORECASE | re.DOTALL)
        if m:
            result["extracted_final_answer"] = m.group(1).strip()
            break

    for pattern in [r"\*\*correct:\*\*\s*(yes|no)", r"\*\*correct\*\*:\s*(yes|no)",
                    r"correct:\s*(yes|no)"]:
        m = re.search(pattern, judge_response, re.IGNORECASE)
        if m:
            result["correct"] = m.group(1).lower() == "yes"
            break

    if result["correct"] is None:
        result["parse_error"] = True
    return result


def mirror_directory_structure(input_dir, output_dir):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    parts = input_dir.parts
    runs_index = None
    for i, part in enumerate(parts):
        if part == "runs":
            runs_index = i
            break
    relative = parts[runs_index + 1:] if runs_index is not None else parts[-4:]
    mirrored = output_dir
    for part in relative:
        mirrored = mirrored / part
    mirrored.mkdir(parents=True, exist_ok=True)
    return mirrored


def main():
    parser = argparse.ArgumentParser(description="Evaluate Oracle runs with OpenRouter.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--ground_truth", default="data/original/browsecomp_plus_decrypted.jsonl")
    parser.add_argument("--eval_dir", default="./evals")
    parser.add_argument("--model", default="openai/gpt-5.4")
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    ground_truth = load_ground_truth(Path(args.ground_truth))
    output_dir = mirror_directory_structure(input_dir, Path(args.eval_dir))
    print(f"Evaluations will be saved to {output_dir}")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files in {input_dir}")
        return
    print(f"Found {len(json_files)} Oracle files to evaluate")

    detected_model_name = None
    try:
        first = json.loads(json_files[0].read_text(encoding="utf-8"))
        detected_model_name = (first.get("metadata") or {}).get("model")
    except Exception:
        pass

    all_results = []
    skipped = 0
    _lock = threading.Lock()

    def process_one(json_path):
        eval_path = output_dir / f"{json_path.stem}_eval.json"
        if eval_path.exists() and not args.force:
            try:
                return json.loads(eval_path.read_text(encoding="utf-8")), True
            except:
                pass

        try:
            run_data = json.loads(json_path.read_text(encoding="utf-8"))
        except:
            return None, False

        query_id = run_data.get("query_id")
        if not query_id or str(query_id) not in ground_truth:
            return None, False

        correct_answer = ground_truth[str(query_id)]["answer"]
        gt_question = ground_truth[str(query_id)]["question"]
        response = run_data.get("response", "") or ""

        if not response.strip():
            result = {
                "json_path": str(json_path), "query_id": query_id,
                "response": response, "correct_answer": correct_answer,
                "judge_result": {"parse_error": True, "error": "Empty response"},
            }
            eval_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            return result, False

        judge_prompt = create_judge_prompt(gt_question, response, correct_answer)
        try:
            judge_text = call_judge(client, judge_prompt, args.model, args.max_output_tokens)
            judge_result = parse_judge_response(judge_text)
        except Exception as e:
            return None, False

        result = {
            "json_path": str(json_path), "query_id": query_id,
            "question": gt_question, "response": response,
            "correct_answer": correct_answer,
            "judge_prompt": judge_prompt, "judge_response": judge_text,
            "judge_result": judge_result,
        }
        eval_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result, False

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
         tqdm(total=len(json_files), desc="Evaluating Oracle") as pbar:
        futures = {executor.submit(process_one, jp): jp for jp in json_files}
        for future in as_completed(futures):
            result, was_skipped = future.result()
            if result is not None:
                with _lock:
                    all_results.append(result)
                    if was_skipped:
                        skipped += 1
            pbar.update(1)

    print(f"\nProcessed {len(all_results)} evaluations ({skipped} skipped)")

    total = len(all_results)
    correct_count = sum(1 for r in all_results if r.get("judge_result", {}).get("correct", False))
    accuracy = round(correct_count / total * 100, 2) if total else 0

    summary = {
        "LLM": detected_model_name or "change me",
        "Accuracy (%)": accuracy,
        "Evaluation Date": datetime.now().date().isoformat(),
        "total": total,
        "correct": correct_count,
    }

    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Accuracy: {accuracy}% ({correct_count}/{total})")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
