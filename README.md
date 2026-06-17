# XBCP: Cross-lingual BrowseComp-Plus

| [Paper](https://arxiv.org/abs/2606.15345v1) | [Dataset](https://huggingface.co/datasets/UTokyo-Yokoya-Lab/XBCP) |

XBCP extends [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) to cross-lingual deep research evaluation. Questions remain in English; evidence documents are translated into 12 languages spanning high-resource (Chinese, French, German, Japanese, Korean, Portuguese, Spanish) and low-resource (Swahili, Wolof, Yoruba, Zulu) regimes, plus English as an untranslated reference.

Two evaluation settings are provided:
- **Cross-lingual**: all evidence documents for a given query appear in the same non-English language.
- **Multilingual**: evidence documents are randomly and uniformly assigned across 12 languages.

This repo contains the complete pipeline to reproduce all experiments reported in the paper.

---

## Setup

### 1. Clone this repo and install dependencies

```bash
git clone https://github.com/YOUR_ORG/XBCP.git
cd XBCP
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn
```

Java 21 is required for BM25 indexing via Pyserini:

```bash
conda install -c conda-forge openjdk=21
```

### 2. Download BrowseComp-Plus base data

XBCP builds on top of [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus). Download the original queries, answers, and relevance judgments:

```bash
pip install datasets
python scripts_build_index/decrypt_dataset.py \
    --output data/original/browsecomp_plus_decrypted.jsonl \
    --generate-tsv topics-qrels/queries.tsv
```

> You may need to log in with `huggingface-cli login` beforehand.

### 3. Download XBCP data and indexes from HuggingFace

```bash
# Download translated corpora
hf download UTokyo-Yokoya-Lab/XBCP data --repo-type dataset --local-dir .

# Download pre-built indexes (all retrievers x 3 corpora)
hf download UTokyo-Yokoya-Lab/XBCP indexes --repo-type dataset --local-dir .
```

After setup, your directory should look like:

```
XBCP/
├── data/
│   ├── original/browsecomp_plus_decrypted.jsonl   # from BrowseComp-Plus
│   ├── crosslingual/                               # from HuggingFace
│   │   ├── crosslingual_corpus.jsonl
│   │   ├── crosslingual_corpus_tevatron.jsonl
│   │   ├── lang_assignment.json
│   │   ├── query_lang_assignment.json
│   │   └── queries_translated.tsv
│   └── multilingual/                                # from HuggingFace
│       ├── multilingual_corpus.jsonl
│       ├── multilingual_corpus_tevatron.jsonl
│       └── lang_assignment.json
├── indexes/
│   ├── original/{bm25,qwen3-embedding-8b,...}/      # from HuggingFace
│   ├── crosslingual/{bm25,qwen3-embedding-8b,...}/
│   └── multilingual/{bm25,qwen3-embedding-8b,...}/
└── topics-qrels/                                    # from BrowseComp-Plus
    ├── queries.tsv
    ├── qrel_evidence.txt
    └── qrel_golds.txt
```

---

## Running Experiments

### Main experiments

End-to-end agent evaluation across three corpus conditions (original, cross-lingual, multilingual):

```bash
bash scripts_xbcp/05_run_gpt_oss_20b.sh          # GPT-OSS-20B x 5 retrievers
bash scripts_xbcp/06_run_qwen36.sh                # Qwen3.6-35B-A3B x 5 retrievers
bash scripts_xbcp/09_run_gpt_oss_120b.sh          # GPT-OSS-120B x 5 retrievers
bash scripts_xbcp/08_run_deepseek.sh              # DeepSeek-V4-Pro x BM25 + Qwen3-8B
bash scripts_xbcp/10_run_tongyi.sh                # Tongyi-30B-A3B x BM25 + Qwen3-8B
```

### Oracle experiments

Gold evidence documents are provided directly in the prompt, bypassing retrieval:

```bash
bash scripts_xbcp/07_run_oracle.sh                # GPT-OSS-20B + Qwen3.6 + GPT-OSS-120B
bash scripts_xbcp/14_run_oracle_tq_tp.sh          # Oracle tq+tp (target-language prompt)
```

### Supplementary experiments

```bash
bash scripts_xbcp/11_run_gpt_oss_agentir.sh       # AgentIR query expansion
bash scripts_xbcp/12_run_gpt_oss_reasoning_effort.sh  # Reasoning effort: low + high
bash scripts_xbcp/13_run_tq_tp.sh                 # Translated query + translated prompt (with retrieval)
```

### Translate queries (for tq+tp experiments)

```bash
python scripts_xbcp/15_translate_queries.py
```

---

## Evaluation

Evaluation uses GPT-5.4 as an LLM-as-Judge via OpenRouter:

```bash
# Evaluate all agent runs
bash scripts_evaluation/batch_eval_openrouter.sh

# Evaluate oracle runs
bash scripts_evaluation/batch_eval_oracle_openrouter.sh
```

### Per-language analysis

```bash
python scripts_xbcp/16_eval_per_language.py --eval-dir evals/crosslingual/gpt_oss_20b_qwen3-8b
```

### Evaluating your own agent

Format your results into a directory under `runs/` with one JSON file per query:

```json
{
    "query_id": "311",
    "tool_call_counts": {"search": 12},
    "status": "completed",
    "retrieved_docids": ["31173", "95218", "..."],
    "result": [{"type": "output_text", "output": "The answer is ..."}]
}
```

Then evaluate:

```bash
python scripts_evaluation/evaluate_with_openrouter.py --input_dir runs/crosslingual/my_agent --eval_dir evals/crosslingual/my_agent
```

---

## Processes of Building XBCP 

The following scripts are used by us to build XBCP from scratch, they can serve as a reference for the building processes.

### Step 1: Prepare evidence docs and language assignments

```bash
python scripts_xbcp/01_prepare_data.py
```

### Step 2: Translate evidence documents

```bash
python scripts_xbcp/02_batch_translate.py run --input data/crosslingual/evidence_docs.jsonl --workers 16
python scripts_xbcp/02_batch_translate.py run --input data/multilingual/evidence_docs.jsonl --workers 16
```

### Step 3: Rebuild corpus

```bash
python scripts_xbcp/03_rebuild_corpus.py --version all
```

### Step 4: Build retrieval indexes

```bash
bash scripts_xbcp/04_build_indexes.sh crosslingual all
bash scripts_xbcp/04_build_indexes.sh multilingual all
```

---

## Directory Structure

```
XBCP/
├── scripts_build_index/     # BrowseComp-Plus: dataset download
├── scripts_xbcp/            # XBCP pipeline (01-16)
│   ├── 01_prepare_data.py           # Extract evidence docs + assign languages
│   ├── 02_batch_translate.py        # Translate with GPT-5.4
│   ├── 03_rebuild_corpus.py         # Merge translations into full corpus
│   ├── 04_build_indexes.sh          # Build BM25 + dense indexes
│   ├── 05-10                        # Run agents (main experiments)
│   ├── 07_run_oracle.{sh,py}        # Oracle experiments
│   ├── 11-12                        # Supplementary experiments
│   ├── 13_run_tq_tp.sh             # Translated query + prompt (with retrieval)
│   ├── 14_run_oracle_tq_tp.sh       # Oracle tq+tp experiment
│   ├── 15_translate_queries.py      # Translate queries for tq+tp
│   └── 16_eval_per_language.py      # Per-language analysis
├── scripts_evaluation/      # LLM-as-Judge evaluation
├── search_agent/            # Agent client implementations
├── searcher/                # Retriever implementations (BM25, FAISS)
├── data/                    # Corpora (from HuggingFace + BrowseComp-Plus)
├── indexes/                 # Retrieval indexes (from HuggingFace)
├── topics-qrels/            # Queries and relevance judgments (from BrowseComp-Plus)
├── runs/                    # Agent run outputs
└── evals/                   # Evaluation results
```

## Agents

| Agent | Client Script | Serving |
|-------|--------------|---------|
| GPT-OSS-20B | `search_agent/gpt_oss_client.py` | Local vLLM |
| GPT-OSS-120B | `search_agent/gpt_oss_remote_client.py` | Remote vLLM endpoint |
| Qwen3.6-35B-A3B | `search_agent/openai_remote_client.py` | Remote vLLM endpoint |
| DeepSeek-V4-Pro | `search_agent/deepseek_remote_client.py` | DeepSeek API |
| Tongyi-30B-A3B | `search_agent/tongyi_client.py` | Remote vLLM endpoint |

## Retrievers

| Retriever | Index Type |
|-----------|-----------|
| BM25 | Pyserini Lucene |
| Qwen3-Embedding-8B | FAISS (Tevatron) |
| Qwen3-Embedding-4B | FAISS (Tevatron) |
| Multilingual-E5-Large | FAISS (Tevatron) |
| Arctic-Embed-L-v2.0 | FAISS (Tevatron) |
