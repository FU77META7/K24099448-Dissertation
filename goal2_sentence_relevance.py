
# goal2_sentence_relevance.py
'''
Implementation for Goal 2 (Sentence-level semantic relevance scoring).

- Loads either:
    (A) a raw dataset JSON (same schema as Goal 1) and segments answers on the fly, or
    (B) the JSONL produced by Goal 1 with pre-segmented sentences.
- Computes cosine similarities between each sentence and the ground_truth string using Sentence-Transformers.
- Classifies each sentence as Relevant/Irrelevant using a configurable threshold (default 0.60).
- Aggregates per-answer metrics and writes both a JSONL (per sentence) and CSV summaries.

Usage (CLI, with pre-segmented sentences from Goal 1):
    python goal2_sentence_relevance.py \
        --groundtruth_from_dataset /path/to/reformatted_gpt_o1_responses_with_labels.json \
        --segmented_jsonl ./segmented_sentences.jsonl \
        --out_scores_jsonl ./sentence_scores.jsonl \
        --out_answer_summary_csv ./answer_summary.csv

Or do end-to-end from raw dataset (it will segment internally):
    python goal2_sentence_relevance.py \
        --dataset /path/to/reformatted_gpt_o1_responses_with_labels.json \
        --out_scores_jsonl ./sentence_scores.jsonl \
        --out_answer_summary_csv ./answer_summary.csv
'''
import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# ---- Sentence segmentation (same as Goal 1, inlined for simplicity) ----
import re
try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception as e:
    raise SystemExit("Please install NLTK: pip install nltk") from e

def ensure_nltk_data() -> None:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

BULLET_PATTERN = re.compile(r"^\s*[\-\u2022\*\(\[]?[a-zA-Z0-9]?\)\s*")
MULTISPACE = re.compile(r"\s+")

def _normalize_space(text: str) -> str:
    return MULTISPACE.sub(" ", text).strip()

def _split_on_semicolons(sent: str) -> List[str]:
    parts = [p.strip() for p in sent.split(";")]
    out = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(p) < 35:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append((buf + " " + p).strip())
                buf = ""
            else:
                out.append(p)
    if buf:
        out.append(buf)
    return out

def _split_on_bullets(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    chunks = []
    buf = []
    for ln in lines:
        if BULLET_PATTERN.match(ln):
            if buf:
                chunks.append(" ".join(buf).strip())
                buf = []
            chunks.append(ln)
        else:
            buf.append(ln)
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]

def segment_answer(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    chunks = _split_on_bullets(text)
    if not chunks:
        chunks = [text]
    out_sents: List[str] = []
    for ch in chunks:
        for s in sent_tokenize(ch):
            s = _normalize_space(s)
            if not s:
                continue
            semi_splits = _split_on_semicolons(s) if ";" in s else [s]
            for ss in semi_splits:
                ss = _normalize_space(ss)
                if ss and len(ss) > 0:
                    out_sents.append(ss)
    deduped = []
    for s in out_sents:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    return deduped

# ---- Embeddings ----
def _backoff_sleep(attempt: int) -> None:
    # Exponential backoff up to ~32s
    time.sleep(min(2 ** attempt, 32))

def _load_model(model_name: str):
    # Isolated import to give clear errors if sentence-transformers isn't installed
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise SystemExit(
            "Please install sentence-transformers: pip install sentence-transformers"
        ) from e
    # Try to load with naive retry to dodge 429s on first pull
    last_err = None
    for attempt in range(6):
        try:
            return SentenceTransformer(model_name)
        except Exception as e:
            last_err = e
            _backoff_sleep(attempt)
    # final try with common alternative
    try:
        return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    except Exception as e2:
        raise SystemExit(f"Failed to load models due to: {last_err} / {e2}")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ---- Core logic ----
def from_dataset_compute(dataset_path: Path, out_scores_jsonl: Path, out_answer_summary_csv: Path,
                         model_name: str, threshold: float) -> None:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    ensure_nltk_data()
    model = _load_model(model_name)

    out_scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_answer_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_scores_jsonl.open("w", encoding="utf-8") as jf, \
         out_answer_summary_csv.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "item_id", "pair_index",
            "num_sentences", "num_relevant",
            "max_similarity", "mean_similarity",
            "classified_answer_relevant", "label_raw"
        ])

        for item in data:
            item_id = item.get("id")
            label_raw = item.get("label", "")
            gt = (item.get("ground_truth") or "").strip()
            pairs = item.get("question_answer_pairs", [])
            if not gt:
                gt_vec = None
            else:
                gt_vec = model.encode([gt])[0]

            for p_idx, pair in enumerate(pairs):
                ans = pair.get("answer", "") or ""
                sents = segment_answer(ans)
                sims: List[float] = []
                rel_flags: List[int] = []

                for s_idx, s in enumerate(sents):
                    if gt_vec is None:
                        sim = math.nan
                        rel = 0
                    else:
                        try:
                            s_vec = model.encode([s])[0]
                            sim = _cosine(gt_vec, s_vec)
                        except Exception as e:
                            sim = math.nan
                        rel = int((not math.isnan(sim)) and (sim >= threshold))
                    sims.append(sim)
                    rel_flags.append(rel)

                    row = {
                        "item_id": item_id,
                        "pair_index": p_idx,
                        "sentence_index": s_idx,
                        "sentence": s,
                        "ground_truth": gt,
                        "similarity": None if math.isnan(sims[-1]) else round(sims[-1], 6),
                        "relevant": rel,
                    }
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")

                num_sents = len(sents)
                valid_sims = [x for x in sims if not (isinstance(x, float) and math.isnan(x))]
                max_sim = max(valid_sims) if valid_sims else float("nan")
                mean_sim = (sum(valid_sims) / len(valid_sims)) if valid_sims else float("nan")
                num_rel = sum(rel_flags)
                answer_relevant = int(num_rel > 0)

                writer.writerow([
                    item_id, p_idx,
                    num_sents, num_rel,
                    "" if math.isnan(max_sim) else f"{max_sim:.6f}",
                    "" if math.isnan(mean_sim) else f"{mean_sim:.6f}",
                    answer_relevant, label_raw
                ])

def from_segments_compute(groundtruth_dataset_path: Path, segmented_jsonl: Path,
                          out_scores_jsonl: Path, out_answer_summary_csv: Path,
                          model_name: str, threshold: float) -> None:
    # Build lookup of ground truth and labels by (item_id)
    ds = json.loads(groundtruth_dataset_path.read_text(encoding="utf-8"))
    gt_map: Dict[Any, Dict[str, Any]] = {}
    for item in ds:
        gt_map[item.get("id")] = {
            "ground_truth": (item.get("ground_truth") or "").strip(),
            "label": item.get("label", ""),
        }
    ensure_nltk_data()
    model = _load_model(model_name)

    # Collect sentences by (item_id, pair_index)
    buckets: Dict[Tuple[Any, int], List[Tuple[int, str]]] = {}
    with segmented_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = (row["item_id"], row["pair_index"])
            buckets.setdefault(key, []).append((row["sentence_index"], row["sentence"]))
    # sort sentences by original index
    for key in buckets:
        buckets[key] = [s for _, s in sorted(buckets[key], key=lambda x: x[0])]

    out_scores_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_answer_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_scores_jsonl.open("w", encoding="utf-8") as jf, \
         out_answer_summary_csv.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "item_id", "pair_index",
            "num_sentences", "num_relevant",
            "max_similarity", "mean_similarity",
            "classified_answer_relevant", "label_raw"
        ])

        for (item_id, p_idx), sentences in sorted(buckets.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            meta = gt_map.get(item_id, {"ground_truth": "", "label": ""})
            gt = meta["ground_truth"]
            label_raw = meta["label"]

            if not gt:
                gt_vec = None
            else:
                gt_vec = model.encode([gt])[0]

            sims: List[float] = []
            rel_flags: List[int] = []
            for s_idx, s in enumerate(sentences):
                if gt_vec is None:
                    sim = math.nan
                    rel = 0
                else:
                    try:
                        s_vec = model.encode([s])[0]
                        sim = _cosine(gt_vec, s_vec)
                    except Exception as e:
                        sim = math.nan
                    rel = int((not math.isnan(sim)) and (sim >= threshold))
                sims.append(sim)
                rel_flags.append(rel)

                row = {
                    "item_id": item_id,
                    "pair_index": p_idx,
                    "sentence_index": s_idx,
                    "sentence": s,
                    "ground_truth": gt,
                    "similarity": None if math.isnan(sims[-1]) else round(sims[-1], 6),
                    "relevant": rel,
                }
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

            num_sents = len(sentences)
            valid_sims = [x for x in sims if not (isinstance(x, float) and math.isnan(x))]
            max_sim = max(valid_sims) if valid_sims else float("nan")
            mean_sim = (sum(valid_sims) / len(valid_sims)) if valid_sims else float("nan")
            num_rel = sum(rel_flags)
            answer_relevant = int(num_rel > 0)

            writer.writerow([
                item_id, p_idx,
                num_sents, num_rel,
                "" if math.isnan(max_sim) else f"{max_sim:.6f}",
                "" if math.isnan(mean_sim) else f"{mean_sim:.6f}",
                answer_relevant, label_raw
            ])

def main():
    parser = argparse.ArgumentParser(description="Goal 2: sentence-level semantic relevance scoring")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path, help="Raw dataset JSON; answers will be segmented internally")
    src.add_argument("--segmented_jsonl", type=Path, help="Pre-segmented sentences JSONL from Goal 1")
    parser.add_argument("--groundtruth_from_dataset", type=Path, help="Dataset JSON to pull ground_truth (required if using --segmented_jsonl)")
    parser.add_argument("--out_scores_jsonl", type=Path, required=True, help="Path to write per-sentence scores JSONL")
    parser.add_argument("--out_answer_summary_csv", type=Path, required=True, help="Path to write per-answer summary CSV")
    parser.add_argument("--threshold", type=float, default=0.60, help="Cosine similarity threshold to mark a sentence as Relevant (default: 0.60)")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-Transformers model name")
    args = parser.parse_args()

    if args.segmented_jsonl and not args.groundtruth_from_dataset:
        raise SystemExit("--groundtruth_from_dataset is required when using --segmented_jsonl")

    if args.dataset:
        from_dataset_compute(args.dataset, args.out_scores_jsonl, args.out_answer_summary_csv,
                             args.model_name, args.threshold)
    else:
        from_segments_compute(args.groundtruth_from_dataset, args.segmented_jsonl,
                              args.out_scores_jsonl, args.out_answer_summary_csv,
                              args.model_name, args.threshold)
    print(f"[OK] Wrote: {args.out_scores_jsonl}")
    print(f"[OK] Wrote: {args.out_answer_summary_csv}")

if __name__ == "__main__":
    main()
