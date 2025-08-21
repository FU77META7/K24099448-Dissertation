
# goal1_sentence_segmentation.py
'''
A simple, robust implementation for Goal 1 (Sentence Segmentation of LLM answers).

- Loads a dataset like 'reformatted_gpt_o1_responses_with_labels.json' (list of items).
- For each item and each question_answer_pair, splits the 'answer' into sentences.
- Uses NLTK Punkt with a couple of light heuristics to handle LLM-ish punctuation (semicolons, bullets).
- Writes a JSONL with one row per sentence and a CSV summary per answer.

Expected dataset schema per item (example):
{
    "id": 1,
    "question_answer_pairs": [{"question": "...", "answer": "..."}, ...],
    "ground_truth": "string",
    "label": "√" or "×"
}

Usage (CLI):
    python goal1_sentence_segmentation.py \
        --input /path/to/reformatted_gpt_o1_responses_with_labels.json \
        --out_jsonl ./segmented_sentences.jsonl \
        --out_summary_csv ./segmentation_summary.csv
'''
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception as e:
    raise SystemExit("Please install NLTK: pip install nltk") from e

def ensure_nltk_data() -> None:
    # Try to ensure punkt and punkt_tab are available; if not, download them.
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

BULLET_PATTERN = re.compile(r"^\s*[\-\u2022\*\(\[]?[a-zA-Z0-9]?\)\s*")
MULTISPACE = re.compile(r"\s+")

def _normalize_space(text: str) -> str:
    return MULTISPACE.sub(" ", text).strip()

def _split_on_semicolons(sent: str) -> List[str]:
    # Heuristic: LLMs often glue clauses with semicolons; split if long enough.
    parts = [p.strip() for p in sent.split(";")]
    # Re-join short fragments back (avoid over-splitting)
    out = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(p) < 35:  # treat as too short to be a standalone sentence
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
    # Split on newlines where line looks like a bullet or numbered list.
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
    '''
    Segment an LLM answer into sentences using:
    1) bullet-aware newline chunking,
    2) NLTK sentence tokenizer inside each chunk,
    3) light semicolon splitting.
    '''
    if not text or not text.strip():
        return []
    chunks = _split_on_bullets(text)
    if not chunks:
        chunks = [text]
    out_sents: List[str] = []
    for ch in chunks:
        # Primary sentence split
        for s in sent_tokenize(ch):
            s = _normalize_space(s)
            if not s:
                continue
            # Optional semicolon split
            semi_splits = _split_on_semicolons(s) if ";" in s else [s]
            for ss in semi_splits:
                ss = _normalize_space(ss)
                if ss and len(ss) > 0:
                    out_sents.append(ss)
    # Deduplicate consecutive duplicates (occasionally LLMs repeat lines)
    deduped = []
    for s in out_sents:
        if not deduped or deduped[-1] != s:
            deduped.append(s)
    return deduped

def process_dataset(
    input_path: Path, out_jsonl: Path, out_summary_csv: Path
) -> None:
    data: List[Dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    # JSONL rows per sentence
    with out_jsonl.open("w", encoding="utf-8") as jf, out_summary_csv.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "item_id",
            "pair_index",
            "num_sentences",
            "avg_sentence_len_chars",
            "label_raw"
        ])
        for item in data:
            item_id = item.get("id")
            label_raw = item.get("label", "")
            pairs = item.get("question_answer_pairs", [])
            for p_idx, pair in enumerate(pairs):
                ans = pair.get("answer", "") or ""
                sents = segment_answer(ans)
                # Write per-sentence JSONL
                for s_idx, s in enumerate(sents):
                    row = {
                        "item_id": item_id,
                        "pair_index": p_idx,
                        "sentence_index": s_idx,
                        "sentence": s,
                    }
                    jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                # CSV summary for this answer
                avg_len = (sum(len(s) for s in sents) / max(1, len(sents))) if sents else 0.0
                writer.writerow([item_id, p_idx, len(sents), f"{avg_len:.2f}", label_raw])

def main():
    parser = argparse.ArgumentParser(description="Goal 1: sentence segmentation for LLM answers")
    parser.add_argument("--input", type=Path, required=True, help="Path to dataset JSON")
    parser.add_argument("--out_jsonl", type=Path, required=True, help="Path to write per-sentence JSONL")
    parser.add_argument("--out_summary_csv", type=Path, required=True, help="Path to write summary CSV")
    args = parser.parse_args()

    ensure_nltk_data()
    process_dataset(args.input, args.out_jsonl, args.out_summary_csv)
    print(f"[OK] Wrote: {args.out_jsonl}")
    print(f"[OK] Wrote: {args.out_summary_csv}")

if __name__ == "__main__":
    main()
