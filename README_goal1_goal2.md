
# README_goal1_goal2.md
## Fine-Grained QA Analysis — Goal 1 & Goal 2 (Implementation)

This repo contains two standalone Python scripts that implement your dissertation’s **Goal 1** (sentence segmentation) and **Goal 2** (sentence-level semantic relevance scoring). They are designed to work directly with your uploaded dataset:

- `reformatted_gpt_o1_responses_with_labels.json`

### Files

- `goal1_sentence_segmentation.py`  
  Segments every model answer into sentences and writes:
  - `segmented_sentences.jsonl` — one JSON line per sentence (with `item_id`, `pair_index`, `sentence_index`, `sentence`).
  - `segmentation_summary.csv` — per-answer counts and average sentence length.

- `goal2_sentence_relevance.py`  
  Computes cosine similarities between each sentence and the item’s `ground_truth` using Sentence-Transformers:
  - `sentence_scores.jsonl` — per sentence scores with a `relevant` flag (thresholded cosine).
  - `answer_summary.csv` — per-answer aggregate metrics.

### Quickstart

1) Create a virtual env (recommended) and install dependencies:
```bash
pip install nltk sentence-transformers numpy
```

2) Run Goal 1 (segmentation):
```bash
python goal1_sentence_segmentation.py \
  --input /mnt/data/reformatted_gpt_o1_responses_with_labels.json \
  --out_jsonl /mnt/data/segmented_sentences.jsonl \
  --out_summary_csv /mnt/data/segmentation_summary.csv
```

3) Run Goal 2 in **either** of two modes:

**A. End-to-end from raw dataset (auto-segment):**
```bash
python goal2_sentence_relevance.py \
  --dataset /mnt/data/reformatted_gpt_o1_responses_with_labels.json \
  --out_scores_jsonl /mnt/data/sentence_scores.jsonl \
  --out_answer_summary_csv /mnt/data/answer_summary.csv
```

**B. From pre-segmented JSONL (exact sentences from Goal 1):**
```bash
python goal2_sentence_relevance.py \
  --groundtruth_from_dataset /mnt/data/reformatted_gpt_o1_responses_with_labels.json \
  --segmented_jsonl /mnt/data/segmented_sentences.jsonl \
  --out_scores_jsonl /mnt/data/sentence_scores.jsonl \
  --out_answer_summary_csv /mnt/data/answer_summary.csv
```

### Notes & Tips

- **Model & rate limits:** The default model is `sentence-transformers/all-MiniLM-L6-v2`. If you hit 429 rate limits on the first download, the script retries with exponential backoff and finally falls back to `multi-qa-MiniLM-L6-cos-v1`.
- **Threshold:** `--threshold` (default `0.60`) controls the relevant/irrelevant cut. Tune this based on your validation—e.g., inspect `sentence_scores.jsonl` and `answer_summary.csv`.
- **NLTK:** The scripts auto-download `punkt` if missing.
- **Idempotent outputs:** It’s safe to re-run; outputs are overwritten.

### Outputs (schema)

**segmented_sentences.jsonl** (Goal 1)
```json
{"item_id": 1, "pair_index": 0, "sentence_index": 0, "sentence": "First sentence."}
```

**segmentation_summary.csv** (Goal 1)
```csv
item_id,pair_index,num_sentences,avg_sentence_len_chars,label_raw
1,0,3,76.33,√
```

**sentence_scores.jsonl** (Goal 2)
```json
{"item_id":1,"pair_index":0,"sentence_index":0,"sentence":"First sentence.","ground_truth":"Odorama","similarity":0.812345,"relevant":1}
```

**answer_summary.csv** (Goal 2)
```csv
item_id,pair_index,num_sentences,num_relevant,max_similarity,mean_similarity,classified_answer_relevant,label_raw
1,0,3,1,0.812345,0.456789,1,√
```

### Repro tweaks you can add later
- Calibrate the relevance threshold with ROC/PR against a human-labeled subset.
- Store sentence/GT embeddings to disk to speed up multi-pass experiments.
- Try alternative encoders (e.g., `all-mpnet-base-v2`) and compare via `answer_summary.csv`.
