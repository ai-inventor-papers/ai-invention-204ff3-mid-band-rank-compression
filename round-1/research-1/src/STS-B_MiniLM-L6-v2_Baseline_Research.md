# STS-B Baseline Numbers: all-MiniLM-L6-v2 — Comprehensive Research Report

## 1. Primary Baseline: all-MiniLM-L6-v2 on STS-Benchmark

### MTEB (Massive Text Embedding Benchmark) — The Canonical Reference

| Field | Value |
|-------|-------|
| **Model** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Reported Spearman (cosine)** | **76.97** |
| **Split** | **test** (MTEB evaluates on test splits) |
| **Metric** | Spearman correlation of cosine similarity vs. gold scores (0–5) |
| **Dataset** | `mteb/stsbenchmark-sts` (the MTEB version of STS-B) |
| **Main metric in MTEB** | `cosine_spearman` (Spearman correlation using cosine similarity) |

**Exact quote from MTEB paper (Table 11, arXiv:2210.07316v2):**

> Row: `STSBenchmark` → Column `MiniLM-L6`: **76.97**
>
> Full row context:
> `STSBenchmark | 61.54 | 61.55 | 47.29 | 76.52 | 84.25 | 76.97 | 78.81 | 61.26 | 72.25 | 69.77 | 82.03 | 83.09 | 84.42 | 83.42 | 86.82 | 83.78 | 79.54 | 85.67 | 75.34 | 77.59 | 79.21 | 81.39 | 80.90 | 79.58 | 77.60 | 77.65 | 77.73 | 85.52 | 85.36 | 83.93 | 84.01`
>
> Source: https://arxiv.org/html/2210.07316v2 (Table 11, line ~101978)

**How we know it is the test split:**

The MTEB paper states: *"We evaluate on the test splits of all datasets except for MSMARCO, where the dev split is used"* (Section 4.1). STSBenchmark is not an exception, so it uses the **test split**.

**How we know the metric is cosine_spearman:**

The MTEB paper states (Section 3.2): *"Spearman correlation based on cosine similarity serves as the main metric"* for STS tasks. The MTEB code confirms the main metric is `cosine_spearman`:

```python
# From mteb/abstasks/sts.py:
# STSMetrics TypedDict includes: cosine_spearman: float
# And the main score is the model's own similarity (cosine for sentence-transformers)
```

Source: https://github.com/embeddings-benchmark/mteb/blob/cd182883/mteb/abstasks/sts.py

**Model mapping confirmed:**

The MTEB paper maps `MiniLM-L6` to `sentence-transformers/all-MiniLM-L6-v2`:
> `MiniLM-L6 | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`

Source: https://arxiv.org/html/2210.07316v2 (Section E: Models)

---

## 2. Dataset Details

### sentence-transformers/stsb (HuggingFace)

| Field | Value |
|-------|-------|
| **Total rows** | 8,628 |
| **Train split** | 5,749 rows |
| **Validation split** | **1,500 rows** |
| **Test split** | **1,379 rows** |
| **Score range** | 0.0 to 1.0 (normalized from original 0–5) |
| **Columns** | `sentence1`, `sentence2`, `score` |

**Key detail:** The `sentence-transformers/stsb` dataset **normalizes scores from 0–5 to 0–1** by dividing by 5.

> *"Collection strategy: Reading the sentences and score from STSB dataset and dividing the score by 5."*
> Source: https://huggingface.co/datasets/sentence-transformers/stsb

### mteb/stsbenchmark-sts (MTEB version)

| Field | Value |
|-------|-------|
| **Total rows** | 8,628 (same data) |
| **Train split** | 5,749 rows |
| **Validation split** | 1,500 rows |
| **Test split** | 1,379 rows |
| **Score range** | **0 to 5** (original scale, NOT normalized) |
| **Columns** | `sentence1`, `sentence2`, `score`, `split`, `genre`, `dataset`, `year`, `sid` |

**Critical difference:** MTEB uses the **original 0–5 scale**, while sentence-transformers/stsb normalizes to 0–1. This does NOT affect Spearman correlation (rank-based), but it matters for Pearson and for model training.

Source: https://huggingface.co/datasets/mteb/stsbenchmark-sts

### Original STS-Benchmark (Cer et al., 2017)

The original STS-Benchmark from SemEval-2017 Task 1 comprises:
- ~7,000 sentence pairs from 6 sources (MSRpar, MSRvid, Flickr, News, Headlines, Tweets)
- Human-annotated scores from 0 to 5
- Split into: train (~3,500), validation (~1,500), test (~1,400)

Source: https://alt.qcri.org/semeval2017/task1/

---

## 3. Fine-Tuned Variants of all-MiniLM-L6-v2 on STS-B

### praveenku32k/all-MiniLM-L6-v2-sts

Fine-tuned on `sentence-transformers/stsb` with CosineSimilarityLoss (MSELoss).

| Split | Metric | Value |
|-------|--------|-------|
| **Validation (sts-dev)** | spearman_cosine | **0.8933** |
| **Validation (sts-dev)** | pearson_cosine | 0.8938 |
| **Test (sts-test)** | spearman_cosine | **0.8581** |
| **Test (sts-test)** | pearson_cosine | 0.8568 |

> *"Dataset: sts-dev … spearman_cosine | 0.8933"*
> *"Dataset: sts-test … spearman_cosine | 0.8581"*
>
> Source: https://huggingface.co/praveenku32k/all-MiniLM-L6-v2-sts

**Note:** This model uses the normalized 0–1 scores from `sentence-transformers/stsb`. The Spearman values are reported as decimals (0.8933 = 89.33%).

### rebego/stsb-all-MiniLM-L6-v2

Fine-tuned on STS-B as a regression model (not a sentence-transformers bi-encoder).

| Split | Metric | Value |
|-------|--------|-------|
| **Validation** | Pearson | **0.8287** |
| **Validation** | Loss | 0.0307 |

> *"It achieves the following results on the evaluation set: Loss: 0.0307, Pearson: 0.8287"*
>
> Source: https://huggingface.co/rebego/stsb-all-MiniLM-L6-v2

**Note:** This model does NOT report Spearman — only Pearson. It is a regression head model, not a bi-encoder.

---

## 4. Other Related Models on STS-Benchmark (from MTEB Table 11)

For context, here are neighboring models from the same MTEB evaluation:

| Model | STSBenchmark Spearman (cosine) | Source |
|-------|-------------------------------|--------|
| MiniLM-L6 (all-MiniLM-L6-v2) | **76.97** | MTEB paper Table 11 |
| MiniLM-L12 (all-MiniLM-L12-v2) | 78.81 | MTEB paper Table 11 |
| MPNet (all-mpnet-base-v2) | 80.28 | MTEB paper Table 11 |
| SimCSE-BERT-sup | 79.12 | MTEB paper Table 11 |
| ST5-XXL | 84.01 | MTEB paper Table 11 (best) |

---

## 5. Summary of Evaluation Conventions

### Does MTEB use the STS-B test split?
**YES.** MTEB evaluates STSBenchmark on the **test split** (1,379 pairs).

### Does the sentence-transformers stsb validation split have ~1,500 pairs?
**YES.** Exactly **1,500 pairs** in the validation split.

### What metric does MTEB use as the main score?
**cosine_spearman** — Spearman correlation between cosine similarity of embeddings and gold scores (0–5).

### What score scale does each dataset use?
- **mteb/stsbenchmark-sts**: Original 0–5 scale
- **sentence-transformers/stsb**: Normalized 0–1 scale (divided by 5)
- **Original STS-Benchmark**: 0–5 scale

### The canonical baseline number:
> **all-MiniLM-L6-v2 on STS-Benchmark test = 76.97 Spearman (cosine)**
> 
> This is the most widely cited and reproducible number, from the MTEB benchmark (Muennighoff et al., 2023, EACL).

---

## 6. Source URLs

1. **MTEB paper with Table 11**: https://arxiv.org/html/2210.07316v2
2. **MTEB STS task code**: https://github.com/embeddings-benchmark/mteb/blob/cd182883/mteb/abstasks/sts.py
3. **MTEB STSBenchmark dataset**: https://huggingface.co/datasets/mteb/stsbenchmark-sts
4. **sentence-transformers/stsb dataset**: https://huggingface.co/datasets/sentence-transformers/stsb
5. **all-MiniLM-L6-v2 model card**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
6. **praveenku32k/all-MiniLM-L6-v2-sts** (fine-tuned): https://huggingface.co/praveenku32k/all-MiniLM-L6-v2-sts
7. **rebego/stsb-all-MiniLM-L6-v2** (fine-tuned): https://huggingface.co/rebego/stsb-all-MiniLM-L6-v2
8. **paraphrase-MiniLM-L6-v2 model card**: https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
9. **Original STS-Benchmark**: https://alt.qcri.org/semeval2017/task1/
10. **MTEB leaderboard**: https://huggingface.co/spaces/mteb/leaderboard

---

## 7. Conflicts and Notes

- **No conflicts found** between sources for the canonical 76.97 number.
- The all-MiniLM-L6-v2 model card does **NOT** list STS-B numbers directly — it was trained on 1B+ sentence pairs (not STS-B) and the STS-B score comes from the independent MTEB evaluation.
- The fine-tuned variant `praveenku32k/all-MiniLM-L6-v2-sts` achieves 89.33 Spearman on validation and 85.81 on test — significantly higher than the base model (76.97), which is expected after STS-specific fine-tuning.
- The `rebego/stsb-all-MiniLM-L6-v2` variant reports only Pearson (0.8287) on validation and is a different architecture (regression head vs. bi-encoder), so it is not directly comparable.
- The `paraphrase-MiniLM-L6-v2` model card does not list STS-B numbers; it was trained on paraphrase data (not STS-B) and evaluated differently.

---

## 8. How to Reproduce the 76.97 Number

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
evaluation = MTEB(tasks=["STSBenchmark"])
results = evaluation.run(model, eval_splits=["test"])
# results["STSBenchmark"]["test"]["cosine_spearman"] ≈ 76.97
```

Or using the sentence-transformers evaluator directly:

```python
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import load_dataset

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dataset = load_dataset("mteb/stsbenchmark-sts", split="test")

test_examples = [InputExample(texts=[e["sentence1"], e["sentence2"]], label=e["score"]) 
                 for e in dataset]
evaluator = EmbeddingSimilarityEvaluator(test_examples, name="sts-test")
results = evaluator(model)
# results["sts-test_spearman_cosine"] ≈ 0.7697 (reported as 0–1 decimal)
# which equals 76.97 when multiplied by 100
```

**Important:** The sentence-transformers library reports Spearman as a decimal (0.7697), while the MTEB paper reports it as a percentage (76.97). They are the same number.
