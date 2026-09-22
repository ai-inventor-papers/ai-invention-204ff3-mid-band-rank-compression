# STS-Benchmark Mid-Band Package

## Overview
Dataset package for evaluating whether a tiny sentence-embedding model (all-MiniLM-L6-v2) can be improved on semantic textual similarity with small-scale fine-tuning.

## Dataset
**Source:** `mteb/stsbenchmark-sts` (Cer et al., 2017)
- **Train:** 1,994 rows (stratified sample from 5,749)
- **Validation:** 1,500 rows (120 reserved as holdout)
- **Test:** 1,379 rows
- **Total:** 4,873 examples

Each row contains:
- `input`: JSON string with `sentence1` and `sentence2`
- `output`: Gold similarity score (0-5) as string
- `metadata_score_band`: low (<2), mid (2-<4), high (>=4)
- `metadata_overlap_bin`: low (<0.2), mid (0.2-<0.5), high (>=0.5) Jaccard token overlap
- `metadata_is_holdout`: True for the 120 reserved validation rows

## Layout
```
data.py                      # Data preparation script
full_data_out.json           # Complete dataset (4,873 examples, 3.4 MB)
mini_full_data_out.json      # First 3 examples
preview_full_data_out.json   # First 3 examples (truncated)
temp/datasets/               # Raw downloaded STS-Benchmark files
logs/                        # Run logs
```

## How to Run
```bash
uv venv .venv --python=3.12
source .venv/bin/activate
uv pip install loguru
python data.py
```

## Restoring Removed Files
No files are marked for deletion. All data is kept.
