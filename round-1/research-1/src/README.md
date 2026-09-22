# STS-B Baseline Research: all-MiniLM-L6-v2

Comprehensive web research collecting published baseline numbers for `sentence-transformers/all-MiniLM-L6-v2` on the STS-Benchmark (STS-B) semantic textual similarity task.

## What This Is

A research report answering: **"What is the published STS-B Spearman correlation for all-MiniLM-L6-v2?"**

## Findings at a Glance

| Model | Split | Spearman (cosine) | Source |
|-------|-------|-------------------|--------|
| all-MiniLM-L6-v2 (base) | test | **76.97** | MTEB (Muennighoff et al., 2023) |
| praveenku32k/all-MiniLM-L6-v2-sts | validation | 89.33 | HuggingFace model card |
| praveenku32k/all-MiniLM-L6-v2-sts | test | 85.81 | HuggingFace model card |
| rebego/stsb-all-MiniLM-L6-v2 | validation | 82.87 (Pearson) | HuggingFace model card |

## Layout

```
.
├── STS-B_MiniLM-L6-v2_Baseline_Research.md   # Full research report with quotes, URLs, and reproduction code
├── README.md                                  # This file
└── .aii/
    └── manifest.yaml                          # Disposable outputs manifest
```

## Key Answers

1. **Canonical baseline**: 76.97 Spearman (cosine) on the STS-Benchmark **test** split, from MTEB (arXiv:2210.07316).
2. **MTEB uses the test split**: Yes — MTEB evaluates on test splits for all datasets except MSMARCO.
3. **Validation split size**: The `sentence-transformers/stsb` dataset has exactly **1,500 pairs** in the validation split.
4. **Score scale**: MTEB uses original 0–5 scores; `sentence-transformers/stsb` normalizes to 0–1.

## How to Run

Nothing to run — this is a static research report. To reproduce the 76.97 number, see the reproduction code in the main report.

## Restoring removed files

No files were removed. This run produced only text files.
