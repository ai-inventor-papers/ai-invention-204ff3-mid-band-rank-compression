# STS-B Fine-Tuning: Mid-Band Rank Compression Test

## What This Experiment Did

This experiment investigates **mid-band rank compression** as a strategy for improving semantic textual similarity (STS) fine-tuning on the STS-Benchmark (STS-B) dataset. The core hypothesis: compressing the ranking of mid-band similarity scores (the region where models struggle most) yields better generalization than uniform training.

The experiment fine-tunes a sentence-transformer model with a custom loss that applies rank compression specifically to the mid-range similarity scores, and compares it against a standard cosine similarity baseline.

## Layout

| Path | Description |
|------|-------------|
| `method.py` | Main experiment script — data loading, model fine-tuning, evaluation, and result generation |
| `method_out.json` | Primary experiment results JSON (metrics, comparisons, statistics) |
| `fine_tuned_model_seed42/` | Saved fine-tuned model checkpoint (seed=42) |
| `logs/` | Run logs with timing, diagnostics, and training curves |
| `.aii/manifest.yaml` | Asset manifest for the disposable outputs checker |
| `README.md` | This file |

## How to Run

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the experiment
python method.py
```

This will:
1. Load and preprocess the STS-B dataset
2. Fine-tune the model with the mid-band rank compression loss
3. Evaluate against a standard baseline
4. Save results to `method_out.json` and the model checkpoint

## Restoring Removed Files

The following files/directories are marked for deletion after this round and can be restored:

### Virtual Environment (`.venv/`)

```bash
uv venv .venv --python=3.12 && uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu sentence-transformers scikit-learn scipy numpy loguru
```

## Dependencies

- Python 3.12
- `torch` (CPU build)
- `sentence-transformers`
- `scikit-learn`
- `scipy`
- `numpy`
- `loguru`

All managed via `uv`.
