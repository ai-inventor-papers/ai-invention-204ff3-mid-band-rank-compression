#!/usr/bin/env python3
"""Load STS-Benchmark from temp/datasets/, standardize to exp_sel_data_out schema,
derive score bands and lexical-overlap metadata, stratified sample ≤2000 train rows,
reserve ~120 validation holdout rows, and save full_data_out.json."""

from loguru import logger
from pathlib import Path
import json
import re
import sys
import math
import resource
from typing import Any

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# Memory limit: 2 GB (artifact plan says ~2 GB RAM needed)
RAM_BUDGET = 2 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

WORKSPACE = Path(__file__).resolve().parent
DATASETS_DIR = WORKSPACE / "temp" / "datasets"
OUTPUT = WORKSPACE / "full_data_out.json"


def tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric tokens for lexical overlap."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard_overlap(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens_a and not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def score_band(score: float) -> str:
    """Coarse score band: low < 2, mid 2-<4, high >= 4."""
    if score < 2.0:
        return "low"
    elif score < 4.0:
        return "mid"
    else:
        return "high"


def overlap_bin(jaccard: float) -> str:
    """Lexical overlap bin: low < 0.2, mid 0.2-<0.5, high >= 0.5."""
    if jaccard < 0.2:
        return "low"
    elif jaccard < 0.5:
        return "mid"
    else:
        return "high"


def stratified_sample(rows: list[dict], max_rows: int, seed: int = 42) -> list[dict]:
    """Stratified sampling by (score_band, overlap_bin) to keep all bands represented."""
    from collections import defaultdict
    import random

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["metadata_score_band"], row["metadata_overlap_bin"])
        groups[key].append(row)

    rng = random.Random(seed)
    sampled: list[dict] = []

    # Ensure at least 1 from each group
    for key in sorted(groups.keys()):
        group = groups[key]
        rng.shuffle(group)
        sampled.append(group[0])

    # Fill remaining slots proportionally
    remaining = max_rows - len(sampled)
    if remaining <= 0:
        return sampled[:max_rows]

    total_group_sizes = sum(len(g) for g in groups.values())
    for key, group in groups.items():
        # Proportional share minus the 1 already taken
        share = max(0, int(remaining * len(group) / total_group_sizes))
        if share > len(group) - 1:
            share = len(group) - 1
        if share > 0:
            rng.shuffle(group)
            sampled.extend(group[1:1 + share])

    return sampled[:max_rows]


def load_split(split_name: str) -> list[dict]:
    """Load a split from the downloaded full JSON file."""
    fname = f"full_mteb_stsbenchmark-sts_default_{split_name}.json"
    fpath = DATASETS_DIR / fname
    if not fpath.exists():
        logger.error(f"Split file not found: {fpath}")
        raise FileNotFoundError(f"Missing split file: {fpath}")
    data = json.loads(fpath.read_text())
    logger.info(f"Loaded {split_name}: {len(data)} rows")
    return data


@logger.catch(reraise=True)
def main():
    logger.info("=== STS-Benchmark Data Preparation ===")

    # Load all splits
    train_rows = load_split("train")
    val_rows = load_split("validation")
    test_rows = load_split("test")

    # Process all rows into standardized examples
    def process_rows(rows: list[dict], split_label: str, is_holdout: bool = False) -> list[dict]:
        examples = []
        for idx, row in enumerate(rows):
            s1 = row["sentence1"]
            s2 = row["sentence2"]
            score = float(row["score"])

            # Compute lexical overlap
            tokens1 = tokenize(s1)
            tokens2 = tokenize(s2)
            jacc = jaccard_overlap(tokens1, tokens2)

            # Build input as JSON string of the two sentences
            input_str = json.dumps({"sentence1": s1, "sentence2": s2})

            # Output is the gold score as string
            output_str = str(score)

            example: dict[str, Any] = {
                "input": input_str,
                "output": output_str,
                "metadata_row_index": idx,
                "metadata_split": split_label,
                "metadata_score": score,
                "metadata_score_band": score_band(score),
                "metadata_overlap_jaccard": round(jacc, 4),
                "metadata_overlap_bin": overlap_bin(jacc),
                "metadata_genre": row.get("genre", ""),
                "metadata_source_dataset": row.get("dataset", ""),
                "metadata_year": row.get("year", ""),
                "metadata_sid": row.get("sid", ""),
                "metadata_task_type": "regression",
                "metadata_is_holdout": is_holdout,
            }
            examples.append(example)
        return examples

    # Process train: sample ≤2000 with stratified sampling
    all_train_examples = process_rows(train_rows, "train")
    sampled_train = stratified_sample(all_train_examples, max_rows=2000, seed=42)
    logger.info(f"Train: {len(all_train_examples)} → {len(sampled_train)} (stratified sample)")

    # Process validation: reserve ~120 holdout rows (stratified)
    all_val_examples = process_rows(val_rows, "validation")
    holdout_size = 120
    holdout_examples = stratified_sample(all_val_examples, max_rows=holdout_size, seed=42)
    holdout_sids = {e["metadata_sid"] for e in holdout_examples}

    # Mark holdout rows
    for ex in all_val_examples:
        if ex["metadata_sid"] in holdout_sids:
            ex["metadata_is_holdout"] = True
        else:
            ex["metadata_is_holdout"] = False

    logger.info(f"Validation: {len(all_val_examples)} total, {holdout_size} holdout reserved")

    # Process test: no holdout
    all_test_examples = process_rows(test_rows, "test")
    logger.info(f"Test: {len(all_test_examples)} rows")

    # Combine all into one dataset group
    all_examples = sampled_train + all_val_examples + all_test_examples

    output_data = {
        "metadata": {
            "source": "mteb/stsbenchmark-sts",
            "description": "STS-Benchmark sentence pairs with human similarity scores (0-5), "
                           "score bands (low/mid/high), lexical overlap bins (low/mid/high), "
                           "and a reserved validation holdout of ~120 rows.",
            "train_rows": len(sampled_train),
            "validation_rows": len(all_val_examples),
            "test_rows": len(all_test_examples),
            "holdout_rows": holdout_size,
            "total_rows": len(all_examples),
            "score_scale": "0-5",
            "score_bands": "low < 2, mid 2-<4, high >= 4",
            "overlap_bins": "low < 0.2, mid 0.2-<0.5, high >= 0.5",
        },
        "datasets": [
            {
                "dataset": "stsbenchmark",
                "examples": all_examples,
            }
        ],
    }

    # Save
    OUTPUT.write_text(json.dumps(output_data, indent=2))
    logger.info(f"Saved {len(all_examples)} examples to {OUTPUT}")

    # Band distribution stats
    from collections import Counter
    band_counts = Counter(e["metadata_score_band"] for e in sampled_train)
    logger.info(f"Train band distribution: {dict(band_counts)}")
    overlap_counts = Counter(e["metadata_overlap_bin"] for e in sampled_train)
    logger.info(f"Train overlap distribution: {dict(overlap_counts)}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
