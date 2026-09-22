#!/usr/bin/env python3
"""
STS-B Fine-Tuning: Mid-Band Rank Compression Test

Runs full experimental protocol:
1. Fix holdout flagging bug by deduplicating validation rows by SID
2. Baseline eval of all-MiniLM-L6-v2 on validation split
3. One-epoch fine-tuning on <=2000 training pairs
4. Post-tuning eval
5. 9-diagnostic battery
6. Repeat with 3 additional random seeds for variance estimation
7. Report wall-clock time and peak memory
8. Save fine-tuned model checkpoint
"""

import json
import gc
import math
import os
import random
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.stats import spearmanr, pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# ── Environment (must be BEFORE any transformers use) ─────────────────────
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ── Logging ────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
os.makedirs("logs", exist_ok=True)
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

# ── Constants ──────────────────────────────────────────────────────────────
DATA_PATH = Path("/ai-inventor/aii_data/runs/run_tTU5_wy6DyGe/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/full_data_out.json")
MODEL_CACHE = Path("/ai-inventor/aii_data/runs/run_tTU5_wy6DyGe/.shared_cache/hf/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
OUTPUT_PATH = Path(__file__).parent / "method_out.json"
CHECKPOINT_DIR = Path(__file__).parent / "fine_tuned_model_seed42"
BASE_WEIGHTS_PATH = Path(__file__).parent / "base_model_weights.pt"

SEEDS = [42, 123, 456, 789]
BATCH_SIZE = 16
MAX_LEN = 64
LR = 2e-5
EPOCHS = 1
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
TRAIN_LIMIT = 2000
DEVICE = torch.device("cpu")

# ── Helpers ────────────────────────────────────────────────────────────────
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def safe_spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    try:
        return float(spearmanr(x, y)[0])
    except Exception:
        return None

def safe_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    try:
        return float(pearsonr(x, y)[0])
    except Exception:
        return None

def safe_mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(mean_absolute_error(x, y))

# ── Embedding helpers ──────────────────────────────────────────────────────
def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_exp = mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_emb = (last_hidden * mask_exp).sum(dim=1)
    len_emb = mask_exp.sum(dim=1).clamp(min=1e-9)
    return sum_emb / len_emb

def compute_embeddings_for_drift(
    model, examples: List[Dict], tokenizer, max_len: int = 64, batch_size: int = 32
) -> torch.Tensor:
    model.eval()
    sentences = []
    for ex in examples:
        inp = json.loads(ex["input"])
        sentences.append(inp["sentence1"])
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            enc = tokenizer(batch, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
            out = model(**{k: v.to(DEVICE) for k, v in enc.items()})
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"].to(DEVICE))
            emb = nn.functional.normalize(emb, p=2, dim=1)
            all_emb.append(emb.cpu())
    return torch.cat(all_emb, dim=0)

# ── Single seed experiment ─────────────────────────────────────────────────
def run_single_seed(
    seed: int,
    train_examples: List[Dict],
    val_examples: List[Dict],
    tokenizer: AutoTokenizer,
    weights_path: Path,
    save_checkpoint: bool = False,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    set_seeds(seed)
    t0 = time.time()

    # Load model from saved weights
    model = AutoModel.from_pretrained(str(MODEL_CACHE))
    model.load_state_dict(torch.load(str(weights_path), map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    # ── Baseline eval ──────────────────────────────────────────────────────
    model.eval()
    s1_list = [json.loads(e["input"])["sentence1"] for e in val_examples]
    s2_list = [json.loads(e["input"])["sentence2"] for e in val_examples]
    gold = np.array([e["metadata_score"] for e in val_examples])

    emb1_base = []
    emb2_base = []
    with torch.no_grad():
        for i in range(0, len(s1_list), BATCH_SIZE):
            b1 = s1_list[i:i+BATCH_SIZE]
            b2 = s2_list[i:i+BATCH_SIZE]
            e1 = tokenizer(b1, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
            e2 = tokenizer(b2, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
            o1 = model(**{k: v.to(DEVICE) for k, v in e1.items()})
            o2 = model(**{k: v.to(DEVICE) for k, v in e2.items()})
            emb1_base.append(nn.functional.normalize(mean_pool(o1.last_hidden_state, e1["attention_mask"].to(DEVICE)), p=2, dim=1).cpu())
            emb2_base.append(nn.functional.normalize(mean_pool(o2.last_hidden_state, e2["attention_mask"].to(DEVICE)), p=2, dim=1).cpu())
    emb1_base = torch.cat(emb1_base, dim=0)
    emb2_base = torch.cat(emb2_base, dim=0)
    base_cosine = torch.mm(emb1_base, emb2_base.t()).diag().numpy()

    baseline = {
        "spearman": safe_spearman(gold, base_cosine),
        "pearson": safe_pearson(gold, base_cosine),
        "mae": safe_mae(gold, base_cosine),
    }
    logger.info(f"Seed {seed} Baseline - Spearman: {baseline['spearman']:.4f}")

    # ── Training ───────────────────────────────────────────────────────────
    # Pre-tokenize training pairs
    train_data = []
    for ex in train_examples:
        inp = json.loads(ex["input"])
        score = float(ex["metadata_score"]) / 5.0
        e1 = tokenizer(inp["sentence1"], max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        e2 = tokenizer(inp["sentence2"], max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        train_data.append({
            "input_ids_1": e1["input_ids"].squeeze(0),
            "attention_mask_1": e1["attention_mask"].squeeze(0),
            "input_ids_2": e2["input_ids"].squeeze(0),
            "attention_mask_2": e2["attention_mask"].squeeze(0),
            "label": torch.tensor(score, dtype=torch.float32),
        })

    class TrainDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_ds = TrainDataset(train_data)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            ids1 = batch["input_ids_1"].to(DEVICE)
            mask1 = batch["attention_mask_1"].to(DEVICE)
            ids2 = batch["input_ids_2"].to(DEVICE)
            mask2 = batch["attention_mask_2"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            o1 = model(input_ids=ids1, attention_mask=mask1)
            o2 = model(input_ids=ids2, attention_mask=mask2)

            e1 = nn.functional.normalize(mean_pool(o1.last_hidden_state, mask1), p=2, dim=1)
            e2 = nn.functional.normalize(mean_pool(o2.last_hidden_state, mask2), p=2, dim=1)

            cos_sim = (e1 * e2).sum(dim=1)
            loss = criterion(cos_sim, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % 100 == 0:
                logger.info(f"  Epoch {epoch+1} Step {step+1}/{len(loader)} Loss: {loss.item():.4f}")

    # ── Post-tuning eval ───────────────────────────────────────────────────
    model.eval()
    emb1_tuned = []
    emb2_tuned = []
    with torch.no_grad():
        for i in range(0, len(s1_list), BATCH_SIZE):
            b1 = s1_list[i:i+BATCH_SIZE]
            b2 = s2_list[i:i+BATCH_SIZE]
            e1 = tokenizer(b1, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
            e2 = tokenizer(b2, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
            o1 = model(**{k: v.to(DEVICE) for k, v in e1.items()})
            o2 = model(**{k: v.to(DEVICE) for k, v in e2.items()})
            emb1_tuned.append(nn.functional.normalize(mean_pool(o1.last_hidden_state, e1["attention_mask"].to(DEVICE)), p=2, dim=1).cpu())
            emb2_tuned.append(nn.functional.normalize(mean_pool(o2.last_hidden_state, e2["attention_mask"].to(DEVICE)), p=2, dim=1).cpu())
    emb1_tuned = torch.cat(emb1_tuned, dim=0)
    emb2_tuned = torch.cat(emb2_tuned, dim=0)
    tuned_cosine = torch.mm(emb1_tuned, emb2_tuned.t()).diag().numpy()

    post_tuning = {
        "spearman": safe_spearman(gold, tuned_cosine),
        "pearson": safe_pearson(gold, tuned_cosine),
        "mae": safe_mae(gold, tuned_cosine),
    }
    post_tuning["spearman_delta"] = post_tuning["spearman"] - baseline["spearman"] if post_tuning["spearman"] is not None and baseline["spearman"] is not None else 0.0
    post_tuning["pearson_delta"] = post_tuning["pearson"] - baseline["pearson"] if post_tuning["pearson"] is not None and baseline["pearson"] is not None else 0.0
    logger.info(f"Seed {seed} Post-tuning - Spearman: {post_tuning['spearman']:.4f} (delta: {post_tuning['spearman_delta']:+.4f})")

    # ── Save checkpoint for primary seed ───────────────────────────────────
    if save_checkpoint and checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    elapsed = time.time() - t0
    mem = get_peak_memory_mb()

    # Clean up
    del model, emb1_base, emb2_base, emb1_tuned, emb2_tuned, train_ds, loader, optimizer, scheduler
    gc.collect()

    result = {
        "seed": seed,
        "baseline": baseline,
        "post_tuning": post_tuning,
        "wall_clock_seconds": elapsed,
        "peak_memory_mb": mem,
        "gold_scores": gold.tolist(),
        "base_cosine": base_cosine.tolist(),
        "tuned_cosine": tuned_cosine.tolist(),
    }
    logger.info(f"Seed {seed} COMPLETE - Spearman delta: {result['post_tuning']['spearman_delta']:+.4f}, Time: {result['wall_clock_seconds']:.1f}s")
    return result

# ── Diagnostics ────────────────────────────────────────────────────────────
def get_band(score: float) -> str:
    if score < 2.0:
        return "low"
    elif score < 4.0:
        return "mid"
    else:
        return "high"

def compute_diagnostics(
    val_examples: List[Dict],
    base_cosine: np.ndarray,
    tuned_cosine: np.ndarray,
    gold_scores: np.ndarray,
    base_embeddings: Optional[torch.Tensor] = None,
    tuned_embeddings: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {}

    # 1. Within-band Spearman
    wbs_base = {}
    wbs_tuned = {}
    for band in ["low", "mid", "high"]:
        mask = np.array([get_band(g) == band for g in gold_scores])
        wbs_base[band] = safe_spearman(gold_scores[mask], base_cosine[mask])
        wbs_tuned[band] = safe_spearman(gold_scores[mask], tuned_cosine[mask])
    diag["within_band_spearman"] = {"base": wbs_base, "tuned": wbs_tuned}

    # 2. Within-band cosine std
    wbc_base = {}
    wbc_tuned = {}
    for band in ["low", "mid", "high"]:
        mask = np.array([get_band(g) == band for g in gold_scores])
        if mask.sum() >= 2:
            wbc_base[band] = float(np.std(base_cosine[mask]))
            wbc_tuned[band] = float(np.std(tuned_cosine[mask]))
        else:
            wbc_base[band] = None
            wbc_tuned[band] = None
    diag["within_band_cosine_std"] = {"base": wbc_base, "tuned": wbc_tuned}

    # 3. Mid-band order inversions
    mid_mask = np.array([get_band(g) == "mid" for g in gold_scores])
    mid_gold = gold_scores[mid_mask]
    mid_base = base_cosine[mid_mask]
    mid_tuned = tuned_cosine[mid_mask]
    n_mid = len(mid_gold)

    def count_inversions(g, c, max_pairs=500):
        n = len(g)
        if n < 2:
            return 0
        if n * (n - 1) // 2 > max_pairs:
            idx = np.random.choice(n, size=min(max_pairs, n), replace=False)
            g, c = g[idx], c[idx]
            n = len(g)
        inv = 0
        for i in range(n):
            for j in range(i + 1, n):
                go = g[i] - g[j]
                co = c[i] - c[j]
                if go != 0 and np.sign(go) != np.sign(co):
                    inv += 1
        return inv

    diag["mid_band_order_inversions"] = {
        "base": count_inversions(mid_gold, mid_base),
        "tuned": count_inversions(mid_gold, mid_tuned),
    }

    # 4. Leave-one-slice-out decomposition
    total_s, _ = spearmanr(gold_scores, tuned_cosine)
    base_s, _ = spearmanr(gold_scores, base_cosine)
    total_delta = total_s - base_s
    loo = {}
    for band in ["low", "mid", "high"]:
        mask = np.array([get_band(g) == band for g in gold_scores])
        hybrid = tuned_cosine.copy()
        hybrid[mask] = base_cosine[mask]
        hs, _ = spearmanr(gold_scores, hybrid)
        delta = hs - base_s
        frac = float(delta / total_delta) if abs(total_delta) > 1e-9 else 0.0
        loo[band] = {"hybrid_spearman": float(hs), "delta": float(delta), "fraction_of_total": frac}
    diag["leave_one_slice_out_decomposition"] = loo

    # 5. Isotonic calibration gain
    try:
        iso = IsotonicRegression(y_min=0.0, y_max=5.0, out_of_bounds="clip")
        iso_cos = iso.fit_transform(base_cosine, gold_scores)
        bp = float(pearsonr(gold_scores, base_cosine)[0])
        ip = float(pearsonr(gold_scores, iso_cos)[0])
        tp = float(pearsonr(gold_scores, tuned_cosine)[0])
        bm = mean_absolute_error(gold_scores, base_cosine)
        im = mean_absolute_error(gold_scores, iso_cos)
        tm = mean_absolute_error(gold_scores, tuned_cosine)
        pg_iso = ip - bp
        pg_tuned = tp - bp
        mg_iso = bm - im
        mg_tuned = bm - tm
        iso_pf = abs(pg_iso) / abs(pg_tuned) if abs(pg_tuned) > 1e-9 else 0.0
        iso_mf = abs(mg_iso) / abs(mg_tuned) if abs(mg_tuned) > 1e-9 else 0.0
        diag["isotonic_calibration"] = {
            "base_pearson": bp, "iso_pearson": ip, "tuned_pearson": tp,
            "base_mae": bm, "iso_mae": im, "tuned_mae": tm,
            "iso_pearson_fraction": float(iso_pf), "iso_mae_fraction": float(iso_mf),
        }
    except Exception as e:
        diag["isotonic_calibration"] = {"error": str(e)}

    # 6. Embedding drift
    if base_embeddings is not None and tuned_embeddings is not None:
        n_drift = min(200, len(base_embeddings))
        idx = np.random.choice(len(base_embeddings), size=n_drift, replace=False)
        drift = 1.0 - torch.mm(base_embeddings[idx], tuned_embeddings[idx].t()).diag().mean().item()
        diag["embedding_drift"] = {"mean_drift": float(drift)}
    else:
        diag["embedding_drift"] = {"mean_drift": None}

    # 7. Lexical quadrant analysis
    quadrants = defaultdict(list)
    for i, ex in enumerate(val_examples):
        overlap = ex.get("metadata_overlap_jaccard", 0.0)
        score = gold_scores[i]
        oc = "low" if overlap < 0.5 else "high"
        gc_cat = "low" if score < 3.0 else "high"
        key = f"{oc}_overlap_{gc_cat}_gold"
        quadrants[key].append(i)

    def quad_spearman(cos_arr):
        res = {}
        for key, indices in quadrants.items():
            if len(indices) >= 3:
                res[key] = safe_spearman(gold_scores[np.array(indices)], cos_arr[np.array(indices)])
            else:
                res[key] = None
        return res

    diag["lexical_quadrant_base"] = quad_spearman(base_cosine)
    diag["lexical_quadrant_tuned"] = quad_spearman(tuned_cosine)

    return diag

# ── Main ───────────────────────────────────────────────────────────────────
@logger.catch(reraise=True)
def main():
    experiment_start = time.time()
    logger.info("=" * 70)
    logger.info("STS-B Fine-tuning Experiment: One-Epoch MiniLM Adaptation")
    logger.info("=" * 70)

    # ── Set RAM limit ──────────────────────────────────────────────────────
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info(f"Loading data from {DATA_PATH}")
    data = json.loads(DATA_PATH.read_text())
    examples = data["datasets"][0]["examples"]

    train_examples = [e for e in examples if e["metadata_split"] == "train"]
    val_examples = [e for e in examples if e["metadata_split"] == "validation"]
    test_examples = [e for e in examples if e["metadata_split"] == "test"]
    logger.info(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    # Fix holdout bug
    val_by_sid = {}
    for ex in val_examples:
        sid = ex["metadata_sid"]
        if sid not in val_by_sid:
            val_by_sid[sid] = ex
    unique_val = list(val_by_sid.values())

    rng = random.Random(42)
    bands = {"low": [], "mid": [], "high": []}
    for ex in unique_val:
        bands[ex["metadata_score_band"]].append(ex)

    holdout_sids = set()
    for band_name, band_exs in bands.items():
        n_sample = max(1, int(120 * len(band_exs) / len(unique_val)))
        n_sample = min(n_sample, len(band_exs))
        for ex in rng.sample(band_exs, n_sample):
            holdout_sids.add(ex["metadata_sid"])

    seen_sids = set()
    for ex in val_examples:
        sid = ex["metadata_sid"]
        if sid in holdout_sids and sid not in seen_sids:
            ex["metadata_is_holdout"] = True
            seen_sids.add(sid)
        else:
            ex["metadata_is_holdout"] = False

    val_eval = [e for e in val_examples if not e["metadata_is_holdout"]]
    logger.info(f"Holdout: {len(holdout_sids)} SIDs, Val eval rows: {len(val_eval)}")

    # Limit training data
    if len(train_examples) > TRAIN_LIMIT:
        rng2 = random.Random(42)
        train_examples = rng2.sample(train_examples, TRAIN_LIMIT)
        logger.info(f"Training limited to {len(train_examples)} examples")

    # ── Load tokenizer and save base weights ───────────────────────────────
    logger.info("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_CACHE))
    base_model = AutoModel.from_pretrained(str(MODEL_CACHE))
    base_model.save_pretrained(str(MODEL_CACHE))  # ensure cached
    torch.save(base_model.state_dict(), str(BASE_WEIGHTS_PATH))
    del base_model
    gc.collect()
    logger.info(f"Base weights saved to {BASE_WEIGHTS_PATH}")

    # ── Run seeds ──────────────────────────────────────────────────────────
    all_results = []
    reduce_seeds = False

    for i, seed in enumerate(SEEDS):
        if reduce_seeds and i > 0:
            logger.info("Reducing to 2 seeds due to time constraints")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed} (run {i+1}/{len(SEEDS)})")
        logger.info(f"{'='*60}")

        try:
            save_ckpt = (seed == SEEDS[0])
            result = run_single_seed(
                seed, train_examples, val_eval, tokenizer, BASE_WEIGHTS_PATH,
                save_checkpoint=save_ckpt, checkpoint_dir=CHECKPOINT_DIR if save_ckpt else None,
            )
            all_results.append(result)

            # Check timing
            if i == 0 and result["wall_clock_seconds"] > 2700:
                logger.warning("First seed > 45 min, reducing to 2 seeds")
                reduce_seeds = True
            elif i > 0:
                remaining = len(SEEDS) - i - 1
                est = result["wall_clock_seconds"] * remaining
                logger.info(f"Estimated remaining: {est/60:.1f} min")

        except Exception as e:
            logger.error(f"Seed {seed} failed: {e}")
            continue

    # ── Compute diagnostics on primary seed ────────────────────────────────
    primary = all_results[0] if all_results else None
    primary_diagnostics = {}
    if primary:
        gold = np.array(primary["gold_scores"])
        base_cos = np.array(primary["base_cosine"])
        tuned_cos = np.array(primary["tuned_cosine"])
        primary_diagnostics = compute_diagnostics(val_eval, base_cos, tuned_cos, gold)

    # ── Seed variance ──────────────────────────────────────────────────────
    deltas = [r["post_tuning"]["spearman_delta"] for r in all_results]
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    std_delta = float(np.std(deltas)) if len(deltas) > 1 else 0.0

    # ── Hypothesis assessment ──────────────────────────────────────────────
    if primary and primary["post_tuning"]["spearman"] is not None:
        spearman_improved = primary["post_tuning"]["spearman_delta"] > 0
        delta_exceeds = abs(primary["post_tuning"]["spearman_delta"]) > 2 * std_delta if std_delta > 0 else False

        loo = primary_diagnostics.get("leave_one_slice_out_decomposition", {})
        mid_frac = abs(loo.get("mid", {}).get("fraction_of_total", 0.0))

        drift = primary_diagnostics.get("embedding_drift", {}).get("mean_drift", 1.0)
        drift_ok = drift is not None and drift < 0.05

        iso = primary_diagnostics.get("isotonic_calibration", {})
        iso_frac = iso.get("iso_pearson_fraction", None)
        iso_ok = iso_frac is not None and iso_frac < 0.5

        mid_supported = (
            spearman_improved and delta_exceeds and mid_frac >= 0.5
            and drift_ok and (iso_ok if iso_frac is not None else True)
        )
    else:
        spearman_improved = False
        delta_exceeds = False
        mid_frac = 0.0
        drift_ok = False
        iso_ok = False
        mid_supported = False

    # ── Build output in exp_gen_sol_out schema ─────────────────────────────
    # Each validation example becomes an "example" with input, output, and predictions
    total_wall = sum(r["wall_clock_seconds"] for r in all_results)
    peak_mem = max(r["peak_memory_mb"] for r in all_results) if all_results else 0.0

    # Build per-example records
    examples_out = []
    if primary:
        gold_arr = np.array(primary["gold_scores"])
        base_arr = np.array(primary["base_cosine"])
        tuned_arr = np.array(primary["tuned_cosine"])
        for i, ex in enumerate(val_eval):
            record = {
                "input": ex["input"],
                "output": str(ex["metadata_score"]),
                "predict_baseline": str(round(float(base_arr[i]), 6)),
                "predict_tuned": str(round(float(tuned_arr[i]), 6)),
                "metadata_row_index": ex.get("metadata_row_index", i),
                "metadata_split": ex.get("metadata_split", "validation"),
                "metadata_score": ex["metadata_score"],
                "metadata_score_band": ex.get("metadata_score_band", "mid"),
                "metadata_overlap_jaccard": ex.get("metadata_overlap_jaccard", 0.0),
                "metadata_overlap_bin": ex.get("metadata_overlap_bin", "mid"),
                "metadata_genre": ex.get("metadata_genre", ""),
                "metadata_source_dataset": ex.get("metadata_source_dataset", ""),
                "metadata_year": ex.get("metadata_year", ""),
                "metadata_sid": ex.get("metadata_sid", ""),
                "metadata_task_type": ex.get("metadata_task_type", "regression"),
                "metadata_is_holdout": ex.get("metadata_is_holdout", False),
                "metadata_baseline_cosine": round(float(base_arr[i]), 6),
                "metadata_tuned_cosine": round(float(tuned_arr[i]), 6),
                "metadata_gold_score": ex["metadata_score"],
            }
            examples_out.append(record)

    # Top-level metadata with aggregate results
    metadata_out = {
        "title": "STS-B Fine-Tuning: Mid-Band Rank Compression Test",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "data_source": str(DATA_PATH),
        "train_examples": len(train_examples),
        "val_examples": len(val_eval),
        "test_examples": len(test_examples),
        "seeds": SEEDS,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "lr": LR,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "device": "cpu",
        "baseline_metrics": {
            "spearman": primary["baseline"]["spearman"] if primary else None,
            "pearson": primary["baseline"]["pearson"] if primary else None,
            "mae": primary["baseline"]["mae"] if primary else None,
        },
        "post_tuning_metrics": {
            "spearman": primary["post_tuning"]["spearman"] if primary else None,
            "pearson": primary["post_tuning"]["pearson"] if primary else None,
            "mae": primary["post_tuning"]["mae"] if primary else None,
            "spearman_delta": primary["post_tuning"]["spearman_delta"] if primary else None,
            "pearson_delta": primary["post_tuning"]["pearson_delta"] if primary else None,
        },
        "diagnostics": primary_diagnostics,
        "seed_variance": {
            "seeds_run": [r["seed"] for r in all_results],
            "spearman_deltas": deltas,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "per_seed_results": [
                {
                    "seed": r["seed"],
                    "baseline_spearman": r["baseline"]["spearman"],
                    "tuned_spearman": r["post_tuning"]["spearman"],
                    "delta": r["post_tuning"]["spearman_delta"],
                    "wall_clock_seconds": r["wall_clock_seconds"],
                    "peak_memory_mb": r["peak_memory_mb"],
                }
                for r in all_results
            ],
        },
        "resource_usage": {
            "total_wall_clock_seconds": total_wall,
            "peak_memory_mb": peak_mem,
            "per_seed_times": [r["wall_clock_seconds"] for r in all_results],
        },
        "model_path": str(CHECKPOINT_DIR),
        "hypothesis_assessment": {
            "spearman_improved": spearman_improved,
            "delta_exceeds_2x_seed_sd": delta_exceeds,
            "mid_band_contribution_fraction": mid_frac,
            "embedding_drift_below_threshold": drift_ok,
            "iso_fraction_below_threshold": iso_ok,
            "mid_band_compression_supported": mid_supported,
        },
    }

    output = {
        "metadata": metadata_out,
        "datasets": [
            {
                "dataset": "stsbenchmark",
                "examples": examples_out,
            }
        ],
    }

    logger.info(f"Saving results to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    bl_s = output["metadata"]["baseline_metrics"]["spearman"]
    pt_s = output["metadata"]["post_tuning_metrics"]["spearman"]
    pt_d = output["metadata"]["post_tuning_metrics"]["spearman_delta"]
    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Total wall-clock time: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    logger.info(f"Peak memory: {peak_mem:.1f} MB")
    if bl_s is not None and pt_s is not None:
        logger.info(f"Primary seed Spearman: {bl_s:.4f} -> {pt_s:.4f} (d={pt_d:+.4f})")
    logger.info(f"Mean delta across seeds: {mean_delta:+.4f} +/- {std_delta:.4f}")
    logger.info(f"Hypothesis assessment: {output['metadata']['hypothesis_assessment']}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
