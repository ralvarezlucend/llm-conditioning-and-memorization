import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tqdm
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def z_complexity_sequence(tokens_1d, vocab_size, level=9, raw_deflate=True):
    """Compression ratio (compressed / original) as a proxy for Kolmogorov complexity."""
    if vocab_size <= 256:
        dtype = np.uint8
    elif vocab_size <= 65536:
        dtype = np.uint16
    else:
        dtype = np.uint32
    raw = tokens_1d.astype(dtype).tobytes()
    wbits = -15 if raw_deflate else 15
    co = zlib.compressobj(level=level, wbits=wbits)
    compressed = co.compress(raw) + co.flush()
    return len(compressed) / len(raw)


# ── Constants ──────────────────────────────────────────────────────────
DATASET = "pile"  # "wikitext" or "pile"
SEQ_LEN = 100
PILE_NUM_EXAMPLES = 100_000
BUCKET_EDGES = [0.0, 0.5, 0.8, 1.0]
NUM_SAMPLES = 5000
SEED = 42
NUM_WORKERS = 8

WIKITEXT_PATH = "/Users/rodrigoalvarezlucendo/Projects/conditional_entropy/natural_language/wikitext-tokenized/"


if __name__ == "__main__":
    # ── Load chosen dataset ────────────────────────────────────────────
    if DATASET == "wikitext":
        print("Loading Wikitext...")
        ds = load_from_disk(WIKITEXT_PATH)
        sequences = np.array(ds["input_ids"])
        vocab_size = int(sequences.max()) + 1
        label = "wikitext"
    else:
        print("Loading & tokenizing Pile...")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        vocab_size = tokenizer.vocab_size
        pile_ds = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)

        seqs = []
        for example in tqdm.tqdm(pile_ds, desc="Pile", total=PILE_NUM_EXAMPLES):
            tokens = tokenizer.encode(example["text"], add_special_tokens=False)[:SEQ_LEN]
            if len(tokens) == SEQ_LEN:
                seqs.append(tokens)
            if len(seqs) >= PILE_NUM_EXAMPLES:
                break
        sequences = np.array(seqs)
        label = "pile"

    # ── Compute z-complexity ───────────────────────────────────────────
    print(f"Computing z-complexity for {label}...")
    _zc = partial(z_complexity_sequence, vocab_size=vocab_size)
    with Pool(NUM_WORKERS) as pool:
        z_ratios = np.array(list(tqdm.tqdm(
            pool.imap(_zc, sequences, chunksize=1024),
            total=len(sequences), desc=label,
        )))

    # ── Bucket & sample ────────────────────────────────────────────────
    print(f"\nBucketing & sampling from: {label} (vocab_size={vocab_size})")

    rng = np.random.default_rng(SEED)
    out_dir = Path("../datasets")
    out_dir.mkdir(exist_ok=True)

    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        mask = (z_ratios >= lo) & (z_ratios < hi)
        indices = np.where(mask)[0]

        if len(indices) < NUM_SAMPLES:
            print(f"[{lo}, {hi}): only {len(indices)} sequences, skipping (need {NUM_SAMPLES})")
            continue

        chosen = rng.choice(indices, size=NUM_SAMPLES, replace=False)
        sampled = sequences[chosen]

        z_complexity_dataset = z_complexity_sequence(sampled.flatten(), vocab_size)
        z_complexity_avg = float(np.mean(z_ratios[chosen]))

        print(f"[{lo}, {hi}): sampled {len(sampled)} sequences")
        print(f"  z_complexity_dataset (concatenated): {z_complexity_dataset:.6f}")
        print(f"  z_complexity_avg (mean per-sequence): {z_complexity_avg:.6f}")

        metadata = {
            "num_samples": len(sampled),
            "seq_len": sampled.shape[1],
            "vocab_size": vocab_size,
            "seed": SEED,
            "prefix_len": 50,
            "target_level": None,
            "random_prefix": None,
            "strategy": None,
            "aggregation": None,
            "pos": None,
            "lookback": None,
            "z_complexity_range": (lo, hi),
            "z_complexity_dataset": z_complexity_dataset,
            "z_complexity_avg": z_complexity_avg,
        }

        fname = f"{label}_zcomp_{lo:.1f}_{hi:.1f}_{NUM_SAMPLES}.npz"
        path = out_dir / fname
        np.savez(path, dataset=sampled, metadata=metadata)
        print(f"  Saved {path} — shape {sampled.shape}")
