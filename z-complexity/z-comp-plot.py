import zlib
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
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
SEQ_LEN = 100
PILE_NUM_EXAMPLES = 100_000
NUM_WORKERS = 8

WIKITEXT_PATH = "/Users/rodrigoalvarezlucendo/Projects/conditional_entropy/natural_language/wikitext-tokenized/"


if __name__ == "__main__":
    # ── Wikitext (already tokenized, fixed-length sequences) ───────────
    print("Loading Wikitext...")
    wikitext_ds = load_from_disk(WIKITEXT_PATH)
    wikitext_seqs = np.array(wikitext_ds["input_ids"])
    wikitext_vocab = int(wikitext_seqs.max()) + 1

    print("Computing z-complexity for Wikitext...")
    _zc_wiki = partial(z_complexity_sequence, vocab_size=wikitext_vocab)
    with Pool(NUM_WORKERS) as pool:
        wikitext_z = np.array(list(tqdm.tqdm(
            pool.imap(_zc_wiki, wikitext_seqs, chunksize=1024),
            total=len(wikitext_seqs), desc="Wikitext",
        )))

    # ── Pile (stream, tokenize, truncate to SEQ_LEN) ──────────────────
    print("Loading & tokenizing Pile...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    pile_vocab = tokenizer.vocab_size
    pile_ds = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)

    pile_seqs = []
    for example in tqdm.tqdm(pile_ds, desc="Pile", total=PILE_NUM_EXAMPLES):
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)[:SEQ_LEN]
        if len(tokens) == SEQ_LEN:
            pile_seqs.append(tokens)
        if len(pile_seqs) >= PILE_NUM_EXAMPLES:
            break
    pile_seqs = np.array(pile_seqs)

    print("Computing z-complexity for Pile...")
    _zc_pile = partial(z_complexity_sequence, vocab_size=pile_vocab)
    with Pool(NUM_WORKERS) as pool:
        pile_z = np.array(list(tqdm.tqdm(
            pool.imap(_zc_pile, pile_seqs, chunksize=1024),
            total=len(pile_seqs), desc="Pile",
        )))

    # ── Overlapping histogram ──────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.hist(wikitext_z, bins=100, alpha=0.5, label="Wikitext", color="steelblue", density=True)
    plt.hist(pile_z, bins=100, alpha=0.5, label="Pile", color="coral", density=True)
    plt.xlabel("Z-complexity")
    plt.ylabel("Density")
    plt.title("Z-complexity distribution: Wikitext vs Pile")
    plt.legend()
    plt.savefig("z-comp-histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved histogram to z-comp-histogram.png")
