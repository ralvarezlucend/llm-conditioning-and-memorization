# z-complexity

Tools for measuring z-complexity (zlib compression ratio as a proxy for Kolmogorov complexity) of token sequences and building complexity-bucketed datasets.

## Files

- **`z-comp-dataset.py`** — Tokenizes sequences from Wikitext or The Pile (100k examples, `SEQ_LEN=100`), computes the z-complexity of each, and buckets them by ratio (`[0.0, 0.5)`, `[0.5, 0.8)`, `[0.8, 1.0)`). Samples 5000 sequences per bucket and saves each bucket as a `.npz` file in `../datasets/` with metadata (vocab size, seed, per-sequence and concatenated z-complexity).

- **`z-comp-plot.py`** — Computes z-complexity for both Wikitext and Pile and saves an overlapping histogram of the two distributions to `z-comp-histogram.png`.

- **`z-comp-histogram.png`** — Output of `z-comp-plot.py`.

## z-complexity

Each token sequence is packed into a byte buffer (`uint8`/`uint16`/`uint32` depending on vocab size) and compressed with raw DEFLATE (`level=9`). The ratio `len(compressed) / len(raw)` is the z-complexity — lower means more compressible (more structured).
