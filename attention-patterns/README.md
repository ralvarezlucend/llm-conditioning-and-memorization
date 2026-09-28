# Attention-pattern analysis

Attention diagnostics for *Conditioning Structure, Not Entropy Alone, Predicts Memorization Dynamics*. The utilities load trained GPT-2 checkpoints as TransformerLens `HookedTransformer`s, average attention over a dataset, and plot how suffix queries attend to prefix keys. The paper uses these patterns to locate planted cues; causal validation of their role in memorization remains future work.

## Notebooks

- [one-layer-attention.ipynb](one-layer-attention.ipynb): fixed and position-dependent cues in the conditioning hierarchy.
- [karity-fixed-random-attention.ipynb](karity-fixed-random-attention.ipynb): K-arity cue retrieval across training checkpoints.
- [natural-language.ipynb](natural-language.ipynb): attention diagnostics for WikiText interventions.

## Running

Training writes checkpoints and metadata under `saved_models/<model_name>/`. Some notebooks still import `interpretability.utils`, the directory's previous name. In a fresh notebook kernel whose working directory is `attention-patterns/`, replace the old import cell, including its parent-directory `sys.path` setup, with local imports such as those below to use [utils.py](utils.py).

For a synthetic NPZ dataset, load the files directly:

```python
from utils import load_model, load_dataset, compute_avg_attn, plot_attention

model = load_model("../saved_models/<model_name>")
input_ids = load_dataset("../datasets/<dataset_name>.npz")
avg_attn = compute_avg_attn(model, input_ids)  # [n_layers, n_heads, seq, seq]
plot_attention(avg_attn, query_range=(50, 100), key_range=(0, 50), reduce="max")
```

The `get_attention` convenience function reads `metadata.json`, but its `MODEL_BASE_PATH` currently points to the original cluster's storage. Set that path to your checkout's `saved_models/` before using it. Each metadata file must point to a valid `model_path` and either `dataset_path` (NPZ with a `dataset` array) or `dataset.tokenized_data_path` (Hugging Face dataset with `input_ids`). Update absolute paths in metadata if checkpoints were copied from another machine.
