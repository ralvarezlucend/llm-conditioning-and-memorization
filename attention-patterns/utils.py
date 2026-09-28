import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.loading_from_pretrained import convert_gpt2_weights
import torch
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_path: str):
    """Load a saved GPT-2 model as a TransformerLens HookedTransformer."""
    hf_model = GPT2LMHeadModel.from_pretrained(model_path)
    config = hf_model.config

    hooked_cfg = HookedTransformerConfig(
        n_layers=config.n_layer,
        d_model=config.n_embd,
        d_head=config.n_embd // config.n_head,
        n_heads=config.n_head,
        d_mlp=config.n_inner if config.n_inner else 4 * config.n_embd,
        d_vocab=config.vocab_size,
        n_ctx=config.n_positions,
        act_fn=config.activation_function,
        normalization_type="LNPre",  # fold_ln folds LN w/b away; LNPre has no learnable w/b
    )

    hooked_model = HookedTransformer(hooked_cfg)
    tl_state_dict = convert_gpt2_weights(hf_model, hooked_cfg)
    hooked_model.load_and_process_state_dict(
        tl_state_dict,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        refactor_factored_attn_matrices=False,
    )
    return hooked_model

def load_dataset(name_or_path: str):
    data = np.load(name_or_path, allow_pickle=True)
    input_ids = torch.from_numpy(data["dataset"].astype(np.int64))
    return input_ids

def compute_avg_attn(model, input_ids, batch_size=64):
    """Compute average attention patterns across all sequences.

    Returns:
        avg_attn_all: [n_layers, n_heads, seq_len, seq_len]
    """
    n_layers = model.cfg.n_layers
    n_seqs = input_ids.shape[0]
    avg_attn_all = None

    model.eval()
    with torch.no_grad():
        for start in range(0, n_seqs, batch_size):
            batch = input_ids[start : start + batch_size]
            _, cache = model.run_with_cache(batch)

            layer_sums = torch.stack([
                cache["pattern", layer, "attn"].sum(dim=0)  # [n_heads, seq, seq]
                for layer in range(n_layers)
            ])  # [n_layers, n_heads, seq, seq]

            avg_attn_all = layer_sums if avg_attn_all is None else avg_attn_all + layer_sums

    return avg_attn_all / n_seqs  # [n_layers, n_heads, seq, seq]

def plot_attention(
    avg_attn_all,
    layers=None,
    query_range=(50, 100),
    key_range=(0, 50),
    tick_step=5,
    cmap="viridis",
    reduce="max",
    save_name=None,
):
    """
    Side-by-side layer heatmaps aggregated across heads.
    One shared y-label and one colorbar.
    """

    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def agg_heads(x, mode):
        if mode == "mean":   return x.mean(axis=0)
        if mode == "median": return np.median(x, axis=0)
        if mode == "max":    return x.max(axis=0)
        if mode == "normalized_sum":
            s = x.sum(axis=0)
            return s / s.max()
        raise ValueError('reduce must be "mean", "median", "max", or "normalized_sum"')

    n_layers, n_heads = avg_attn_all.shape[:2]
    q0, q1 = query_range
    k0, k1 = key_range

    if layers is None:
        layers = list(range(n_layers))
    elif isinstance(layers, int):
        layers = [layers]

    # Aggregate per layer
    imgs = []
    for L in layers:
        head_stack = np.stack(
            [to_np(avg_attn_all[L, h, q0:q1, k0:k1]) for h in range(n_heads)],
            axis=0,
        )
        imgs.append(agg_heads(head_stack, reduce))

    # Ticks
    q_ticks = list(range(0, q1 - q0, tick_step))
    k_ticks = list(range(0, k1 - k0, tick_step))
    q_labels = [str(q0 + i) for i in q_ticks]
    k_labels = [str(k0 + i) for i in k_ticks]

    ncols = len(layers)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(4.2 * ncols, 4.6),
        sharey=True,
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]

    for ax, L, A in zip(axes, layers, imgs):
        im = ax.imshow(A, cmap=cmap, aspect="equal")  # ← no vmin/vmax

        ax.set_title(f"Layer {L}", fontsize=20, fontweight="bold")
        ax.set_xlabel("Key (Prefix)", fontsize=18)

        ax.set_xticks(k_ticks)
        ax.set_xticklabels(k_labels, fontsize=14, rotation=45, ha="right")

        ax.set_yticks(q_ticks)
        ax.set_yticklabels(q_labels, fontsize=14)

    fig.supylabel("Query (Suffix)", fontsize=18)

    # One shared colorbar (uses last im)
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.ax.tick_params(labelsize=14)

    if save_name:
        outdir = Path("plots")
        outdir.mkdir(exist_ok=True)
        fig.savefig(outdir / save_name, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
    else:
        plt.show()

def plot_attention_per_head(
    avg_attn_all,
    layers=None,
    query_range=(50, 100),
    key_range=(0, 50),
    tick_step=5,
    cmap="viridis",
    save_name=None,
):
    """
    Grid of attention heatmaps: rows = layers, cols = heads.
    Each cell shows one head's attention (no aggregation).
    """
    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    n_layers_total, n_heads = avg_attn_all.shape[:2]
    q0, q1 = query_range
    k0, k1 = key_range

    if layers is None:
        layers = list(range(n_layers_total))
    elif isinstance(layers, int):
        layers = [layers]

    imgs = [
        [to_np(avg_attn_all[L, h, q0:q1, k0:k1]) for h in range(n_heads)]
        for L in layers
    ]

    q_ticks = list(range(0, q1 - q0, tick_step))
    k_ticks = list(range(0, k1 - k0, tick_step))
    q_labels = [str(q0 + i) for i in q_ticks]
    k_labels = [str(k0 + i) for i in k_ticks]

    nrows = len(layers)
    ncols = n_heads
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.8 * ncols, 4.2 * nrows),
        sharey="row",
        constrained_layout=True,
    )
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [list(axes)]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, L in enumerate(layers):
        for h in range(n_heads):
            ax = axes[row_idx][h]
            im = ax.imshow(imgs[row_idx][h], cmap=cmap, aspect="equal")

            if row_idx == 0:
                ax.set_title(f"Head {h}", fontsize=16, fontweight="bold")
            if h == 0:
                ax.set_ylabel(f"Layer {L}", fontsize=14)
            if row_idx == nrows - 1:
                ax.set_xlabel("Key (Prefix)", fontsize=12)

            ax.set_xticks(k_ticks)
            ax.set_xticklabels(k_labels if row_idx == nrows - 1 else [], fontsize=10, rotation=45, ha="right")
            ax.set_yticks(q_ticks)
            ax.set_yticklabels(q_labels, fontsize=10)

    fig.colorbar(im, ax=[axes[r][c] for r in range(nrows) for c in range(ncols)], shrink=0.5, pad=0.02)

    if save_name:
        outdir = Path("plots")
        outdir.mkdir(exist_ok=True)
        fig.savefig(outdir / save_name, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
    else:
        plt.show()


def plot_levels_comparison(
    attn_dict,
    layer=0,
    query_range=(50, 100),
    key_range=(0, 50),
    tick_step=5,
    cmap="viridis",
    reduce="max",
    save_name=None,
):
    """
    Side-by-side level heatmaps aggregated across heads at a fixed layer.
    Each subplot is one level/condition from attn_dict.

    Args:
        attn_dict:   {level_name: avg_attn [n_layers, n_heads, seq, seq]}
        layer:       which layer to plot for all levels
        query_range: (start, end) token positions for query axis (rows)
        key_range:   (start, end) token positions for key axis (cols)
        tick_step:   spacing between tick labels
        cmap:        matplotlib colormap name
        reduce:      how to aggregate heads — "max", "mean", or "median"
        save_name:   if given, save to plots/<save_name> and close; else show
    """

    def to_np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    def agg_heads(x, mode):
        if mode == "mean":   return x.mean(axis=0)
        if mode == "median": return np.median(x, axis=0)
        if mode == "max":    return x.max(axis=0)
        raise ValueError('reduce must be "mean", "median", or "max"')

    q0, q1 = query_range
    k0, k1 = key_range

    level_names = list(attn_dict.keys())
    imgs = []
    for name in level_names:
        avg_attn = attn_dict[name]
        n_heads = avg_attn.shape[1]
        head_stack = np.stack(
            [to_np(avg_attn[layer, h, q0:q1, k0:k1]) for h in range(n_heads)],
            axis=0,
        )
        imgs.append(agg_heads(head_stack, reduce))

    q_ticks  = list(range(0, q1 - q0, tick_step))
    k_ticks  = list(range(0, k1 - k0, tick_step))
    q_labels = [str(q0 + i) for i in q_ticks]
    k_labels = [str(k0 + i) for i in k_ticks]

    ncols = len(level_names)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(4.2 * ncols, 4.6),
        sharey=True,
        constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]

    for ax, name, A in zip(axes, level_names, imgs):
        im = ax.imshow(A, cmap=cmap, aspect="equal")

        ax.set_title(name, fontsize=20, fontweight="bold")
        ax.set_xlabel("Key (Prefix)", fontsize=18)

        ax.set_xticks(k_ticks)
        ax.set_xticklabels(k_labels, fontsize=14, rotation=45, ha="right")

        ax.set_yticks(q_ticks)
        ax.set_yticklabels(q_labels, fontsize=14)

    fig.supylabel("Query (Suffix)", fontsize=18)

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.ax.tick_params(labelsize=14)

    if save_name:
        outdir = Path("plots")
        outdir.mkdir(exist_ok=True)
        fig.savefig(outdir / save_name, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
    else:
        plt.show()


def load_lut_positions(npz_path: str) -> np.ndarray:
    """Load lut_positions from an NPZ file. Shape: (K, suffix_len)."""
    data = np.load(npz_path, allow_pickle=True)
    return data["lut_positions"]


def compute_lut_attn_accuracy(model, input_ids, lut_positions, prefix_len, batch_size=64, top_ns=[1, 3, 5]):
    """Compute how well model attention top-N positions cover ground-truth LUT positions.

    A position (seq, suffix_pos) is a hit under top-N if, for every position p in the
    ground-truth set, there exists at least one (layer, head) pair whose top-N attended
    prefix positions include p. Equivalently: the union of top-N prefix positions across
    all (layer, head) pairs must be a superset of the ground-truth set (distributed
    criterion). Larger N → larger coverage set → higher or equal accuracy.

    Args:
        model: HookedTransformer model.
        input_ids: [n_sequences, seq_len] int64 tensor.
        lut_positions: [K, suffix_len] int64 array of ground-truth prefix positions.
        prefix_len: number of prefix tokens.
        batch_size: sequences per forward pass.
        top_ns: list of N values for the top-N distributed criterion (default [1, 3, 5]).

    Returns dict with:
        overall_accuracy: scalar float — top-1 accuracy (backward compatible)
        per_seq_accuracy: [n_sequences] float array — top-1 (backward compatible)
        per_pos_accuracy: [suffix_len] float array — top-1 (backward compatible)
        overall_accuracy_top{N}: scalar float for each N in top_ns
        per_seq_accuracy_top{N}: [n_sequences] float array for each N in top_ns
        per_pos_accuracy_top{N}: [suffix_len] float array for each N in top_ns
    """
    lut_positions = np.asarray(lut_positions)
    K, suffix_len = lut_positions.shape
    n_sequences = input_ids.shape[0]

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    max_n = max(top_ns)

    # hits_by_n[n][seq, suffix_pos] = 1 if covered[b] ⊇ gt_set for seq b under top-n
    hits_by_n = {n: np.zeros((n_sequences, suffix_len), dtype=np.float32) for n in top_ns}

    model.eval()
    with torch.no_grad():
        for start in range(0, n_sequences, batch_size):
            end = min(start + batch_size, n_sequences)
            batch = input_ids[start:end]  # [B, seq_len]
            B = batch.shape[0]

            _, cache = model.run_with_cache(batch)

            for t_rel in range(suffix_len):
                t_abs = prefix_len + t_rel
                gt_set = set(int(lut_positions[k, t_rel]) for k in range(K))

                covered = {n: [set() for _ in range(B)] for n in top_ns}

                for layer in range(n_layers):
                    layer_attn_np = cache["pattern", layer, "attn"].detach().cpu().numpy()
                    for head in range(n_heads):
                        attn_row = layer_attn_np[:, head, t_abs, :prefix_len]  # [B, prefix_len]
                        for b in range(B):
                            row = attn_row[b]
                            for n in top_ns:
                                if n == 1:
                                    covered[1][b].add(int(np.argmax(row)))
                                else:
                                    idxs = np.argpartition(row, -n)[-n:]
                                    covered[n][b].update(int(i) for i in idxs)

                for n in top_ns:
                    for b in range(B):
                        hits_by_n[n][start + b, t_rel] = float(gt_set.issubset(covered[n][b]))

    first_n = top_ns[0]
    result = dict(
        overall_accuracy=float(hits_by_n[first_n].mean()),
        per_seq_accuracy=hits_by_n[first_n].mean(axis=1),
        per_pos_accuracy=hits_by_n[first_n].mean(axis=0),
    )
    for n in top_ns:
        result[f"overall_accuracy_top{n}"] = float(hits_by_n[n].mean())
        result[f"per_seq_accuracy_top{n}"] = hits_by_n[n].mean(axis=1)
        result[f"per_pos_accuracy_top{n}"] = hits_by_n[n].mean(axis=0)
    return result


def get_attention(model_name: str):
    MODEL_BASE_PATH = Path("/nlpscratch/home/ralvarez/memorization-llms/saved_models")
    metadata_path = MODEL_BASE_PATH / model_name / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    model_path = metadata["model_path"]
    dataset_path = metadata["dataset_path"]

    model = load_model(model_path)

    if dataset_path is not None:
        input_ids = load_dataset(dataset_path)
    else:
        tokenized_data_path = metadata.get("dataset", {}).get("tokenized_data_path")
        if tokenized_data_path is None:
            raise ValueError(
                f"metadata.json for '{model_name}' has no dataset_path and no "
                "dataset.tokenized_data_path fallback."
            )
        from datasets import load_from_disk
        project_root = MODEL_BASE_PATH.parent
        hf_dataset = load_from_disk(str(project_root / tokenized_data_path))
        input_ids = torch.tensor(hf_dataset["input_ids"], dtype=torch.int64)

    avg_attn_all = compute_avg_attn(model, input_ids)

    return avg_attn_all