"""Helper module for the natural-language experiment summary notebook.

Keeps all wandb fetching, caching, YAML parsing and plotting out of the summary
notebook so the notebook itself stays narrative + a few one-line calls.

Usage (from analysis/nl_summary.ipynb):
    import nl_plots as nlp
    runs = nlp.load_runs()
    hist = nlp.load_histories(runs)
    nlp.plot_family_curves(runs, hist)
    nlp.steps_to_threshold_table(runs, hist)
    nlp.config_table()
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Larger, more readable text everywhere (applied on import).
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "legend.title_fontsize": 13,
    "figure.titlesize": 18,
})

# --- Paths (anchored to this file so cwd does not matter; reuses existing cache) ---
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CACHE_DIR = HERE / ".wandb_cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- wandb coordinates ---
ENTITY = "r-alvarezlucendo16"
PROJECTS = ["memorization", "llm-memorization"]
STEP_KEY = "train/global_step"
METRIC = "train/exact_match_total"  # HF Trainer prefixes custom metrics with 'train/'

# tokenized_data_path -> (family, pretty label)
NL_DATASETS = {
    "wikitext_nl":                         ("prefix-structure", "WikiText (orig prefix)"),
    "wikitext_translated":                 ("prefix-structure", "Translated prefix"),
    "wikitext_translated_uid":             ("prefix-structure", "Translated + UID (pos0)"),
    "wikitext_translated_uid5":            ("prefix-structure", "Translated + UID (pos5)"),
    "wikitext_paraphrase":                 ("paraphrase",        "Paraphrase (9x/seq)"),
    "wikitext_paraphrase_control":         ("paraphrase",        "Paraphrase control (indep)"),
    "wikitext_prefix_paraphrase":          ("prefix-paraphrase", "Prefix-paraphrase"),
    "wikitext_prefix_paraphrase_control":  ("prefix-paraphrase", "Prefix-paraphrase control"),
    "wikitext_translation_latin":          ("translation",       "Translation Latin"),
    "wikitext_translation_diverse":        ("translation",       "Translation diverse"),
    "wikitext_multilingual_latin":         ("multilingual",      "Multilingual Latin"),
    "wikitext_multilingual_diverse":       ("multilingual",      "Multilingual diverse"),
}

# Maps the NL experiment configs to their datasets (for the config table).
NL_EXPERIMENTS = [
    "natural-language", "translation", "multilingual",
    "paraphrase", "prefix-paraphrase",
]


# --------------------------------------------------------------------------- #
# wandb loading (server-side prefilter + on-disk cache)
# --------------------------------------------------------------------------- #
def _api(timeout: int = 60):
    import wandb
    return wandb.Api(timeout=timeout)


def _dataset_size(logged, final_step, batch_size, epoch) -> float | None:
    """Number of training sequences for a run.

    wandb only logs ``dataset_metadata.num_samples`` for a handful of runs, so
    when it is missing we recover the (effective) size from the HF Trainer's
    epoch counter: ``size = batch_size * steps_per_epoch = batch_size *
    global_step / epoch``. This is exact up to +-batch_size, since the Trainer
    counts epochs in whole optimizer steps. Note the size is a per-run swept
    variable here, not a constant per dataset.
    """
    if logged is not None:
        return float(logged)
    if final_step and epoch and batch_size:
        return round(final_step * batch_size / epoch)
    return None


def load_runs(force_refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame of NL runs (one row per run), cached to disk."""
    cache = CACHE_DIR / "runs_df.pkl"
    if cache.exists() and not force_refresh:
        return pd.read_pickle(cache)

    api = _api()
    nl_filter = {"config.dataset.tokenized_data_path": {"$in": list(NL_DATASETS)}}
    rows = []
    for proj in PROJECTS:
        try:
            runs = list(api.runs(f"{ENTITY}/{proj}", filters=nl_filter))
        except Exception as e:  # noqa: BLE001
            print(f"skip {proj}: {e}")
            continue
        print(f"{proj}: {len(runs)} NL runs (server-filtered)")
        for r in runs:
            cfg = dict(r.config)
            ds = cfg.get("dataset", {}) or {}
            path = ds.get("tokenized_data_path")
            if path not in NL_DATASETS:
                continue
            fam, label = NL_DATASETS[path]
            tr = cfg.get("training", {}) or {}
            md = cfg.get("model", {}) or {}
            mdata = cfg.get("dataset_metadata", {}) or {}
            summ = dict(r.summary)
            bs = tr.get("batch_size")
            final_step = summ.get(STEP_KEY)
            logged_ns = mdata.get("num_samples")
            rows.append({
                "project": proj, "run_id": r.id, "name": r.name, "state": r.state,
                "family": fam, "dataset_label": label, "tokenized_data_path": path,
                "tags": ",".join(r.tags),
                "model_layers": md.get("n_layer"), "n_embd": md.get("n_embd"),
                "vocab_size": mdata.get("vocab_size"), "num_samples": logged_ns,
                "dataset_size": _dataset_size(logged_ns, final_step, bs, summ.get("train/epoch")),
                "max_steps": tr.get("max_steps"), "batch_size": bs,
                "lr": tr.get("learning_rate"), "seed": tr.get("seed"),
                "final_exact_match": summ.get(METRIC), "final_step": final_step,
            })
    runs_df = (pd.DataFrame(rows)
               .sort_values(["family", "dataset_label", "seed"])
               .reset_index(drop=True))
    runs_df.to_pickle(cache)
    print(f"cached {len(runs_df)} runs -> {cache}")
    return runs_df


def _fetch_one(rec, force_refresh: bool):
    run_id, project, state = rec["run_id"], rec["project"], rec["state"]
    cache_f = CACHE_DIR / f"hist__{project}__{run_id}.pkl"
    if cache_f.exists() and not force_refresh and state == "finished":
        return run_id, pd.read_pickle(cache_f)
    try:
        r = _api().run(f"{ENTITY}/{project}/{run_id}")
        h = pd.DataFrame(list(r.scan_history(keys=[METRIC, STEP_KEY])))
    except Exception as e:  # noqa: BLE001  (e.g. no API key, crashed/deleted run)
        print(f"  could not fetch {project}/{run_id} ({state}): {e}")
        return run_id, pd.DataFrame()
    if state == "finished" and not h.empty:
        h.to_pickle(cache_f)
    return run_id, h


def load_histories(runs_df: pd.DataFrame, force_refresh: bool = False,
                   max_workers: int = 16) -> dict[str, pd.DataFrame]:
    """Return {run_id: history_df}; finished runs cached permanently, threaded."""
    recs = runs_df[["run_id", "project", "state"]].to_dict("records")
    hist: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fetch_one, rec, force_refresh) for rec in recs]
        for fut in as_completed(futures):
            rid, df = fut.result()
            if not df.empty:
                hist[rid] = df
    print(f"histories ready: {len(hist)} / {len(recs)}  (cache: {CACHE_DIR})")
    return hist


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _filter_family(df: pd.DataFrame, family: str | None) -> pd.DataFrame:
    return df if family is None else df[df["family"] == family]


def summary_table(runs_df: pd.DataFrame, family: str | None = None) -> pd.DataFrame:
    """Final exact-match per dataset, grouped by run shape so unlike sweeps don't mix.

    `size_min`/`size_max` give the per-run training-set size (see `_dataset_size`);
    a wide spread means the group mixes runs trained on very different numbers of
    sequences, so compare its memorization speed with care.
    """
    df = _filter_family(runs_df, family)
    return (df
            .groupby(["family", "dataset_label", "max_steps", "batch_size"])
            .agg(n_runs=("run_id", "count"),
                 layers=("model_layers", "first"), n_embd=("n_embd", "first"),
                 size_min=("dataset_size", "min"), size_max=("dataset_size", "max"),
                 final_em_mean=("final_exact_match", "mean"),
                 final_em_min=("final_exact_match", "min"),
                 final_em_max=("final_exact_match", "max"))
            .round(4).reset_index())


def _steps_to(df: pd.DataFrame, thresh: float) -> float:
    d = df.dropna(subset=[METRIC]).sort_values(STEP_KEY)
    hit = d[d[METRIC] >= thresh]
    return int(hit[STEP_KEY].iloc[0]) if len(hit) else np.nan


def steps_to_threshold_table(runs_df: pd.DataFrame, hist: dict,
                             thresh: float = 0.5, family: str | None = None) -> pd.DataFrame:
    """Mean steps to reach `thresh` exact-match, per dataset (lower = memorized faster)."""
    df = _filter_family(runs_df, family).copy()
    col = f"steps_to_{thresh}"
    df[col] = df["run_id"].map(lambda r: _steps_to(hist[r], thresh) if r in hist else np.nan)
    out = (df.groupby(["family", "dataset_label"])[col]
           .mean().round(0).reset_index())
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_family_curves(runs_df: pd.DataFrame, hist: dict, save: str | None = None):
    """One subplot per family; overlay datasets (averaging seeds where present)."""
    families = list(dict.fromkeys(runs_df["family"]))
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, fam in zip(axes, families):
        sub = runs_df[runs_df["family"] == fam]
        for label, grp in sub.groupby("dataset_label"):
            parts = [hist[rid][[STEP_KEY, METRIC]].dropna()
                     for rid in grp["run_id"] if rid in hist]
            if not parts:
                continue
            # average across runs/seeds at each step (robust to duplicate steps)
            curve = (pd.concat(parts, ignore_index=True)
                     .groupby(STEP_KEY)[METRIC].mean().sort_index())
            ax.plot(curve.index, curve.values, label=label, linewidth=2.5)
        ax.set_title(fam)
        ax.set_xlabel("train step")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("exact match (total)")
    plt.suptitle("NL memorization curves by family", y=1.03)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_family(runs_df: pd.DataFrame, hist: dict, family: str,
                thresh: float = 0.5, save: str | None = None):
    """Isolated exact-match curves for a single family (one panel per section)."""
    sub = runs_df[runs_df["family"] == family]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, grp in sub.groupby("dataset_label"):
        parts = [hist[rid][[STEP_KEY, METRIC]].dropna()
                 for rid in grp["run_id"] if rid in hist]
        if not parts:
            continue
        curve = (pd.concat(parts, ignore_index=True)
                 .groupby(STEP_KEY)[METRIC].mean().sort_index())
        ax.plot(curve.index, curve.values, label=label, linewidth=2.5)
    ax.axhline(thresh, ls="--", c="grey", lw=1.5, alpha=0.6)
    ax.set_title(f"{family}: memorization curves")
    ax.set_xlabel("train step")
    ax.set_ylabel("exact match (total)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_steps_to_threshold(runs_df: pd.DataFrame, hist: dict,
                            thresh: float = 0.5, save: str | None = None):
    """Horizontal bar chart of mean steps-to-threshold, grouped/colored by family."""
    tbl = steps_to_threshold_table(runs_df, hist, thresh)
    col = f"steps_to_{thresh}"
    tbl = tbl.dropna(subset=[col]).sort_values(["family", col])
    fams = list(dict.fromkeys(tbl["family"]))
    cmap = {f: c for f, c in zip(fams, plt.cm.tab10.colors)}
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(tbl) + 1.5))
    ypos = range(len(tbl))
    ax.barh(list(ypos), tbl[col], color=[cmap[f] for f in tbl["family"]])
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([f"{r.dataset_label}" for r in tbl.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel(f"mean steps to {thresh:.0%} exact match  (lower = faster)")
    ax.set_title("How fast each NL condition is memorized")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[f]) for f in fams]
    ax.legend(handles, fams, title="family")
    for i, v in zip(ypos, tbl[col]):
        ax.text(v + 5, i, f"{int(v)}", va="center", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Config info read from the YAML files (ground truth for intended setup)
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def config_table(family: str | None = None) -> pd.DataFrame:
    """Read conf/experiment/*.yaml + conf/dataset/*.yaml into a per-dataset table."""
    exp_dir = REPO_ROOT / "conf" / "experiment"
    ds_dir = REPO_ROOT / "conf" / "dataset"
    rows = []
    for exp in NL_EXPERIMENTS:
        cfg = _load_yaml(exp_dir / f"{exp}.yaml")
        training = cfg.get("training", {}) or {}
        params = (cfg.get("hydra", {}) or {}).get("sweeper", {}).get("params", {}) or {}
        datasets = [d.strip() for d in str(params.get("dataset", "")).split(",") if d.strip()]
        for dname in datasets:
            dcfg = _load_yaml(ds_dir / f"{dname}.yaml")
            enc = dcfg.get("tiktoken_encoding")
            path = dcfg.get("tokenized_data_path")
            fam, label = NL_DATASETS.get(path, (None, None))
            rows.append({
                "family": fam,
                "setup": label,
                "tokenizer": enc or "gpt2 (default)",
                "cfg_max_steps": training.get("max_steps"),
            })
    df = pd.DataFrame(rows)
    # `family` is only needed to filter; the human-readable `setup` replaces it.
    return _filter_family(df, family).drop(columns="family").reset_index(drop=True)
