import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def get_runs_data(
        tag: str, 
        metrics: list[str],
        entity: str = "r-alvarezlucendo16",
        project: str = "llm-memorization",
        step: str = "train/global_step",
    ) -> pd.DataFrame:
    import wandb

    api = wandb.Api()
    filters = {"tags": tag}
   
    runs = list(api.runs(f"{entity}/{project}", filters))
    print(f"Found {len(runs)} runs with tag '{tag}'")

    dfs = []
    for run in runs:
        keys = metrics + [step]
        history = list(run.scan_history(keys=keys))
        meta = {"_run_id": run.id, "_run_name": run.name}
        config = dict(run.config)

        # Flatten nested config
        for key in ["dataset", "model", "training", "dataset_metadata"]:
            if key in config:
                for subkey, value in config[key].items():
                    meta[f"cfg.{key}.{subkey}"] = value

        # Create DataFrames and concatenate efficiently (avoid fragmentation)
        history_df = pd.DataFrame(history)
        meta_df = pd.DataFrame([meta] * len(history_df))
        df = pd.concat([history_df, meta_df], axis=1)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df

def get_per_position_data(
    tag: str,
    subset: str,
    entity: str = "r-alvarezlucendo16",
    project: str = "memorization",
    step: str = "train/global_step",
) -> pd.DataFrame:
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"tags": tag}))
    print(f"Found {len(runs)} runs with tag '{tag}'")

    dfs = []
    for run in runs:
        pos_keys = sorted(
            [k for k in run.summary.keys() if f"per_pos_acc_{subset}/t" in k],
            key=lambda k: int(k.split("/t")[-1]),
        )
        history = list(run.scan_history(keys=pos_keys + [step]))
        meta = {"_run_id": run.id}
        config = dict(run.config)
        for key in ["dataset", "model", "training", "dataset_metadata"]:
            if key in config:
                for subkey, value in config[key].items():
                    meta[f"cfg.{key}.{subkey}"] = value
        history_df = pd.DataFrame(history)
        meta_df = pd.DataFrame([meta] * len(history_df))
        dfs.append(pd.concat([history_df, meta_df], axis=1))

    return pd.concat(dfs, ignore_index=True)


def first_steps_to_threshold(
    df: pd.DataFrame,
    subset: str,
    threshold: float,
    step_key: str = "train/global_step",
) -> pd.Series:
    pos_cols = sorted(
        [c for c in df.columns if f"per_pos_acc_{subset}/t" in c],
        key=lambda c: int(c.split("/t")[-1]),
    )
    result = {}
    for col in pos_cols:
        t = int(col.split("/t")[-1])
        col_data = df[[step_key, col]].dropna().sort_values(step_key)
        reached = col_data[col_data[col] >= threshold]
        result[t] = reached[step_key].iloc[0] if len(reached) > 0 else np.nan
    return pd.Series(result).sort_index()


def show_config_differences(data: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in data.columns if "cfg." in col]
    configs = data.groupby("_run_id")[cols].first()
    # Convert lists to tuples so they're hashable for nunique()
    configs = configs.map(lambda x: tuple(x) if isinstance(x, list) else x)
    mask = configs.nunique(axis=0) > 1
    variable_config_columns = configs.columns[mask]
    return configs[variable_config_columns]

def filter_data(data: pd.DataFrame, filters: dict) -> pd.DataFrame:
    df = data.copy()
    for key, value in filters.items():
        df = df[df[key] == value]
    return df


def _process_single_run(run, metric, step_key):
    """Process a single wandb run - designed for parallel execution."""
    try:
        # Fetch history for this run
        history = list(run.scan_history(keys=[metric, step_key]))
        if not history:
            return None

        # Parse config efficiently
        config = json.loads(run.config)
        meta = {"_run_id": run.id, "_run_name": run.name}

        # Flatten nested config
        for key in ["dataset", "model", "training"]:
            if key in config and "value" in config[key]:
                for subkey, value in config[key]["value"].items():
                    meta[f"cfg.{key}.{subkey}"] = value

        # Add other config items
        meta.update(
            {
                k: v
                for k, v in config.items()
                if k not in ["dataset", "model", "training"]
            }
        )

        # Create DataFrames and concatenate efficiently (avoid fragmentation)
        history_df = pd.DataFrame(history)
        meta_df = pd.DataFrame([meta] * len(history_df))
        df = pd.concat([history_df, meta_df], axis=1)

        return df
    except Exception as e:
        print(f"Error processing run {run.name}: {e}")
        return None


def _fetch_wandb_data(entity, project, tag, metric, step_key, max_workers=16):
    """Fetch wandb data with parallel processing."""
    api = wandb.Api()
    filters = {"tags": tag} if tag else {}
    runs = list(api.runs(f"{entity}/{project}", filters))

    print(
        f"Fetched {len(runs)} runs, processing in parallel with {max_workers} workers..."
    )

    # Process runs in parallel
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {
            executor.submit(_process_single_run, run, metric, step_key): run
            for run in runs
        }

        for i, future in enumerate(as_completed(future_to_run), 1):
            result = future.result()
            if result is not None:
                dfs.append(result)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(runs)} runs")

    print(f"Successfully processed {len(dfs)}/{len(runs)} runs")

    # Concatenate all dataframes at once
    return pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame()


def _find_threshold_steps_vectorized(group, step_key, metric, threshold):
    """Vectorized threshold detection using numpy."""
    metric_vals = group[metric].to_numpy()
    step_vals = group[step_key].to_numpy()

    # Find first index where metric >= threshold
    mask = metric_vals >= threshold
    if mask.any():
        idx = np.argmax(mask)  # First True index
        return step_vals[idx]
    return np.nan


def create_memorization_heatmap(
    entity: str = "r-alvarezlucendo16",
    project: str = "llm-memorization",
    tag: str = None,
    threshold: float = 0.95,
    metric: str = "train/exact_match_repeated",
    step_key: str = "train/global_step",
    pivot_index: str = "repeated_batch_ratio",
    pivot_columns: str = "n_samples",
    repeated_pool_ratio: float = None,
    figsize: tuple = (12, 6),
    cache_file: str = None,
    max_workers: int = 16,
):
    """
    Create a heatmap showing steps to reach threshold for different configurations.
    """
    # Try to load from cache
    if cache_file and Path(cache_file).exists():
        print(f"Loading data from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
    else:
        # Fetch data with parallelization
        df = _fetch_wandb_data(entity, project, tag, metric, step_key, max_workers)

        # Save to cache if requested
        if cache_file:
            print(f"Saving data to cache: {cache_file}")
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(df, f)

    # Filter by repeated_pool_ratio if specified
    if repeated_pool_ratio is not None:
        df = df[df["cfg.dataset.repeated_pool_ratio"] == repeated_pool_ratio]
        print(f"Filtered to repeated_pool_ratio={repeated_pool_ratio}: {len(df)} rows")

    # Group and calculate steps to threshold using vectorized operation
    group_keys = [
        "_run_name",
        f"cfg.dataset.{pivot_columns}",
        f"cfg.dataset.{pivot_index}",
        "cfg.dataset.repeated_pool_ratio",
    ]

    # Use transform with vectorized function
    results_list = []
    for name_tuple, group in df.groupby(group_keys):
        step_to_threshold = _find_threshold_steps_vectorized(
            group, step_key, metric, threshold
        )
        results_list.append(
            {
                pivot_columns: name_tuple[1],
                pivot_index: name_tuple[2],
                "repeated_pool_ratio": name_tuple[3],
                "steps_to_threshold": step_to_threshold,
            }
        )

    results_df = pd.DataFrame(results_list)

    # Create pivot table and heatmap
    pivot = (
        results_df.pivot(
            index=pivot_index, columns=pivot_columns, values="steps_to_threshold"
        )
        .sort_index()
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=figsize)
    data = pivot.to_numpy(dtype=float)
    mask = np.isnan(data)
    cmap = sns.color_palette("flare", as_cmap=True)
    cmap.set_bad("lightgray")  # NaNs
    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)

    # Set ticks and labels
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=18)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=18)
    ax.set_xlabel("Number of Samples", fontsize=20)
    ax.set_ylabel("Repeated Ratio", fontsize=20)

    # Update title to include pool ratio if specified
    title = f"Steps to Reach Threshold (> {threshold})"
    if repeated_pool_ratio is not None:
        title += f" | Pool Ratio = {repeated_pool_ratio}"
    ax.set_title(title, fontsize=22)

    # Colorbar
    cbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data)),
            cmap=im.get_cmap(),
        ),
        ax=ax,
    )
    cbar.set_label("Steps to Threshold", fontsize=16)

    # Annotate each cell with the value
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if not mask[i, j]:
                ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=20)

    plt.tight_layout()

    return fig, results_df
