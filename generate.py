"""
Generate datasets at progressive difficulty levels.

This module provides functions to generate datasets and test them against
various conditional entropy difficulty levels using embedded patterns.
"""

from pathlib import Path

import yaml

from embedded import generate_dataset
from generate_karity import generate_karity_dataset_pipeline


def generate_training_configs(dataset_names, output_dir, base_params, batch_size=4, config_name_prefix=None, experiment_subdir=None, train_overrides=None, train_defaults_overrides=None):
    """Generate Hydra YAML training configs, max batch_size datasets per file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove .npz extension from dataset names
    dataset_names = [name.replace(".npz", "") for name in dataset_names]

    # Build informative config filename prefix
    seeds = base_params.get("seeds", [0])
    seeds_str = "-".join(str(s) for s in seeds)

    if config_name_prefix is None:
        vocab_size = base_params["vocab_size"]
        num_samples = base_params["num_samples"]
        config_name_prefix = f"train_V{vocab_size}_N{num_samples}_seeds{seeds_str}"

    # Split into batches
    batches = [
        dataset_names[i : i + batch_size]
        for i in range(0, len(dataset_names), batch_size)
    ]

    generated_files = []
    # Build optional training overrides block
    train_overrides_block = ""
    if train_overrides:
        import json
        lines = ["training:"]
        for k, v in train_overrides.items():
            if isinstance(v, bool):
                lines.append(f"  {k}: {'true' if v else 'false'}")
            elif isinstance(v, list):
                lines.append(f"  {k}: {json.dumps(v)}")
            else:
                lines.append(f"  {k}: {v}")
        train_overrides_block = "\n" + "\n".join(lines) + "\n"

    # Build extra defaults lines (e.g., model override)
    extra_defaults = ""
    if train_defaults_overrides:
        extra_defaults = "\n".join(f"    - {d}" for d in train_defaults_overrides) + "\n"

    for batch_idx, batch in enumerate(batches, start=1):
        config_content = f"""# @package _global_

defaults:
    - /train
    - override /dataset: npz
{extra_defaults}    - override /hydra/launcher: ray
    - _self_

dataset_name: ???
{train_overrides_block}
hydra:
    sweeper:
        params:
            dataset_name: {",".join(batch)}
"""
        if experiment_subdir:
            experiment_dir = output_dir / f"experiment/{experiment_subdir}"
        else:
            experiment_dir = output_dir / "experiment"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        output_path = experiment_dir / f"{config_name_prefix}_batch_{batch_idx}.yaml"
        with open(output_path, "w") as f:
            f.write(config_content)
        generated_files.append(output_path)

    return generated_files


def run_karity(config):
    """
    Generate K-arity datasets with shared base parameters.
    """
    base_params = config.get("base_params", {})
    datasets_config = config.get("datasets", [])
    seeds = base_params.get("seeds", [0])

    print("\nBase parameters (shared across all datasets):")
    for key, value in base_params.items():
        print(f"  {key}: {value}")

    results_summary = []

    for seed in seeds:
        for ds_conf in datasets_config:
            K = ds_conf["K"]
            pos_fn = base_params.get("pos_fn_type", "random_map")
            pos_fn_tag = f"_{pos_fn}" if pos_fn != "random_map" else ""
            output = f"karity_K{K}{pos_fn_tag}_V{base_params['vocab_size']}_N{base_params['num_samples']}_seed{seed}.npz"

            params = base_params.copy()
            params["seed"] = seed

            metadata = generate_karity_dataset_pipeline(
                K=K,
                params=params,
                output_path=output,
            )

            results_summary.append(
                {
                    "K": K,
                    "output": output,
                }
            )

    # Generate training configs if enabled
    if config.get("generate_train_configs", False):
        dataset_names = [info["output"] for info in results_summary]
        generated_files = generate_training_configs(
            dataset_names, output_dir="conf", base_params=base_params,
            config_name_prefix=config.get("config_name_prefix"),
            train_overrides=config.get("train_overrides"),
            train_defaults_overrides=config.get("train_defaults_overrides"),
        )
        print("Generated training configs:")
        for f in generated_files:
            print(f"  {f}")


def run(config):
    """
    Generate multiple datasets with shared base parameters.
    """
    base_params = config.get("base_params", {})
    datasets_config = config.get("datasets", [])
    aggregation = base_params.get("aggregation", None)
    seeds = base_params.get("seeds", [0])
    dataset_subdir = config.get("dataset_subdir", None)

    print("\nBase parameters (shared across all datasets):")
    for key, value in base_params.items():
        print(f"  {key}: {value}")

    results_summary = []

    for seed in seeds:
        for ds_conf in datasets_config:
            target_level = ds_conf["target_level"]
            strategy = ds_conf.get("strategy", None)

            # Extract level-specific kwargs (everything except target_level, num_samples, vocab_size)
            level_kwargs = {
                k: v for k, v in ds_conf.items() if k not in ("target_level", "num_samples", "vocab_size")
            }

            # Use per-dataset num_samples/vocab_size if provided, otherwise base
            effective_num_samples = ds_conf.get("num_samples", base_params.get("num_samples"))
            if effective_num_samples is None:
                raise ValueError(
                    f"num_samples not specified for dataset {ds_conf} and not in base_params"
                )
            effective_vocab_size = ds_conf.get("vocab_size", base_params.get("vocab_size"))

            output_parts = [f"level{target_level}"]
            if strategy:
                output_parts.append(strategy)
            output_parts.append(aggregation)

            # Add kwargs in alphabetical order (for consistent naming)
            for key in sorted(level_kwargs.keys()):
                if key == "strategy":
                    continue
                value = level_kwargs[key]
                output_parts.append(f"{key}{value}")

            output_parts.append(f"V{effective_vocab_size}")
            output_parts.append(f"N{effective_num_samples}")
            output_parts.append(f"seed{seed}")
            filename = "_".join(output_parts) + ".npz"
            output = f"{dataset_subdir}/{filename}" if dataset_subdir else filename

            # Merge base_params with dataset-specific config
            params = base_params.copy()
            params["seed"] = seed
            for key in ("num_samples", "vocab_size"):
                if key in ds_conf:
                    params[key] = ds_conf[key]

            metadata = generate_dataset(
                target_level=target_level,
                params=params,
                output_path=output,
                **level_kwargs,
            )

            results_summary.append(
                {
                    "level": target_level,
                    "output": output,
                    "aggregation": aggregation,
                    "passed": metadata["all_levels"][target_level]["passed"],
                    "function": metadata["all_levels"][target_level].get("function"),
                }
            )

    # Generate training configs if enabled
    if config.get("generate_train_configs", False):
        dataset_names = [info["output"] for info in results_summary]
        batch_size = config.get("train_config_batch_size", 4)

        # Use custom prefix if N or V varies across datasets
        has_varying_n = any("num_samples" in ds for ds in datasets_config)
        has_varying_v = any("vocab_size" in ds for ds in datasets_config)
        seeds_str = "-".join(str(s) for s in seeds)
        if has_varying_v and has_varying_n:
            prefix = f"train_Vvary_Nvary_seeds{seeds_str}"
        elif has_varying_v:
            prefix = f"train_Vvary_N{base_params.get('num_samples', 'unknown')}_seeds{seeds_str}"
        elif has_varying_n:
            prefix = f"train_V{base_params['vocab_size']}_Nvary_seeds{seeds_str}"
        else:
            prefix = None

        # Explicit prefix from config takes priority over auto-generated one
        prefix = config.get("config_name_prefix", prefix)

        generated_files = generate_training_configs(
            dataset_names, output_dir="conf", base_params=base_params,
            batch_size=batch_size, config_name_prefix=prefix,
            experiment_subdir=dataset_subdir,
            train_overrides=config.get("train_overrides"),
            train_defaults_overrides=config.get("train_defaults_overrides"),
        )
        print("Generated training configs:")
        for f in generated_files:
            print(f"  {f}")


def main():
    import sys

    conf_dir = Path("conf")
    config_name = sys.argv[1] if len(sys.argv) > 1 else "embedded"
    config_path = conf_dir / f"{config_name}.yaml"

    # Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Dispatch: karity configs use the separate pipeline
    is_karity = config.get("mode") == "karity" or any(
        "K" in ds_conf for ds_conf in config.get("datasets", [])
    )
    if is_karity:
        run_karity(config)
    else:
        run(config)


if __name__ == "__main__":
    main()
