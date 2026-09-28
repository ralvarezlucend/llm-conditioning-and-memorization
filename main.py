import json
import os
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from runner import get_trainer
from utils import set_seed


def _build_model_name(base_name: str, model_cfg) -> str:
    """Build a unique model name that encodes dataset identity and model architecture.

    Format: {base_name}_model-{n_layer}l-{n_head}h-{n_embd}e
    Example: level_1_aggregation_lookup_pos-0_seed_0_model-2l-8h-512e
    """
    return f"{base_name}_model-{model_cfg.n_layer}l-{model_cfg.n_head}h-{model_cfg.n_embd}e"


def _save_metadata(save_dir: str, dataset_path: str | None, config: DictConfig, trainer):
    """Save a metadata.json alongside the model with full config (same as wandb)."""
    from runner import filter_config_for_wandb

    metadata = filter_config_for_wandb(config, dataset=trainer.train_dataset)
    metadata["model_path"] = os.path.abspath(save_dir)
    metadata["dataset_path"] = os.path.abspath(dataset_path) if dataset_path else None

    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))

    if hasattr(config.dataset, "repeated_batch_ratio") and hasattr(
        config.dataset, "repeated_pool_size"
    ):
        batch_size = config.training.batch_size
        repeated_batch_ratio = config.dataset.repeated_batch_ratio
        repeated_pool_size = config.dataset.repeated_pool_size

        n_repeated_batch = int(batch_size * repeated_batch_ratio)

        if n_repeated_batch >= repeated_pool_size:
            print(
                f"""Skipping experiment with repeated pool size {repeated_pool_size} and
                repeated batch ratio {repeated_batch_ratio} ({n_repeated_batch} repeated samples in batch).
                """
            )
            return

    set_seed(config.training.seed)

    trainer = get_trainer(config)
    trainer.train()

    if config.training.save_model:
        dataset_name = getattr(config, "dataset_name", None)
        npz_path = getattr(config.dataset, "npz_path", None)
        tokenized_data_path = getattr(config.dataset, "tokenized_data_path", None)

        if dataset_name:
            base_name = dataset_name
        elif npz_path:
            base_name = Path(npz_path).stem
        elif tokenized_data_path:
            base_name = Path(tokenized_data_path).stem
        else:
            raise ValueError(
                "save_model is true but no dataset_name, npz_path, or tokenized_data_path "
                "found in config. Set `dataset_name=<name>` in your Hydra overrides."
            )

        model_name = _build_model_name(base_name, config.model)
        save_path = os.path.join(get_original_cwd(), "saved_models", model_name)
        # Unwrap compiled model before saving (torch.compile wraps the model)
        if hasattr(trainer.model, "_orig_mod"):
            trainer.model = trainer.model._orig_mod
        trainer.save_model(save_path)
        print(f"Model saved to {save_path}")

        # Determine the dataset file path for metadata
        dataset_file_path = None
        import numpy as np
        dataset = trainer.train_dataset
        if hasattr(dataset, "repeated_pool"):
            ds_path = os.path.join(get_original_cwd(), "datasets", f"{model_name}_dataset.npz")
            np.savez(ds_path, dataset=dataset.repeated_pool["input_ids"].numpy())
            print(f"Dataset exported to {ds_path}")
            dataset_file_path = ds_path
        elif npz_path:
            dataset_file_path = os.path.join(get_original_cwd(), npz_path)

        _save_metadata(save_path, dataset_file_path, config, trainer)

        # Copy intermediate checkpoints to saved_models/{model_name}/checkpoint-{step}/
        import shutil
        import glob
        ckpt_dirs = sorted(glob.glob("output/checkpoint-*"))
        for ckpt_dir in ckpt_dirs:
            step = Path(ckpt_dir).name  # e.g. "checkpoint-500"
            dest = os.path.join(save_path, step)
            if not os.path.exists(dest):
                shutil.copytree(ckpt_dir, dest)
                _save_metadata(dest, dataset_file_path, config, trainer)
                print(f"Checkpoint copied to {dest}")

    if config.training.report_to == "wandb":
        import wandb

        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Shutdown Ray cluster after all jobs complete
        import ray

        if ray.is_initialized():
            ray.shutdown()
