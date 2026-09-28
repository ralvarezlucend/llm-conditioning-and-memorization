import inspect
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
)

from dataset import BaseHybridDataset, NpzDataset, VariableVocabDataset
from model_utils import get_cached_tokenizer
from trainer import CustomTrainer

# TODO: understand this better
def hybrid_data_collator(features):
    """Custom data collator for BaseHybridDataset that handles dynamic mask fields (e.g., is_synthetic, is_regular)."""
    # Find the mask field name (anything that's not input_ids, attention_mask, or labels)
    mask_field = None
    for key in features[0].keys():
        if key not in ["input_ids", "attention_mask", "labels"]:
            mask_field = key
            break

    # Extract mask values and ensure they're scalar tensors
    mask_values = [f[mask_field].item() if isinstance(f[mask_field], torch.Tensor) and f[mask_field].dim() == 0
                   else f[mask_field] for f in features]

    result = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }

    if mask_field:
        result[mask_field] = torch.tensor(mask_values, dtype=torch.bool)

    return result


def create_dataset(config: DictConfig, pad_token_id: int, seed: int, ignore_pad_tokens_in_loss: bool):
    """Create dataset and return it with its vocab size."""
    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    dataset_config["seed"] = seed

    # Get the dataset class to check which parameters it accepts
    target_class_path = config.dataset._target_
    module_path, class_name = target_class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    dataset_class = getattr(module, class_name)

    # Get the __init__ signature to see which parameters are accepted
    init_signature = inspect.signature(dataset_class.__init__)
    init_params = set(init_signature.parameters.keys())

    # Only add parameters if the dataset class accepts them
    if "pad_token_id" in init_params:
        dataset_config["pad_token_id"] = pad_token_id
    if "ignore_pad_tokens_in_loss" in init_params:
        dataset_config["ignore_pad_tokens_in_loss"] = ignore_pad_tokens_in_loss

    train_dataset = instantiate(dataset_config)

    if hasattr(train_dataset, "vocab_size"):
        vocab_size = train_dataset.vocab_size
        print(f"Using vocab_size from dataset: {vocab_size}")
    else:
        vocab_size = get_cached_tokenizer().vocab_size
        print(f"Using vocab_size from GPT-2 tokenizer: {vocab_size}")

    return train_dataset, vocab_size, dataset_config


def get_gpt2_based_on_emb_dim(emb_dim: int) -> str:
    """Select the GPT-2 variant whose embedding dim >= emb_dim for PCA reduction."""
    if emb_dim <= 768:
        return "gpt2"
    elif emb_dim <= 1024:
        return "gpt2-medium"
    elif emb_dim <= 1280:
        return "gpt2-large"
    elif emb_dim <= 1600:
        return "gpt2-xl"
    else:
        raise NotImplementedError(f"No GPT-2 variant with embedding dim >= {emb_dim}")


def init_with_gpt2_emb(model, vocab_size: int, seq_length: int):
    """Initialize wte and wpe from a pretrained GPT-2 model using PCA."""
    emb_dim = model.config.n_embd
    gpt2_model_name = get_gpt2_based_on_emb_dim(emb_dim)

    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    tok_emb = gpt2_model.transformer.wte.weight   # [50257, gpt2_n_embd]
    pos_emb = gpt2_model.transformer.wpe.weight   # [1024, gpt2_n_embd]
    gpt2_n_embd = tok_emb.shape[1]
    print(f"Initializing embeddings from {gpt2_model_name} (PCA: {gpt2_n_embd} -> {emb_dim})...")

    # Handle vocab size mismatch
    if vocab_size > tok_emb.shape[0]:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds GPT-2 vocab size ({tok_emb.shape[0]}). "
            "Cannot initialize embeddings from GPT-2."
        )
    tok_emb = tok_emb[:vocab_size]  # slice to match model vocab

    # PCA-reduce token embeddings
    pca_tok = PCA(n_components=emb_dim)
    tok_emb_pca = torch.tensor(
        pca_tok.fit_transform(tok_emb.detach().float().numpy()),
        dtype=model.transformer.wte.weight.dtype,
    )

    # PCA-reduce position embeddings
    pca_pos = PCA(n_components=emb_dim)
    pos_emb_pca = torch.tensor(
        pca_pos.fit_transform(pos_emb.detach().float().numpy()),
        dtype=model.transformer.wpe.weight.dtype,
    )

    model.transformer.wte.weight.data.copy_(tok_emb_pca)
    model.transformer.wpe.weight.data.copy_(pos_emb_pca[:seq_length])

    del gpt2_model  # free memory


def create_model(vocab_size: int, seq_length: int, model_cfg, pad_token_id: int, compile_model: bool = False):
    """Create and compile GPT-2 model."""
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_head=model_cfg.n_head,
        n_embd=model_cfg.n_embd,
        n_layer=model_cfg.n_layer,
        n_positions=seq_length,  # Maximum sequence length for positional embeddings
        pad_token_id=pad_token_id,
    )
    model = AutoModelForCausalLM.from_config(gpt2_config)

    # Initialize and freeze embeddings from GPT-2 if requested
    freeze_from_gpt2 = getattr(model_cfg, "freeze_embeddings_from_gpt2", False)
    if freeze_from_gpt2:
        init_with_gpt2_emb(model, vocab_size, seq_length)
        model.transformer.wte.weight.requires_grad = False
        model.transformer.wpe.weight.requires_grad = False
        print("Frozen wte and wpe (initialized from GPT-2)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Model params: {total_params / 1e6:.2f}M -> {total_params} params")
    if frozen_params > 0:
        print(f"  Trainable: {trainable_params / 1e6:.2f}M ({trainable_params} params)")
        print(f"  Frozen: {frozen_params / 1e6:.2f}M ({frozen_params} params)")

    if compile_model:
        # Configure and compile model (after freezing to optimize for trainable graph)
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model)

    return model



def filter_config_for_wandb(config: DictConfig, dataset=None, model_vocab_size=None) -> dict:
    """
    Filter Hydra config and add dataset metadata for wandb logging.

    Includes:
    - config.dataset - Dataset configuration from conf/dataset/*.yaml
    - config.model - Model architecture from conf/model/*.yaml
    - config.training - Training hyperparameters from conf/training/*.yaml
    - dataset_metadata - Runtime metadata from dataset instance (if available)

    Excludes:
    - config.hydra - Hydra runtime metadata
    - Any other runtime-generated fields

    Args:
        config: Full Hydra DictConfig object
        dataset: Optional dataset instance to extract metadata from
        model_vocab_size: Optional model vocab_size (for NpzDataset override tracking)

    Returns:
        Filtered dictionary containing YAML configs and dataset metadata
    """
    # Whitelist of config sections to include
    yaml_config_keys = ["dataset", "model", "training"]

    filtered_config = {}

    for key in yaml_config_keys:
        if hasattr(config, key):
            filtered_config[key] = OmegaConf.to_container(
                config[key], resolve=True
            )

    # Add dataset metadata if available (e.g., from NpzDataset)
    if dataset is not None and hasattr(dataset, 'target_level'):
        filtered_config['dataset_metadata'] = {
            'target_level': dataset.target_level,
            'vocab_size': dataset.vocab_size,
            'num_samples': dataset.num_samples,
            'seq_len': dataset.seq_len,
            'prefix_len': dataset.prefix_len,
            'random_prefix': dataset.random_prefix,
            'strategy': dataset.strategy,
            'aggregation': dataset.aggregation,
            'seed': dataset.seed,
            'pos': dataset.pos,
            'lookback': dataset.lookback,
        }

        # For NpzDataset, also log the model's actual vocab_size if different
        if isinstance(dataset, NpzDataset) and model_vocab_size is not None:
            if model_vocab_size != dataset.vocab_size:
                filtered_config['dataset_metadata']['model_vocab_size'] = model_vocab_size

    # Add VariableVocabDataset info bits metadata
    if dataset is not None and isinstance(dataset, VariableVocabDataset):
        filtered_config.setdefault('dataset_metadata', {})
        filtered_config['dataset_metadata'].update({
            'prefix_info_bits': dataset.prefix_info_bits,
            'suffix_info_bits': dataset.suffix_info_bits,
            'total_info_bits': dataset.total_info_bits,
        })

    # Add karity metadata if available
    if dataset is not None and hasattr(dataset, 'K') and dataset.K is not None:
        filtered_config.setdefault('dataset_metadata', {})
        filtered_config['dataset_metadata']['K'] = dataset.K
        filtered_config['dataset_metadata']['mode'] = dataset.mode

    # Log num_samples for any dataset that supports len()
    if dataset is not None and hasattr(dataset, '__len__'):
        filtered_config.setdefault('dataset_metadata', {})
        filtered_config['dataset_metadata'].setdefault('num_samples', len(dataset))

    return filtered_config


def setup_wandb(config: DictConfig, dataset=None, model_vocab_size=None):
    """
    Initialize wandb if configured.

    Args:
        config: Full Hydra DictConfig object
        dataset: Optional dataset instance to extract metadata from
        model_vocab_size: Optional model vocab_size (for NpzDataset override tracking)
    """
    if config.training.report_to != "wandb":
        return

    import wandb

    wandb.init(
        entity="r-alvarezlucendo16",
        project=config.training.get("wandb_project", "memorization"),
        name=None,
        tags=list(config.training.get("wandb_tags", [])),
        config=filter_config_for_wandb(config, dataset, model_vocab_size),
        reinit=True,
    )


def get_trainer(config: DictConfig):
    """Create and configure trainer with all components."""
    seed = config.training.seed
    outdir = "output"

    # Setup components
    tokenizer = get_cached_tokenizer()

    pad_token_id = tokenizer.eos_token_id

    train_dataset, vocab_size, dataset_config = create_dataset(
        config, pad_token_id, seed, config.training.ignore_pad_tokens_in_loss
    )

    # Get sequence length - from config or dataset instance
    seq_length = (
        getattr(config.dataset, "max_length", None)
        or getattr(config.dataset, "length", None)
        or getattr(train_dataset, "seq_len", None)
    )
    if seq_length is None:
        raise ValueError(
            "Cannot determine sequence length from config or dataset. "
            "Dataset config must specify 'max_length' or 'length', "
            "or dataset instance must have 'seq_len' attribute."
        )

    model = create_model(
        vocab_size, seq_length, config.model, pad_token_id,
        compile_model=config.training.compile_model,
    )
    setup_wandb(config, train_dataset, model_vocab_size=vocab_size)

    # Training arguments
    targs = TrainingArguments(
        output_dir=outdir,
        per_device_train_batch_size=config.training.batch_size,
        bf16=config.training.bf16,
        max_steps=config.training.max_steps,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        lr_scheduler_type=config.training.lr_scheduler_type,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        optim=config.training.optim,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        report_to=config.training.report_to,
        seed=seed,
        save_strategy="steps" if config.training.save_steps > 0 else "no",
        max_grad_norm=config.training.max_grad_norm,
        gradient_checkpointing=config.training.get("gradient_checkpointing", False),
        remove_unused_columns=False,
    )

    data_collator = hybrid_data_collator if isinstance(train_dataset, BaseHybridDataset) else None

    return CustomTrainer(
        dataset_config=dataset_config,
        loss_logging_mode=config.training.loss_logging_mode,
        ignore_pad_tokens_in_loss=config.training.ignore_pad_tokens_in_loss,
        model=model,
        args=targs,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
