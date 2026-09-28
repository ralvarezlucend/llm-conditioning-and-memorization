import torch
from transformers import AutoModelForCausalLM, GPT2Config


def load_model(model_path: str):
    """Load a saved GPT-2 model as a TransformerLens HookedTransformer."""
    import os
    from safetensors.torch import load_file
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    from transformer_lens.loading_from_pretrained import convert_gpt2_weights

    config = GPT2Config.from_pretrained(model_path)
    hf_model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

    bin_path = os.path.join(model_path, "pytorch_model.bin")
    st_path  = os.path.join(model_path, "model.safetensors")

    if os.path.exists(st_path):
        state_dict = load_file(st_path, device="cpu")
    else:
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)

    missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
    assert missing == ["lm_head.weight"], f"Unexpected missing keys: {missing}"
    assert unexpected == [], f"Unexpected keys in checkpoint: {unexpected}"
    hf_model.tie_weights()

    hooked_cfg = HookedTransformerConfig(
        n_layers=config.n_layer,
        d_model=config.n_embd,
        d_head=config.n_embd // config.n_head,
        n_heads=config.n_head,
        d_mlp=config.n_inner if config.n_inner else 4 * config.n_embd,
        d_vocab=config.vocab_size,
        n_ctx=config.n_positions,
        act_fn=config.activation_function,
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
