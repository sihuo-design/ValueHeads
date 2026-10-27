"""Model registry and loading utilities."""
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama3.1_8B_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2_3B_instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen3-8B": "Qwen/Qwen3-8B",
    "qwen3-4B": "Qwen/Qwen3-4B",
    "yi-1.5-9B": "01-ai/Yi-1.5-9B-Chat",
    "yi-1.5-6B": "01-ai/Yi-1.5-6B-Chat",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_geometry(config):
    """Returns (num_layers, num_heads, head_dim) of the attention stack."""
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // num_heads
    return num_layers, num_heads, head_dim


def load_model(model_name, device="cuda:0", dtype=torch.float16):
    """Loads a model / tokenizer pair by short name (see MODELS) or HF path.

    Gated models (Llama) need `huggingface-cli login` or the HF_TOKEN env variable.
    """
    path = MODELS.get(model_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device, trust_remote_code=True
    ).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def o_proj(model, layer):
    """The attention output projection of `layer`; its input is the concatenation of all head outputs."""
    return model.model.layers[layer].self_attn.o_proj
