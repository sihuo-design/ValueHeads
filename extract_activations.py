"""Step 1: collect last-token per-head activations.

Labeled probing sets (one per value, train + val splits):
    python extract_activations.py --model qwen3-8B --dataset honesty --split train
    python extract_activations.py --model qwen3-8B --dataset honesty --split val
Unlabeled prompt sets, only used to estimate the activation std for steering:
    python extract_activations.py --model qwen3-8B --dataset jailbreak --split val

Writes {out_dir}/{model}_{dataset}_{split}_head_wise.npy with shape (N, num_layers, num_heads * head_dim)
(float16) and, for labeled sets, {model}_{dataset}_{split}_labels.npy.
"""
import argparse
import os
import random

import numpy as np
from tqdm import tqdm

from valueheads.data import VALUES, probing_set, prompt_set
from valueheads.hooks import ActivationCollector
from valueheads.models import load_model, model_geometry, set_seed
from valueheads.probing import feature_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1_8B_instruct")
    parser.add_argument("--dataset", required=True, choices=VALUES + ["jailbreak", "malicious", "beavertails"])
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max_samples", type=int, default=None, help="random subsample (keeps label order)")
    parser.add_argument("--out_dir", default="features")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.dataset in VALUES:
        texts, labels = probing_set(args.dataset, args.split)
    else:
        texts, labels = prompt_set(args.dataset), None
    if args.max_samples and len(texts) > args.max_samples:
        keep = sorted(random.Random(args.seed).sample(range(len(texts)), args.max_samples))
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep] if labels is not None else None
    print(f"{args.dataset}/{args.split}: {len(texts)} examples")

    model, tokenizer = load_model(args.model, args.device)
    num_layers, num_heads, head_dim = model_geometry(model.config)

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = feature_path(args.out_dir, args.model, args.dataset, args.split)
    acts = np.lib.format.open_memmap(prefix + "_head_wise.npy", mode="w+", dtype=np.float16,
                                     shape=(len(texts), num_layers, num_heads * head_dim))
    with ActivationCollector(model) as collector:
        for i, text in enumerate(tqdm(texts, desc="collecting")):
            input_ids = tokenizer(text, return_tensors="pt").input_ids
            acts[i] = collector.collect(input_ids).numpy().astype(np.float16)
    acts.flush()
    if labels is not None:
        np.save(prefix + "_labels.npy", np.array(labels))
    print(f"saved {prefix}_head_wise.npy")


if __name__ == "__main__":
    main()
