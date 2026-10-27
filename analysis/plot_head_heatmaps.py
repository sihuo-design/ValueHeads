"""Layer x head heatmaps of value-head scores (Fig. 2 and Fig. 5-9).

    python analysis/plot_head_heatmaps.py --model qwen3-8B
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from valueheads.models import MODELS, model_geometry  # noqa: E402
from valueheads.probing import load_head_scores  # noqa: E402

VALUES = ["harmless", "helpfulness", "complexity", "coherence", "verbosity", "honesty"]


def score_map(scores, num_layers, num_heads):
    """Min-max normalized and squared accuracies as a (layers, heads) array."""
    arr = np.array([scores[i] for i in range(num_layers * num_heads)])
    arr = ((arr - arr.min()) / (arr.max() - arr.min())) ** 2
    return arr.reshape(num_layers, num_heads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-8B")
    parser.add_argument("--score_dir", default="results/heads")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    num_layers, num_heads, _ = model_geometry(AutoConfig.from_pretrained(MODELS.get(args.model, args.model)))
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
    for ax, value in zip(axes.flatten(), VALUES):
        arr = score_map(load_head_scores(args.score_dir, args.model, value), num_layers, num_heads)
        im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=0.6, vmax=1.0, origin="lower")
        ax.set_title(value, fontsize=26)
        ax.tick_params(labelsize=22)
    fig.text(0.5, 0.06, "Head Number", ha="center", fontsize=26)
    fig.text(0.09, 0.5, "Layer Number", va="center", rotation="vertical", fontsize=26)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04).ax.tick_params(labelsize=22)

    out = args.out or f"results/figures/heatmap_{args.model}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
