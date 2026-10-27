"""Correlation analyses of value heads (Appendix D).

across-models (Fig. 10): Pearson correlation of head scores between models, per value. Score maps are
    linearly interpolated to the layer x head grid of the first model before flattening.
across-values (Fig. 11): Pearson correlation between values within one model, computed on binary
    vectors marking each value's top-15% heads.

    python analysis/value_correlations.py across-models
    python analysis/value_correlations.py across-values --models qwen3-4B llama3.2_3B_instruct
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from valueheads.data import VALUES  # noqa: E402
from valueheads.models import MODELS, model_geometry  # noqa: E402
from valueheads.probing import load_head_scores, top_heads  # noqa: E402


def geometry(model):
    num_layers, num_heads, _ = model_geometry(AutoConfig.from_pretrained(MODELS.get(model, model)))
    return num_layers, num_heads


def score_grid(score_dir, model, value):
    num_layers, num_heads = geometry(model)
    scores = load_head_scores(score_dir, model, value)
    return np.array([scores[i] for i in range(num_layers * num_heads)]).reshape(num_layers, num_heads)


def resample(grid, rows, cols):
    """Linear interpolation of a 2D grid onto (rows, cols), axis by axis."""
    tmp = np.stack([np.interp(np.linspace(0, 1, rows), np.linspace(0, 1, grid.shape[0]), grid[:, h])
                    for h in range(grid.shape[1])], axis=1)
    return np.stack([np.interp(np.linspace(0, 1, cols), np.linspace(0, 1, tmp.shape[1]), tmp[l])
                     for l in range(rows)])


PAPER_ORDER = ["llama3.1_8B_instruct", "llama3.2_3B_instruct", "qwen3-4B", "qwen3-8B", "yi-1.5-6B", "yi-1.5-9B"]


def across_models(args):
    models = args.models or PAPER_ORDER
    rows, cols = geometry(models[0])
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, value in zip(axes.flatten(), VALUES):
        vecs = [resample(score_grid(args.score_dir, m, value), rows, cols).flatten() for m in models]
        sns.heatmap(np.corrcoef(vecs), ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    xticklabels=models, yticklabels=models)
        ax.set_title(value, fontsize=13, fontweight="bold")
    plt.suptitle("Pearson Correlation of Head Importance Between Models (per Value)", fontsize=15)
    plt.tight_layout()
    return fig


def across_values(args):
    models = args.models or ["qwen3-4B", "llama3.2_3B_instruct"]
    fig, axes = plt.subplots(1, len(models), figsize=(9 * len(models), 10), squeeze=False)
    for ax, model in zip(axes[0], models):
        num_layers, num_heads = geometry(model)
        k = int(num_layers * num_heads * args.ratio)
        vecs = []
        for value in VALUES:
            vec = np.zeros(num_layers * num_heads)
            vec[top_heads(load_head_scores(args.score_dir, model, value), k)] = 1
            vecs.append(vec)
        sns.heatmap(np.corrcoef(vecs), ax=ax, annot=True, fmt=".2f", cmap="viridis", cbar=False, square=True,
                    xticklabels=VALUES, yticklabels=VALUES)
        ax.set_title(model, fontsize=20)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["across-models", "across-values"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--score_dir", default="results/heads")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    fig = across_models(args) if args.mode == "across-models" else across_values(args)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"correlation_{args.mode}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
