"""Angles between value directions on heads shared by two values (Appendix K, Fig. 12, Table 17).

    python analysis/direction_angles.py --models llama3.1_8B_instruct qwen3-8B
"""
import argparse
import csv
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from valueheads.directions import mass_mean_directions  # noqa: E402
from valueheads.models import MODELS, model_geometry  # noqa: E402
from valueheads.probing import load_features, load_head_scores, top_heads  # noqa: E402

VALUES = ["harmless", "helpfulness", "honesty"]
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]


def angle_deg(a, b):
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["llama3.1_8B_instruct", "llama3.2_3B_instruct", "qwen3-8B", "qwen3-4B"])
    parser.add_argument("--ratio", type=float, default=0.15)
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--score_dir", default="results/heads")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(1, len(args.models), figsize=(7 * len(args.models), 5), squeeze=False)
    for ax, model in zip(axes[0], args.models):
        num_layers, num_heads, _ = model_geometry(AutoConfig.from_pretrained(MODELS.get(model, model)))
        k = int(num_layers * num_heads * args.ratio)
        dirs, heads = {}, {}
        for value in VALUES:
            train = load_features(args.feature_dir, model, value, "train")
            val = load_features(args.feature_dir, model, value, "val")
            dirs[value] = mass_mean_directions([train[0], val[0]], [train[1], val[1]], num_heads)
            heads[value] = set(top_heads(load_head_scores(args.score_dir, model, value), k))
        for (a, b), color in zip(itertools.combinations(VALUES, 2), COLORS):
            angles = [angle_deg(dirs[a][h], dirs[b][h]) for h in sorted(heads[a] & heads[b])]
            if not angles:
                continue
            rows.append([model, f"{a} x {b}", len(angles), np.mean(angles), np.median(angles), np.std(angles)])
            print(f"{model:22s} {a:>11s} x {b:<11s} n={len(angles):4d} mean={np.mean(angles):6.2f}")
            ax.hist(angles, bins=20, range=(0, 180), alpha=0.6, color=color,
                    label=f"{a} x {b} (n={len(angles)}, mean={np.mean(angles):.1f}°)")
            ax.axvline(np.mean(angles), color=color, linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("# Heads")
        ax.legend(fontsize=8, loc="upper left")

    with open(os.path.join(args.out_dir, "direction_angles.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "pair", "n_shared_heads", "mean_deg", "median_deg", "std_deg"])
        writer.writerows(rows)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "direction_angles.png"), dpi=150, bbox_inches="tight")
    print(f"saved {args.out_dir}/direction_angles.(csv|png)")


if __name__ == "__main__":
    main()
