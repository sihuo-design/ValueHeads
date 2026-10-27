"""Significance of value-head masking (Appendix H, Table 12).

For every (model, eval set), compares the masked score of the top-k value heads with the masked
scores of several random-head draws (mask_heads.py --select random --seed s) using a one-sided
one-sample t-test, H1: mean(random) > top-k.

    python analysis/significance.py --summary results/mask/summary.csv
"""
import argparse

import pandas as pd
from scipy.stats import ttest_1samp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/mask/summary.csv")
    parser.add_argument("--out", default="results/mask/significance.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    df = df[df["eval_set"].map(lambda s: s in ("jailbreak", "malicious")) | (df["head_value"] == df["eval_set"])]
    df = df[df["temperature"].isna()] if "temperature" in df else df
    rows = []
    for (model, eval_set), group in df.groupby(["model", "eval_set"]):
        random = group[group["select"] == "random"]["masked"].to_numpy()
        topk = group[group["select"] == "topk"]["masked"]
        if len(random) < 2 or topk.empty:
            continue
        topk = float(topk.iloc[-1])
        p = ttest_1samp(random, topk, alternative="greater").pvalue
        rows.append({"model": model, "eval_set": eval_set, "random_mean": random.mean(),
                     "random_std": random.std(ddof=1), "n_random": len(random), "topk": topk, "p": p})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out.to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
