"""Step 2: train a linear probe per head and rank heads by test accuracy (value heads).

    python probe_heads.py --model qwen3-8B --value honesty

Writes {score_dir}/{model}_{value}_heads.json = {flat_head_idx: test accuracy}, sorted descending,
where flat_head_idx = layer * num_heads + head.
"""
import argparse

from transformers import AutoConfig

from valueheads.data import VALUES
from valueheads.models import MODELS, model_geometry
from valueheads.probing import head_scores_path, load_features, save_head_scores, train_head_probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1_8B_instruct")
    parser.add_argument("--value", required=True, choices=VALUES)
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--score_dir", default="results/heads")
    parser.add_argument("--no_layer_bias", action="store_true",
                        help="do not append the layer-mean activation to each head's features")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _, num_heads, _ = model_geometry(AutoConfig.from_pretrained(MODELS.get(args.model, args.model)))
    train_acts, y_train = load_features(args.feature_dir, args.model, args.value, "train")
    val_acts, y_val = load_features(args.feature_dir, args.model, args.value, "val")

    accs = train_head_probes(train_acts, y_train, val_acts, y_val, num_heads,
                             seed=args.seed, layer_bias=not args.no_layer_bias)
    path = head_scores_path(args.score_dir, args.model, args.value)
    save_head_scores(accs, path)
    print(f"saved {path}; best head acc = {accs.max():.3f}")


if __name__ == "__main__":
    main()
