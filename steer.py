"""Step 4: inference-time value intervention on value heads (Sec. 5, Tables 2-4).

Single value (Table 2), e.g. honesty heads evaluated on held-out TruthfulQA:
    python steer.py --model yi-1.5-9B --values honesty --eval_set honesty --alpha 1.0
Multiple values (Table 3), lambda_k = 1/K by default:
    python steer.py --model qwen3-8B --values harmless helpfulness complexity coherence verbosity honesty \
        --eval_set beavertails --alpha 1.0
Custom trade-off between values:
    python steer.py --model qwen3-8B --values harmless honesty helpfulness --weights 0.6 0.2 0.2 \
        --eval_set hh-rlhf

Each head in the union of the values' top heads is shifted along sum_k lambda_k dir_k over the values
whose top heads contain it (normalized, scaled by the activation std along the direction).
"""
import argparse
import csv
import json
import os

from tqdm import tqdm

from valueheads.data import EVAL_SETS, VALUES, eval_set
from valueheads.directions import mass_mean_directions, steering_vectors
from valueheads.generation import generate
from valueheads.hooks import HeadSteer
from valueheads.judge import Judge
from valueheads.models import load_model, model_geometry, set_seed
from valueheads.probing import load_features, load_head_scores, top_heads

# activations used to estimate sigma for each evaluation set (extract_activations.py, split "val")
STD_SETS = {"jailbreak": "jailbreak", "malicious": "malicious", "beavertails": "beavertails",
            "hh-rlhf": "harmless", **{v: v for v in VALUES}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1_8B_instruct")
    parser.add_argument("--values", nargs="+", required=True, choices=VALUES)
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="lambda_k per value (default 1/K); sum |lambda_k| <= 1")
    parser.add_argument("--eval_set", required=True, choices=list(EVAL_SETS))
    parser.add_argument("--judge_values", nargs="+", default=None, choices=VALUES,
                        help="values scored by the judge (default: the eval set's value, or all values "
                             "for hh-rlhf / beavertails)")
    parser.add_argument("--alpha", type=float, default=1.0, help="global intervention strength")
    parser.add_argument("--ratio", type=float, default=0.15, help="fraction of heads per value")
    parser.add_argument("--std_set", default=None, help="activation set for sigma (default: see STD_SETS)")
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None,
                        help="sample (top-p 0.9) instead of greedy decoding; vary --seed across runs")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--score_dir", default="results/heads")
    parser.add_argument("--out_dir", default="results/steer")
    parser.add_argument("--judge_url", default="http://localhost:30000/v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    weights = args.weights or [1.0 / len(args.values)] * len(args.values)
    assert len(weights) == len(args.values), "--weights must match --values"
    weights = dict(zip(args.values, weights))
    judge_values = args.judge_values or ([EVAL_SETS[args.eval_set]] if EVAL_SETS[args.eval_set] else VALUES)

    model, tokenizer = load_model(args.model, args.device)
    num_layers, num_heads, head_dim = model_geometry(model.config)
    k = int(num_layers * num_heads * args.ratio)

    value_heads, value_dirs = {}, {}
    for value in args.values:
        value_heads[value] = top_heads(load_head_scores(args.score_dir, args.model, value), k)
        train = load_features(args.feature_dir, args.model, value, "train")
        val = load_features(args.feature_dir, args.model, value, "val")
        value_dirs[value] = mass_mean_directions([train[0], val[0]], [train[1], val[1]], num_heads)
    std_acts, _ = load_features(args.feature_dir, args.model, args.std_set or STD_SETS[args.eval_set], "val")
    vectors = steering_vectors(value_heads, value_dirs, weights, std_acts, num_heads, head_dim)

    questions = eval_set(args.eval_set, args.num_eval)
    judge = Judge(args.judge_url)

    tag = f"{args.eval_set}_steer-{'+'.join(args.values)}_alpha{args.alpha}"
    if args.weights:
        tag += "_w" + "-".join(f"{w:g}" for w in weights.values())
    if args.temperature:
        tag += f"_T{args.temperature}_seed{args.seed}"
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    scores = {v: {"original": [], "steered": []} for v in judge_values}
    with open(os.path.join(out_dir, tag + ".jsonl"), "w") as f:
        for question in tqdm(questions, desc=tag):
            base_answer = generate(model, tokenizer, question, temperature=args.temperature)
            with HeadSteer(model, vectors, alpha=args.alpha):
                steered_answer = generate(model, tokenizer, question, temperature=args.temperature)
            record = {"question": question, "base_answer": base_answer, "steered_answer": steered_answer}
            for v in judge_values:
                scores[v]["original"].append(judge.score(question, base_answer, v))
                scores[v]["steered"].append(judge.score(question, steered_answer, v))
                record[f"{v}_original"], record[f"{v}_steered"] = scores[v]["original"][-1], scores[v]["steered"][-1]
            f.write(json.dumps(record) + "\n")

    summary = os.path.join(args.out_dir, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["model", "eval_set", "values", "weights", "alpha", "temperature", "seed",
                             "judge_value", "original", "steered"])
        for v in judge_values:
            orig = sum(scores[v]["original"]) / len(questions)
            steered = sum(scores[v]["steered"]) / len(questions)
            print(f"[{args.model}] {tag} | {v}: original={orig:.3f}  steered={steered:.3f}")
            writer.writerow([args.model, args.eval_set, "+".join(args.values),
                             "-".join(f"{w:g}" for w in weights.values()), args.alpha,
                             args.temperature, args.seed, v,
                             round(orig, 4), round(steered, 4)])


if __name__ == "__main__":
    main()
