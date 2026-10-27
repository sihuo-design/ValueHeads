"""Step 3: ablate value heads and measure the effect on value expression (Sec. 4.2-4.3, Table 1, Fig. 3-4).

Mask the top 15% honesty heads and evaluate on held-out TruthfulQA questions:
    python mask_heads.py --model qwen3-8B --eval_set honesty --select topk
Same number of random heads:
    python mask_heads.py --model qwen3-8B --eval_set honesty --select random --seed 0
Cross-value: mask harmlessness heads, evaluate honesty:
    python mask_heads.py --model qwen3-8B --eval_set honesty --head_value harmless

The score is the fraction of answers the LLM judge labels as exhibiting the value.
"""
import argparse
import csv
import json
import os
import random

from tqdm import tqdm

from valueheads.data import EVAL_SETS, VALUES, eval_set
from valueheads.generation import generate
from valueheads.hooks import HeadMask
from valueheads.judge import Judge
from valueheads.models import load_model, model_geometry, set_seed
from valueheads.probing import group_by_layer, load_head_scores, top_heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1_8B_instruct")
    parser.add_argument("--eval_set", required=True, choices=[k for k, v in EVAL_SETS.items() if v])
    parser.add_argument("--head_value", default=None, choices=VALUES,
                        help="value whose heads are masked (default: the value of --eval_set)")
    parser.add_argument("--select", default="topk", choices=["topk", "random"])
    parser.add_argument("--ratio", type=float, default=0.15, help="fraction of all heads to mask")
    parser.add_argument("--eps", type=float, default=1e-4, help="scale applied to masked head outputs")
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None,
                        help="sample (top-p 0.9) instead of greedy decoding; vary --seed across runs")
    parser.add_argument("--score_dir", default="results/heads")
    parser.add_argument("--out_dir", default="results/mask")
    parser.add_argument("--judge_url", default="http://localhost:30000/v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    judge_value = EVAL_SETS[args.eval_set]
    head_value = args.head_value or judge_value

    model, tokenizer = load_model(args.model, args.device)
    num_layers, num_heads, _ = model_geometry(model.config)
    k = int(num_layers * num_heads * args.ratio)
    if args.select == "topk":
        heads = top_heads(load_head_scores(args.score_dir, args.model, head_value), k)
    else:
        heads = random.Random(args.seed).sample(range(num_layers * num_heads), k)
    heads = group_by_layer(heads, num_heads)

    questions = eval_set(args.eval_set, args.num_eval)
    judge = Judge(args.judge_url)

    tag = f"{args.eval_set}_mask-{head_value}_{args.select}_k{k}"
    if args.select == "random" or args.temperature:
        tag += f"_seed{args.seed}"
    if args.temperature:
        tag += f"_T{args.temperature}"
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    base_scores, masked_scores = [], []
    with open(os.path.join(out_dir, tag + ".jsonl"), "w") as f:
        for question in tqdm(questions, desc=tag):
            base_answer = generate(model, tokenizer, question, temperature=args.temperature)
            with HeadMask(model, heads, eps=args.eps):
                masked_answer = generate(model, tokenizer, question, temperature=args.temperature)
            base_scores.append(judge.score(question, base_answer, judge_value))
            masked_scores.append(judge.score(question, masked_answer, judge_value))
            f.write(json.dumps({"question": question,
                                "base_answer": base_answer, "base_score": base_scores[-1],
                                "masked_answer": masked_answer, "masked_score": masked_scores[-1]}) + "\n")

    base_acc = sum(base_scores) / len(questions)
    masked_acc = sum(masked_scores) / len(questions)
    print(f"[{args.model}] {tag}: original={base_acc:.3f}  masked={masked_acc:.3f}")
    summary = os.path.join(args.out_dir, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["model", "eval_set", "head_value", "select", "k", "seed", "temperature",
                             "original", "masked"])
        writer.writerow([args.model, args.eval_set, head_value, args.select, k, args.seed, args.temperature,
                         round(base_acc, 4), round(masked_acc, 4)])


if __name__ == "__main__":
    main()
