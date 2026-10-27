"""Shared code for the training-based multi-objective baselines (MODPO, MOPFT; Table 4, Table 15).

For a fair comparison with inference-time intervention, both baselines perform a one-step parameter
adaptation: LoRA gradients are accumulated over all preference pairs, then a single optimizer step is taken.
"""
import csv
import json
import os
import random
import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from valueheads.data import eval_set, first_qa, load_helpsteer, load_hhrlhf, single_turn, truthfulqa_split  # noqa: E402
from valueheads.generation import generate  # noqa: E402
from valueheads.judge import Judge  # noqa: E402

VALUES = ["harmless", "honesty", "helpfulness"]


# ----------------------------------------------------------------------------- preference pairs
def pairs_harmless(max_pairs=1000):
    pairs = []
    for ex in load_hhrlhf("train"):
        if len(pairs) >= max_pairs:
            break
        if not single_turn(ex):
            continue
        q, chosen = first_qa(ex["chosen"])
        _, rejected = first_qa(ex["rejected"])
        if None in (q, chosen, rejected):
            continue
        pairs.append((f"\n\nHuman: {q}\n\nAssistant:", f" {chosen}", f" {rejected}", "harmless"))
    return pairs


def pairs_honesty():
    pairs = []
    for ex in truthfulqa_split("train"):
        if ex["correct_answers"] and ex["incorrect_answers"]:
            pairs.append((f"Question: {ex['question']} Answer:", f" {ex['correct_answers'][0]}",
                          f" {ex['incorrect_answers'][0]}", "honesty"))
    return pairs


def pairs_helpfulness():
    pos, neg = {}, {}
    for ex in load_helpsteer("train"):
        if ex["helpfulness"] == 4:
            pos.setdefault(ex["prompt"], []).append(ex["response"])
        elif ex["helpfulness"] in (0, 1):
            neg.setdefault(ex["prompt"], []).append(ex["response"])
    return [(f"Question: {p}\nAnswer:", f" {pos[p][0]}", f" {neg[p][0]}", "helpfulness")
            for p in pos if p in neg]


def preference_pairs(seed=42):
    """(prompt, chosen, rejected, value) tuples over harmlessness, honesty and helpfulness, shuffled."""
    pairs = pairs_harmless() + pairs_honesty() + pairs_helpfulness()
    random.Random(seed).shuffle(pairs)
    print(f"{len(pairs)} preference pairs")
    return pairs


# ----------------------------------------------------------------------------- training
def add_lora(model):
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
                        target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def one_step_adaptation(model, pairs, loss_fn, lr=5e-5):
    """Accumulates loss_fn(model, pair) / len(pairs) over all pairs, then takes one AdamW step."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    optimizer.zero_grad()
    model.train()
    total = 0.0
    for pair in tqdm(pairs, desc="accumulating gradients"):
        loss = loss_fn(model, pair)
        (loss / len(pairs)).backward()
        total += loss.item()
    optimizer.step()
    model.eval()
    print(f"mean loss: {total / len(pairs):.4f}")


def masked_labels(tokenizer, prefix, completion, max_length, device):
    """input_ids and labels for `prefix + completion`, with the prefix tokens masked out (-100)."""
    prefix_len = min(len(tokenizer.encode(prefix, add_special_tokens=True)), max_length)
    ids = torch.tensor(tokenizer.encode(prefix + completion, add_special_tokens=True)[:max_length]).unsqueeze(0)
    labels = ids.clone()
    labels[0, :prefix_len] = -100
    return ids.to(device), labels.to(device)


# ----------------------------------------------------------------------------- evaluation
def evaluate(model, tokenizer, method, model_name, out_dir="results/baselines",
             judge_url="http://localhost:30000/v1", num_eval=None, temperature=None, seed=42):
    """Scores base (adapter disabled) and adapted answers on HH-RLHF and BeaverTails for the 3H values."""
    judge = Judge(judge_url)
    os.makedirs(os.path.join(out_dir, model_name), exist_ok=True)
    summary = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as fs:
        writer = csv.writer(fs)
        if new:
            writer.writerow(["model", "method", "eval_set", "temperature", "seed", "judge_value", "original", "adapted"])
        for name in ["hh-rlhf", "beavertails"]:
            questions = eval_set(name, num_eval)
            scores = {v: {"original": [], "adapted": []} for v in VALUES}
            suffix = f"_T{temperature}_seed{seed}" if temperature else ""
            with open(os.path.join(out_dir, model_name, f"{name}_{method}{suffix}.jsonl"), "w") as f:
                for question in tqdm(questions, desc=f"eval {name}"):
                    with model.disable_adapter():
                        base_answer = generate(model, tokenizer, question, temperature=temperature)
                    adapted_answer = generate(model, tokenizer, question, temperature=temperature)
                    record = {"question": question, "base_answer": base_answer, "adapted_answer": adapted_answer}
                    for v in VALUES:
                        scores[v]["original"].append(judge.score(question, base_answer, v))
                        scores[v]["adapted"].append(judge.score(question, adapted_answer, v))
                        record[f"{v}_original"] = scores[v]["original"][-1]
                        record[f"{v}_adapted"] = scores[v]["adapted"][-1]
                    f.write(json.dumps(record) + "\n")
            for v in VALUES:
                orig = sum(scores[v]["original"]) / len(questions)
                adapted = sum(scores[v]["adapted"]) / len(questions)
                print(f"[{model_name}] {method} {name} | {v}: original={orig:.3f}  adapted={adapted:.3f}")
                writer.writerow([model_name, method, name, temperature, seed, v, round(orig, 4), round(adapted, 4)])
