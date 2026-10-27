"""Multi-objective DPO baseline [Zhou et al., 2023], one-step LoRA adaptation on 3H preference pairs.

    python baselines/modpo.py --model llama3.2_3B_instruct
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.common import add_lora, evaluate, masked_labels, one_step_adaptation, preference_pairs  # noqa: E402
from valueheads.models import load_model, set_seed  # noqa: E402


def mean_token_logprob(model, input_ids, labels):
    logits = model(input_ids=input_ids).logits[:, :-1]
    labels = labels[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(2, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float()
    return (logp * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2_3B_instruct")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="sample instead of greedy decoding")
    parser.add_argument("--judge_url", default="http://localhost:30000/v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    model, tokenizer = load_model(args.model, args.device)
    model = add_lora(model)

    def dpo_loss(model, pair):
        prompt, chosen, rejected, _ = pair
        c_ids, c_lab = masked_labels(tokenizer, prompt, chosen, args.max_length, args.device)
        r_ids, r_lab = masked_labels(tokenizer, prompt, rejected, args.max_length, args.device)
        with torch.no_grad(), model.disable_adapter():
            ref = mean_token_logprob(model, c_ids, c_lab) - mean_token_logprob(model, r_ids, r_lab)
        policy = mean_token_logprob(model, c_ids, c_lab) - mean_token_logprob(model, r_ids, r_lab)
        return -F.logsigmoid(args.beta * (policy - ref)).mean()

    one_step_adaptation(model, preference_pairs(args.seed), dpo_loss, lr=args.lr)
    evaluate(model, tokenizer, "modpo", args.model, judge_url=args.judge_url, num_eval=args.num_eval,
             temperature=args.temperature, seed=args.seed)


if __name__ == "__main__":
    main()
