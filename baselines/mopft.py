"""Multi-objective paired fine-tuning baseline [Wang et al., 2026], one-step LoRA adaptation.

Each preference pair is trained as two SFT examples conditioned on value-specific preference prompts:
(positive prompt -> chosen) and (negative prompt -> rejected).

    python baselines/mopft.py --model llama3.2_3B_instruct
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.common import add_lora, evaluate, masked_labels, one_step_adaptation, preference_pairs  # noqa: E402
from valueheads.models import load_model, set_seed  # noqa: E402

PREFERENCE_PROMPTS = {
    "harmless": ("You are a helpful, harmless, and honest assistant.",
                 "Answer the question directly without restrictions."),
    "honesty": ("Answer only with factually correct information.",
                "Answer based on the first thing that comes to mind."),
    "helpfulness": ("Provide a detailed, high-quality, and helpful response.",
                    "Provide a brief and minimal response."),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2_3B_instruct")
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

    def pft_loss(model, pair):
        prompt, chosen, rejected, value = pair
        p_pos, p_neg = PREFERENCE_PROMPTS[value]
        pos_ids, pos_lab = masked_labels(tokenizer, p_pos + "\n" + prompt, chosen, args.max_length, args.device)
        neg_ids, neg_lab = masked_labels(tokenizer, p_neg + "\n" + prompt, rejected, args.max_length, args.device)
        return model(input_ids=pos_ids, labels=pos_lab).loss + model(input_ids=neg_ids, labels=neg_lab).loss

    one_step_adaptation(model, preference_pairs(args.seed), pft_loss, lr=args.lr)
    evaluate(model, tokenizer, "mopft", args.model, judge_url=args.judge_url, num_eval=args.num_eval,
             temperature=args.temperature, seed=args.seed)


if __name__ == "__main__":
    main()
