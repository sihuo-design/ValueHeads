# Towards Multi-Human-Value Alignment via Value Localization in LLMs

Official code for the NeurIPS 2026 paper
**"Towards Multi-Human-Value Alignment via Value Localization in LLMs"**
by Xueqi Ma, Yanbei Jiang, Xingjun Ma, James Bailey and Sarah Monazam Erfani.

We localize *value heads*, the attention heads that mediate six value dimensions (harmlessness,
honesty, helpfulness, complexity, verbosity and coherence), using per-head linear probes. We then
steer these heads at inference time to align an LLM with one or several values, with no retraining.

```
 probing data ──► extract_activations.py ──► probe_heads.py ──► value heads (ranked per value)
                                                                    │
                       mask_heads.py   ◄────────────────────────────┤  ablation: are they causal?
                       steer.py        ◄────────────────────────────┘  single / multi-value intervention
```

## Installation

```bash
conda create -n valueheads python=3.10 -y && conda activate valueheads
pip install -r requirements.txt
huggingface-cli login     # needed for the gated Llama models
```

All results are scored by an LLM judge (Qwen3-30B-A3B-Instruct-2507) behind an OpenAI-compatible
endpoint. Start it on a separate GPU with [vLLM](https://github.com/vllm-project/vllm):

```bash
GPU=1 bash scripts/serve_judge.sh      # serves http://localhost:30000/v1
```

Supported models (`--model`): `llama3.1_8B_instruct`, `llama3.2_3B_instruct`, `qwen3-8B`, `qwen3-4B`,
`yi-1.5-9B`, `yi-1.5-6B`. Any HF decoder whose layers expose `model.layers[i].self_attn.o_proj`
can be passed by its HF path.

## Quick start

```bash
# 1-2. localize value heads (activations + probes for all six values)
bash scripts/localize.sh qwen3-8B

# 3. mask the top-15% honesty heads vs. the same number of random heads
python mask_heads.py --model qwen3-8B --eval_set honesty --select topk
python mask_heads.py --model qwen3-8B --eval_set honesty --select random --seed 0

# 4a. single-value intervention
python steer.py --model qwen3-8B --values honesty --eval_set honesty --alpha 1.0

# 4b. multi-value intervention (lambda_k = 1/K by default, or set --weights)
python steer.py --model qwen3-8B --eval_set beavertails --alpha 1.0 \
    --values harmless helpfulness complexity coherence verbosity honesty
```

`bash scripts/evaluate.sh <model>` runs the full set of masking and intervention experiments.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Head activations | `extract_activations.py --dataset <value> --split train/val` | `features/{model}_{dataset}_{split}_head_wise.npy`, `_labels.npy` |
| 2. Probing | `probe_heads.py --value <value>` | `results/heads/{model}_{value}_heads.json` (head index → test accuracy, sorted) |
| 3. Ablation | `mask_heads.py --eval_set <set> [--head_value <value>] --select topk/random` | `results/mask/summary.csv` |
| 4. Intervention | `steer.py --values <v1> [<v2> ...] --eval_set <set>` | `results/steer/summary.csv` |

A head index is `layer * num_heads + head`. Head activations are the last-token inputs of each
layer's `o_proj`, that is, the concatenated per-head attention outputs. Reading, masking and steering
heads all use plain PyTorch forward pre-hooks on `o_proj` ([valueheads/hooks.py](valueheads/hooks.py)).

**Probing data** (Sec. 3.1). The question and answer are concatenated and the last-token activation
is probed. For each head, a logistic regression is fit on its activation concatenated with the mean
activation of its layer.

| Value | Dataset | Positive / negative |
|---|---|---|
| harmlessness | HH-RLHF, single-turn | chosen / rejected |
| honesty | TruthfulQA (80/20 split, seed 42) | correct / incorrect answers |
| helpfulness, coherence | HelpSteer2 | rating 4 / rating 0–1 |
| complexity, verbosity | HelpSteer2 | rating 3–4 / rating 0 |

**Evaluation sets** (`--eval_set`): `jailbreak` (JailbreakBench) and `malicious` (MaliciousInstruct),
both scored for harmlessness; `honesty` (the held-out 20% of TruthfulQA questions); `helpfulness`,
`complexity`, `verbosity` and `coherence` (HelpSteer2 validation prompts); and `hh-rlhf` and
`beavertails` (multi-value sets, scored on all values).

**Masking** scales the output of the selected heads by `eps = 1e-4`. **Steering** adds
`alpha * sigma * dir` to each selected head. Here `dir` is the normalized mass-mean direction
(positive minus negative mean) and `sigma` is the std of activations projected onto it. For several
values, the directions of a shared head are combined as `sum_k lambda_k dir_k`
([valueheads/directions.py](valueheads/directions.py)).

## Repository layout

```
valueheads/            core library
  models.py            model registry / loading
  data.py              probing and evaluation datasets
  hooks.py             activation collection, head masking, head steering
  probing.py           per-head probes and head ranking
  directions.py        value directions and (multi-value) steering vectors
  judge.py             LLM-judge prompts and client
  generation.py        answer generation
extract_activations.py, probe_heads.py, mask_heads.py, steer.py
analysis/              figures and correlation / angle analyses
baselines/             MODPO and MOPFT (one-step LoRA adaptation)
scripts/               judge server and end-to-end run scripts
data/                  JailbreakBench and MaliciousInstruct prompts
```

<!-- ## Citation

```bibtex
@inproceedings{ma2026valuelocalization,
  title     = {Towards Multi-Human-Value Alignment via Value Localization in {LLMs}},
  author    = {Ma, Xueqi and Jiang, Yanbei and Ma, Xingjun and Bailey, James and Erfani, Sarah Monazam},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
``` -->


## Responsible use

Value heads can also be used to *suppress* safety behavior: masking the harmlessness heads makes
models comply with harmful requests (Appendix L). This code is released for research on
interpretability and alignment. Do not use it to deploy models with weakened safeguards.
