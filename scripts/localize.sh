#!/bin/bash
# Steps 1-2: extract head activations and rank value heads for one model.
#   bash scripts/localize.sh qwen3-8B
set -e
MODEL=${1:-llama3.1_8B_instruct}

for VALUE in harmless helpfulness honesty complexity verbosity coherence; do
    for SPLIT in train val; do
        python extract_activations.py --model "$MODEL" --dataset "$VALUE" --split "$SPLIT"
    done
    python probe_heads.py --model "$MODEL" --value "$VALUE"
done

# unlabeled prompt sets, only needed to estimate the steering std (steer.py)
for SET in jailbreak malicious beavertails; do
    python extract_activations.py --model "$MODEL" --dataset "$SET" --split val
done
