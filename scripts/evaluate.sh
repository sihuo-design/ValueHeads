#!/bin/bash
# Steps 3-4: masking (Table 1, Table 12) and positive intervention (Tables 2-3) for one model.
# Requires scripts/localize.sh and a running judge (scripts/serve_judge.sh).
#   bash scripts/evaluate.sh qwen3-8B
set -e
MODEL=${1:-llama3.1_8B_instruct}
SINGLE_SETS="jailbreak malicious honesty helpfulness complexity verbosity coherence"

# Table 1 / Table 12: top-15% value heads vs. 5 random draws of the same number of heads
for SET in $SINGLE_SETS; do
    python mask_heads.py --model "$MODEL" --eval_set "$SET" --select topk
    for SEED in 0 1 2 3 4; do
        python mask_heads.py --model "$MODEL" --eval_set "$SET" --select random --seed "$SEED"
    done
done

# Fig. 4: mask the heads of one value, evaluate another
for HEAD_VALUE in harmless helpfulness complexity coherence verbosity honesty; do
    for SET in jailbreak helpfulness complexity coherence verbosity honesty; do
        python mask_heads.py --model "$MODEL" --eval_set "$SET" --head_value "$HEAD_VALUE" --select topk
    done
done

# Table 2: single-value intervention
for VALUE_SET in "harmless jailbreak" "honesty honesty" "helpfulness helpfulness" \
                 "complexity complexity" "verbosity verbosity" "coherence coherence"; do
    set -- $VALUE_SET
    for ALPHA in 0.3 0.5 1.0; do
        python steer.py --model "$MODEL" --values "$1" --eval_set "$2" --alpha "$ALPHA"
    done
done

# Table 3: multi-value intervention with lambda_k = 1/K
for SET in beavertails hh-rlhf; do
    for ALPHA in 0.3 0.5 1.0; do
        python steer.py --model "$MODEL" --eval_set "$SET" --alpha "$ALPHA" \
            --values harmless helpfulness complexity coherence verbosity honesty
    done
done
