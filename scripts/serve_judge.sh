#!/bin/bash
# Serve the LLM judge (Qwen3-30B-A3B-Instruct-2507) behind an OpenAI-compatible API on port 30000.
# All evaluation scripts read it via --judge_url (default http://localhost:30000/v1).
GPU=${GPU:-1}
CUDA_VISIBLE_DEVICES=$GPU vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --gpu_memory_utilization 0.8 \
    --max_model_len 4096 \
    --host 127.0.0.1 \
    --port 30000
