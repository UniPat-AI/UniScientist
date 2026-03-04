#!/bin/bash

# ============================================================
# UniScientist - Local Model Deployment
# ============================================================
# Deploy a local LLM using vLLM as an OpenAI-compatible server.
# This must be running before executing inference_local_qwen.sh
# or inference_local_aggregate.py.
# ============================================================

MODEL_PATH=""  # Path to the local model weights (e.g., /path/to/UniScientist-30B-A3B)

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port 8000 \
    --max-model-len 131072 \
    --tensor-parallel-size 2 \
    --data-parallel-size 4 \
    --gpu-memory-utilization 0.8 > /tmp/vllm_local_qwen.log 2>&1 &
