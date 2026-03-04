#!/bin/bash

# ============================================================
# UniScientist - Agentic Inference Script
# ============================================================
# This script runs a single round of agentic inference.
# Run it multiple times to collect multiple rollouts for
# later aggregation via inference_local_aggregate.py.
# ============================================================

# --- Tool API Keys (required for web search & page visit) ---
export SUMMARY_MODEL_NAME=""          # Model name used by the visit tool to summarize webpage content (via OpenRouter)
export JINA_API_KEYS=""               # Jina Reader API key(s) for webpage reading, comma-separated if multiple
export SERPER_KEY_ID=""               # Serper API key for Google web search and Google Scholar search
export OPENROUTER_BASE_URL=""         # OpenRouter API base URL (e.g., https://openrouter.ai/api/v1)
export OPENROUTER_API_KEY=""          # OpenRouter API key

# --- Local LLM Server ---
export LOCAL_BASE_URL="http://localhost:8000/v1"  # vLLM server endpoint (see local_deploy.sh)

# --- Task Configuration ---
export BENCHMARK="research"           # Benchmark / task name, used for naming the output file
export STORED_MODEL_NAME=""           # Model identifier, used for naming the output file
export DATA_PATH=""                   # Path to input data file (.jsonl), each line: {"problem": "...", "answer": "..."}
export ROLLOUT_COUNT=1                # Number of rollouts per question in this run

# --- Concurrency ---
export LLM_MAX_CONCURRENCY=32        # Max concurrent LLM requests
export TOOL_MAX_CONCURRENCY=32        # Max concurrent tool calls (search, visit, etc.)

python inference_local_qwen.py
