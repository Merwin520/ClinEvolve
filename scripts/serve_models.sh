#!/bin/bash
# Serve the task model (Qwen3.5-9B) and the repairer model (Qwen3.5-27B) with vLLM, as used in the paper.
# Usage: scripts/serve_models.sh task PORT GPU     e.g. scripts/serve_models.sh task 8801 1
#        scripts/serve_models.sh meta PORT GPU     e.g. scripts/serve_models.sh meta 8500 0
# Model weights: Qwen/Qwen3.5-9B and Qwen/Qwen3.5-27B (Hugging Face). Thinking mode is disabled.
set -euo pipefail
KIND=$1; PORT=$2; export CUDA_VISIBLE_DEVICES=$3
export VLLM_API_KEY=${VLLM_API_KEY:-dummy}
if [ "$KIND" = task ]; then
  exec vllm serve "${TASK_MODEL_PATH:-Qwen/Qwen3.5-9B}" --host 127.0.0.1 --port "$PORT" --served-model-name qwen3.5-9b \
    --dtype auto --gpu-memory-utilization 0.92 --max-model-len 32768 --max-num-seqs 96 --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}'
else
  exec vllm serve "${META_MODEL_PATH:-Qwen/Qwen3.5-27B}" --host 127.0.0.1 --port "$PORT" --served-model-name qwen3.5-27b \
    --dtype auto --gpu-memory-utilization 0.92 --max-model-len 32768 --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}'
fi
