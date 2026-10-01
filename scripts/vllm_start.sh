#!/bin/bash
# Start vLLM on your NVIDIA GPU node with prefix caching (Layer 3) enabled.
# Adjust model and memory utilization for your VRAM.
set -e
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
