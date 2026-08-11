#!/bin/bash
nohup python -u run.py \
    --test_file ./data/tasks_test.jsonl \
    --api_key YOUR_OPENAI_API_KEY \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --seed 42 > test_tasks.log &

# To run against an OpenAI-compatible provider such as MiniMax, replace the
# --api_key and --api_model lines above with, e.g.:
#   --api_key YOUR_MINIMAX_API_KEY \
#   --api_provider minimax \
#   --api_model MiniMax-M3 \
# Use MiniMax-M2.7 together with --text_only for text-only runs.
# For the China region endpoint use --api_provider minimax_cn instead.
# Alternatively, pass --api_base_url https://api.minimax.io/v1 directly.
