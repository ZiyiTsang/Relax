#!/usr/bin/env bash
# Usage: bash $0 <absolute-model-path> <absolute-prompt-data-path>
set -euo pipefail

MODEL_ARGS=(
    --hf-checkpoint "$1"
    --ref-load "$1"
    --megatron-to-hf-mode bridge
    --swiglu
    --num-layers 36
    --hidden-size 2048
    --ffn-hidden-size 11008
    --num-attention-heads 16
    --group-query-attention
    --num-query-groups 2
    --use-rotary-position-embeddings
    --disable-bias-linear
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --rotary-base 1000000
    --vocab-size 151936
    --kv-channels 128
)

ROLLOUT_ARGS=(
    --prompt-data "$2"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type math
    --num-rollout 100
    --rollout-batch-size 4
    --n-samples-per-prompt 8
    --rollout-max-response-len 1024
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size 32
)

DR_GRPO_ARGS=(
    --advantage-estimator grpo
    --calculate-per-token-loss
    --kl-coef 0.0
    --kl-loss-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --disable-grpo-std-normalization
    --pg-loss-aggregation seq-mean-token-sum-norm
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-dynamic-batch-size
    --max-tokens-per-gpu 8192
    --log-probs-max-tokens-per-gpu 8192
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.7
)

ray job submit --no-wait -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 4], "rollout": [1, 4], "advantages": [1, 0]}' \
    --colocate \
    --save checkpoints/qwen25-3b-dr-grpo \
    --save-interval 50 \
    "${MODEL_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${DR_GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}"
