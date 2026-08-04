#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Qwen2.5-3B 2xGPU Dr.GRPO colocate training script (single-node dev run).
#
# Dr.GRPO (https://arxiv.org/abs/2503.20783) drops GRPO's group-wise std
# normalization and normalizes the policy-gradient token loss by a fixed
# response-length scale (--pg-loss-scale-factor). The scale defaults to
# --rollout-max-response-len.
#
# Usage:
#   EXP_DIR=/path/to/exp bash examples/algorithms/dr_grpo/run-qwen25-3b-dr-grpo-2xgpu-colocate.sh

set -ex
set -o pipefail

now=$(date "+%Y-%m-%d-%H:%M:%S")

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${SCRIPT_DIR}/../../../scripts/entrypoint/local.sh"
fi
source "${MODEL_CONFIG_DIR}/qwen25-3B.sh"

PROJECT_NAME="${PROJECT_NAME:=Relax/dev/dr-grpo}"
EXP_DIR="${EXP_DIR:-${SCRIPT_DIR}/../../../../exps}"
MODEL_DIR="${MODEL_DIR:-${EXP_DIR}}"
DATA_DIR="${DATA_DIR:-${EXP_DIR}}"
NUM_ROLLOUT="${NUM_ROLLOUT:=100}"

CKPT_ARGS=(
   --hf-checkpoint ${MODEL_DIR}/Qwen2.5-3B/
   --ref-load ${MODEL_DIR}/Qwen2.5-3B/
   --megatron-to-hf-mode bridge
   --save ${EXP_DIR}/Qwen2.5-3B_mcore_2xgpu/
   --save-interval 100
   --max-actor-ckpt-to-keep 1
)

PROMPT_SET="${PROMPT_SET:-${DATA_DIR}/math_deepmath_deal.jsonl}"
EVAL_PROMPT_SET="${EVAL_PROMPT_SET:-${DATA_DIR}/aime24/test.jsonl}"

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key ground_truth
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --rollout-top-p 1.0
   --rollout-top-k -1
   --global-batch-size 32
   --reward-num-workers 2
)

DR_GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-grpo-std-normalization
   --pg-loss-aggregation seq-mean-token-sum-norm
   --calculate-per-token-loss
   --kl-coef 0.0
   --kl-loss-coef 0.0
   --entropy-coef 0.0
   --eps-clip 0.2
)

EVAL_ARGS=(
   --log-passrate
   --skip-eval-before-train
   --eval-interval 5
   --eval-prompt-data aime ${EVAL_PROMPT_SET}
   --eval-input-key problem
   --eval-label-key answer
   --n-samples-per-eval-prompt 8
   --eval-max-response-len 8192
   --eval-temperature 1.0
   --eval-top-p 1.0
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
   --tensor-model-parallel-size 2
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --sequence-parallel
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 8192
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

WANDB_ARGS=(
   --use-wandb
   --wandb-group ${WANDB_RUN_GROUP:-DR-GRPO}
)

mkdir -p log
ray job submit ${RAY_NO_WAIT:+--no-wait} --address="http://${HOST_IP:-127.0.0.1}:8265" \
   ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -m relax.entrypoints.train \
   --resource '{"actor": [1, 2], "rollout": [1, 2], "advantages": [1, 0]}' \
   --colocate \
   --max-staleness 0 \
   --num-data-storage-units 1 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${DR_GRPO_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${WANDB_ARGS[@]}"  2>&1 | tee log/run-dr-grpo-qwen25-3B-2xgpu-${now}.log
