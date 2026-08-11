#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source scripts/models/qwen3-4B-Instruct-2507.sh

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

export RELAX_OPD_PER_POS_TOKEN_IDS=1

student_model="${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH}"
teacher_model="${TEACHER_MODEL_PATH:-${student_model}}"
data_path="${DATA_PATH:-${SDPO_DATA_ROOT:?Set SDPO_DATA_ROOT}/tooluse/train.jsonl}"
now="$(date '+%Y-%m-%d-%H:%M:%S')"
experiment_name="${EXPERIMENT_NAME:-sdpo-tooluse-${now}}"

CKPT_ARGS=(
    --hf-checkpoint "${student_model}"
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${data_path}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --group-rm
    --custom-rm-path examples.on_policy_distillation.sdpo.reward.score
    --reward-key score
    --num-rollout 2
    --rollout-batch-size 1
    --n-samples-per-prompt 2
    --global-batch-size 2
    --rollout-max-prompt-len 10240
    --rollout-max-response-len 8192
    --rollout-max-context-len 18944
    --rollout-temperature 1.0
    --use-fault-tolerance
)

EVAL_ARGS=(
    --skip-eval-before-train
)

OPD_ARGS=(
    --use-opd
    --opd-feedback-class "relax.utils.opd.feedback.ToolUseSDPOFeedback"
    --opd-type sglang
    --teacher-hf-checkpoint "${teacher_model}"
    --teacher-num-gpus-per-engine 1
    --teacher-sglang-mem-fraction-static 0.5
    --teacher-sglang-chunked-prefill-size 4096
    --teacher-sglang-max-running-requests 16
    --teacher-sglang-disable-cuda-graph
    --opd-loss-coef 1.0
    --opd-kl-coef 0.0
    --opd-disable-rl-reward
    --opd-token-selection student_topk
    --opd-log-prob-top-k 100
    --opd-kl-type jsd
    --opd-jsd-alpha 0.5
    --opd-norm-mode tail
    --opd-teacher-timeout-s 120
    --use-rollout-logprobs
    --sdpo-teacher-update-mode ema
    --sdpo-teacher-ema-alpha 0.01
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --clip-grad 1.0
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --context-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu 18944
)

SGLANG_ARGS=(
    --rollout-num-gpus 1
    --rollout-num-gpus-per-engine 1
    --sglang-load-format dummy
    --sglang-mem-fraction-static 0.45
    --sglang-disable-cuda-graph
)

MISC_ARGS=(
    --resource '{"actor": [1, 2], "rollout": [1, 1], "teacher": [1, 1]}'
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
    --offload
    --use-health-check
    --actor-num-gpus-per-node 2
    --num-gpus-per-node 2
    --tb-experiment-name "${experiment_name}"
)

exec python -m relax.entrypoints.train \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" \
    "${OPD_ARGS[@]}" "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
