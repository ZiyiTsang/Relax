#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source "$(dirname "${BASH_SOURCE[0]}")/run-2gpu.sh"
source scripts/models/qwen3-4B-Instruct-2507.sh

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RELAX_OPD_PER_POS_TOKEN_IDS=1

relax_python="${RELAX_PYTHON}"
student_model="${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH}"
teacher_model="${TEACHER_MODEL_PATH:-${student_model}}"
data_path="${DATA_PATH:-${SDPO_DATA_ROOT:?Set SDPO_DATA_ROOT}/sciknoweval/chemistry/train.jsonl}"
actor_gpus="${ACTOR_GPUS:-2}"
rollout_gpus="${ROLLOUT_GPUS:-1}"
teacher_gpus="${TEACHER_GPUS:-1}"
now="$(date '+%Y-%m-%d-%H:%M:%S')"
experiment_name="${EXPERIMENT_NAME:-sdpo-sciknoweval-chemistry-${now}}"

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
    --num-rollout "${NUM_ROLLOUT:-200}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-32}"
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-10240}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-18944}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    --use-fault-tolerance
)

EVAL_ARGS=(
    --skip-eval-before-train
)

OPD_ARGS=(
    --use-opd
    --opd-feedback-class "relax.utils.opd.feedback.SciKnowEvalSDPOFeedback"
    --opd-type sglang
    --teacher-hf-checkpoint "${teacher_model}"
    --teacher-num-gpus-per-engine "${TEACHER_ENGINE_GPUS:-${teacher_gpus}}"
    --teacher-sglang-mem-fraction-static "${TEACHER_MEM_FRACTION:-0.5}"
    --teacher-sglang-chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
    --teacher-sglang-max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-16}"
    --teacher-sglang-disable-cuda-graph
    --opd-loss-coef 1.0
    --opd-kl-coef 0.0
    --opd-disable-rl-reward
    --opd-token-selection student_topk
    --opd-log-prob-top-k "${OPD_TOP_K:-100}"
    --opd-kl-type jsd
    --opd-jsd-alpha 0.5
    --opd-norm-mode tail
    --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-120}"
    --use-rollout-logprobs
    --sdpo-teacher-update-mode "${SDPO_TEACHER_UPDATE_MODE:-static}"
    --sdpo-teacher-ema-alpha "${SDPO_TEACHER_EMA_ALPHA:-0.01}"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LEARNING_RATE:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.01
    --clip-grad 1.0
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE:-2}"
    --context-parallel-size "${CP_SIZE:-1}"
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-18944}"
)

SGLANG_ARGS=(
    --rollout-num-gpus "${rollout_gpus}"
    --rollout-num-gpus-per-engine "${ROLLOUT_ENGINE_GPUS:-${rollout_gpus}}"
    --sglang-load-format dummy
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.45}"
    --sglang-disable-cuda-graph
)

MISC_ARGS=(
    --resource "{\"actor\": [1, ${actor_gpus}], \"rollout\": [1, ${rollout_gpus}], \"teacher\": [1, ${teacher_gpus}]}"
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
    --offload
    --use-health-check
    --actor-num-gpus-per-node "${actor_gpus}"
    --num-gpus-per-node "${NUM_GPUS:-${actor_gpus}}"
    --tb-experiment-name "${experiment_name}"
)

exec "${relax_python}" -m relax.entrypoints.train \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" \
    "${OPD_ARGS[@]}" "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}"
