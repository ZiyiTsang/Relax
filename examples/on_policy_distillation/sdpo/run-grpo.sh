#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

set -a
[ -f .env ] && source .env
set +a

source scripts/models/qwen3-4B-Instruct-2507.sh

export PYTHONPATH="${PROJECT_ROOT}${MEGATRON:+:${MEGATRON}}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RELAX_OPD_PER_POS_TOKEN_IDS=1
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDA_DEVICE_MAX_CONNECTIONS,RELAX_OPD_PER_POS_TOKEN_IDS"

relax_python="${RELAX_PYTHON:-python3}"
student_model_path="${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH to the student checkpoint}"
data_path="${DATA_PATH:?Set DATA_PATH to a prepared SciKnowEval or ToolAlpaca JSONL file}"
actor_gpus="${ACTOR_GPUS:-2}"
rollout_gpus="${ROLLOUT_GPUS:-${actor_gpus}}"
tp_size="${TP_SIZE:-2}"
cp_size="${CP_SIZE:-1}"
now="$(date "+%Y-%m-%d-%H:%M:%S")"
experiment_name="${EXPERIMENT_NAME:-grpo-${now}}"

resource_json="{\"actor\": [1, ${actor_gpus}], \"rollout\": [1, ${rollout_gpus}]}"

command=(
    "${relax_python}" -m relax.entrypoints.train
    --resource "${resource_json}"
    --max-staleness 0
    --num-data-storage-units 1
    --colocate
    --offload
    --use-health-check
    --actor-num-gpus-per-node "${actor_gpus}"
    --num-gpus-per-node "${NUM_GPUS:-${actor_gpus}}"
    "${MODEL_ARGS[@]}"
    --tensor-model-parallel-size "${tp_size}"
    --context-parallel-size "${cp_size}"
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --calculate-per-token-loss
    --advantage-estimator grpo
    --eps-clip 0.2
    --eps-clip-high 0.28
    --optimizer adam
    --lr "${LEARNING_RATE:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.01
    --clip-grad 1.0
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
    --rollout-num-gpus "${rollout_gpus}"
    --rollout-num-gpus-per-engine "${ROLLOUT_ENGINE_GPUS:-${rollout_gpus}}"
    --sglang-load-format dummy
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.8}"
    --sglang-disable-cuda-graph
    --tb-experiment-name "${experiment_name}"
    --skip-eval-before-train
    --hf-checkpoint "${student_model_path}"
    --megatron-to-hf-mode bridge
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
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-2048}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-2048}"
    --rollout-temperature 1.0
    --use-fault-tolerance
)

if [ "${RELAX_DRY_RUN:-0}" = 1 ]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
