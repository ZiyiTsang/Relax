#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

repo="${REPO:-worktree}"
mode="${MODE:-sdpo}"
actor_gpus="${ACTOR_GPUS:-2}"

case "${repo}" in
    worktree)
        repo_root="${WORKTREE_ROOT}"
        relax_python="${RELAX_PYTHON:-python3}"
        megatron="${MEGATRON:?Set MEGATRON for the selected Relax checkout}"
        ;;
    main)
        repo_root="${MAIN_REPO_ROOT:?Set MAIN_REPO_ROOT for REPO=main}"
        relax_python="${MAIN_RELAX_PYTHON:?Set MAIN_RELAX_PYTHON for REPO=main}"
        megatron="${MAIN_MEGATRON:?Set MAIN_MEGATRON for REPO=main}"
        ;;
    *)
        echo "REPO must be main or worktree" >&2
        exit 2
        ;;
esac

case "${mode}" in
    grpo|sapo)
        rollout_gpus="${ROLLOUT_GPUS:-${actor_gpus}}"
        teacher_gpus="${TEACHER_GPUS:-0}"
        ;;
    opd|sdpo)
        rollout_gpus="${ROLLOUT_GPUS:-1}"
        teacher_gpus="${TEACHER_GPUS:-1}"
        ;;
    *)
        echo "MODE must be grpo, sapo, opd, or sdpo" >&2
        exit 2
        ;;
esac

if [[ "${mode}" == "sdpo" && "${repo}" != "worktree" ]]; then
    echo "MODE=sdpo is only available in the worktree checkout" >&2
    exit 2
fi

if [[ "${mode}" == "opd" || "${mode}" == "sdpo" ]]; then
    if (( rollout_gpus + teacher_gpus != actor_gpus )); then
        echo "OPD colocate requires rollout GPUs + teacher GPUs == actor GPUs" >&2
        exit 2
    fi
elif (( teacher_gpus != 0 || rollout_gpus != actor_gpus )); then
    echo "GRPO/SAPO colocate requires rollout GPUs == actor GPUs and no teacher" >&2
    exit 2
fi

cd "${repo_root}"
model_config_path="${MODEL_CONFIG_PATH:-${WORKTREE_ROOT}/scripts/models/qwen3-4B-Instruct-2507.sh}"
source "${model_config_path}"

student_model_path="${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH}"
data_path="${DATA_PATH:?Set DATA_PATH}"
teacher_model_path="${TEACHER_MODEL_PATH:-${student_model_path}}"
tp_size="${TP_SIZE:-2}"
cp_size="${CP_SIZE:-1}"
num_rollout="${NUM_ROLLOUT:-20}"
rollout_batch_size="${ROLLOUT_BATCH_SIZE:-2}"
n_samples_per_prompt="${N_SAMPLES_PER_PROMPT:-2}"
global_batch_size="${GLOBAL_BATCH_SIZE:-4}"
experiment_name="${EXPERIMENT_NAME:-${repo}-${mode}-2gpu}"

export PYTHONPATH="${repo_root}${megatron:+:${megatron}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export RELAX_OPD_PER_POS_TOKEN_IDS="${RELAX_OPD_PER_POS_TOKEN_IDS:-1}"
export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}CUDA_DEVICE_MAX_CONNECTIONS,RELAX_OPD_PER_POS_TOKEN_IDS"

resource_json="{\"actor\": [1, ${actor_gpus}], \"rollout\": [1, ${rollout_gpus}]"
if [[ "${mode}" == "opd" || "${mode}" == "sdpo" ]]; then
    resource_json+=", \"teacher\": [1, ${teacher_gpus}]"
fi
resource_json+="}"

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
    --optimizer adam
    --lr "${LEARNING_RATE:-1e-6}"
    --lr-decay-style constant
    --weight-decay 0.01
    --clip-grad 1.0
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-2048}"
    --rollout-num-gpus "${rollout_gpus}"
    --rollout-num-gpus-per-engine "${ROLLOUT_ENGINE_GPUS:-${rollout_gpus}}"
    --sglang-load-format dummy
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.45}"
    --sglang-disable-cuda-graph
    --sglang-enable-weights-cpu-backup
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-16}"
    --tb-experiment-name "${experiment_name}"
    --skip-eval-before-train
    --hf-checkpoint "${student_model_path}"
    --megatron-to-hf-mode bridge
    --prompt-data "${data_path}"
    --input-key prompt
    --label-key label
    --metadata-key metadata
    --apply-chat-template
    --reward-key score
    --num-rollout "${num_rollout}"
    --rollout-batch-size "${rollout_batch_size}"
    --n-samples-per-prompt "${n_samples_per_prompt}"
    --global-batch-size "${global_batch_size}"
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-2048}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-128}"
    --rollout-temperature 1.0
    --use-fault-tolerance
    --use-wandb
    --wandb-mode "${WANDB_MODE:-online}"
    --wandb-project "${WANDB_PROJECT:-relax-sdpo-opd-regression}"
    --wandb-group "${WANDB_GROUP:-${repo}-${mode}-2gpu}"
    --disable-wandb-random-suffix
)

case "${mode}" in
    grpo)
        command+=(
            --rm-type dapo
            --advantage-estimator grpo
            --eps-clip 0.2
            --eps-clip-high 0.28
        )
        ;;
    sapo)
        command+=(
            --rm-type dapo
            --advantage-estimator sapo
            --sapo-tau-pos "${SAPO_TAU_POS:-1.0}"
            --sapo-tau-neg "${SAPO_TAU_NEG:-1.05}"
            --eps-clip 0.2
            --eps-clip-high 0.28
        )
        ;;
    opd)
        command+=(
            --rm-type dapo
            --advantage-estimator grpo
            --eps-clip 0.2
            --eps-clip-high 0.28
            --use-opd
            --opd-type sglang
            --teacher-hf-checkpoint "${teacher_model_path}"
            --teacher-num-gpus-per-engine "${TEACHER_ENGINE_GPUS:-${teacher_gpus}}"
            --teacher-sglang-disable-cuda-graph
            --opd-kl-coef 1.0
            --opd-loss-coef 0.0
            --opd-kl-type reverse_kl
            --opd-token-selection student_sampled
            --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-120}"
            --opd-disable-rl-reward
            --use-rollout-logprobs
        )
        ;;
    sdpo)
        command+=(
            --group-rm
            --custom-rm-path examples.on_policy_distillation.sdpo.reward.score
            --advantage-estimator grpo
            --eps-clip 0.2
            --eps-clip-high 0.28
            --use-opd
            --opd-type sglang
            --teacher-hf-checkpoint "${teacher_model_path}"
            --teacher-num-gpus-per-engine "${TEACHER_ENGINE_GPUS:-${teacher_gpus}}"
            --teacher-sglang-disable-cuda-graph
            --opd-loss-coef "${OPD_LOSS_COEF:-1.0}"
            --opd-loss-mode sdpo
            --opd-kl-coef 0.0
            --opd-token-selection student_topk
            --opd-log-prob-top-k "${OPD_TOP_K:-100}"
            --opd-kl-type "${OPD_KL_TYPE:-jsd}"
            --opd-jsd-alpha 0.5
            --opd-norm-mode tail
            --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-120}"
            --use-rollout-logprobs
        )
        ;;
esac

if [[ "${RELAX_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
