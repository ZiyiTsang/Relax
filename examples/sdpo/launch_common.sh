# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
set -a
[ -f .env ] && source .env
set +a

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source scripts/entrypoint/local.sh
fi

launch_sdpo_example() {
    local mode="$1"
    local now
    now="$(date "+%Y-%m-%d-%H:%M:%S")"

    export RELAX_OPD_PER_POS_TOKEN_IDS=1
    export RELAX_PROPAGATE_ENV_VARS="${RELAX_PROPAGATE_ENV_VARS:+${RELAX_PROPAGATE_ENV_VARS},}RELAX_OPD_PER_POS_TOKEN_IDS"

    local student_model_path="${STUDENT_MODEL_PATH:?Set STUDENT_MODEL_PATH to the student checkpoint}"
    local teacher_model_path="${TEACHER_MODEL_PATH:-${student_model_path}}"
    local data_path="${DATA_PATH:?Set DATA_PATH to a prepared SciKnowEval, tooluse, or ToolAlpaca JSONL file}"
    local actor_gpus="${ACTOR_GPUS:-4}"
    local rollout_gpus="${ROLLOUT_GPUS:-2}"
    local teacher_gpus="${TEACHER_GPUS:-2}"
    local tp_size="${TP_SIZE:-2}"
    local cp_size="${CP_SIZE:-1}"
    local experiment_name="${EXPERIMENT_NAME:-sdpo-${mode}-${now}}"

    local model_args=(
        --swiglu
        --num-layers 36
        --hidden-size 2560
        --ffn-hidden-size 9728
        --num-attention-heads 32
        --group-query-attention
        --num-query-groups 8
        --use-rotary-position-embeddings
        --disable-bias-linear
        --normalization RMSNorm
        --norm-epsilon 1e-6
        --rotary-base 5000000
        --vocab-size 151936
        --kv-channels 128
        --qk-layernorm
    )
    local checkpoint_args=(
        --hf-checkpoint "${student_model_path}"
        --megatron-to-hf-mode bridge
    )
    local rollout_args=(
        --prompt-data "${data_path}"
        --input-key prompt
        --label-key label
        --metadata-key metadata
        --apply-chat-template
        --group-rm
        --custom-rm-path examples.sdpo.reward.score
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
    local opd_args=()
    local resource_json="{\"actor\": [1, ${actor_gpus}], \"rollout\": [1, ${rollout_gpus}]}"
    if [ "${mode}" = sdpo ]; then
        resource_json="{\"actor\": [1, ${actor_gpus}], \"rollout\": [1, ${rollout_gpus}], \"teacher\": [1, ${teacher_gpus}]}"
        opd_args=(
            --use-opd
            --opd-type sglang
            --teacher-hf-checkpoint "${teacher_model_path}"
            --teacher-num-gpus-per-engine "${teacher_gpus}"
            --opd-teacher-prompt-key prompt
            --opd-loss-coef "${OPD_LOSS_COEF:-1.0}"
            --opd-kl-coef 0.0
            --opd-token-selection student_topk
            --opd-log-prob-top-k "${OPD_TOP_K:-100}"
            --opd-kl-type "${OPD_KL_TYPE:-jsd}"
            --opd-jsd-alpha 0.5
            --opd-norm-mode tail
            --opd-teacher-timeout-s "${OPD_TEACHER_TIMEOUT_S:-120}"
            --use-rollout-logprobs
        )
    fi

    python3 -m relax.entrypoints.train \
        --resource "${resource_json}" \
        --max-staleness 0 \
        --num-data-storage-units 1 \
        --colocate \
        --offload \
        --use-health-check \
        --tensor-model-parallel-size "${tp_size}" \
        --context-parallel-size "${cp_size}" \
        --pipeline-model-parallel-size 1 \
        --sequence-parallel \
        --calculate-per-token-loss \
        --advantage-estimator grpo \
        --eps-clip 0.2 \
        --eps-clip-high 0.28 \
        --optimizer adam \
        --lr "${LEARNING_RATE:-1e-6}" \
        --lr-decay-style constant \
        --weight-decay 0.01 \
        --clip-grad 1.0 \
        --use-dynamic-batch-size \
        --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}" \
        --rollout-num-gpus "${rollout_gpus}" \
        --rollout-num-gpus-per-engine "${ROLLOUT_ENGINE_GPUS:-${rollout_gpus}}" \
        --sglang-load-format dummy \
        --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION:-0.8}" \
        --sglang-disable-cuda-graph \
        --tb-experiment-name "${experiment_name}" \
        --skip-eval-before-train \
        "${model_args[@]}" \
        "${checkpoint_args[@]}" \
        "${rollout_args[@]}" \
        "${opd_args[@]}"
}
