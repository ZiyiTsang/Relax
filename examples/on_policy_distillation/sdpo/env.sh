#!/usr/bin/env bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

ray stop >/dev/null 2>&1 || true

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export RELAX_VENV="/data/share/ziyi/venv/relax-sdpo"
source "${RELAX_VENV}/bin/activate"
export RELAX_PYTHON="${RELAX_VENV}/bin/python"
export MEGATRON="/data/share/ziyi/venv/relax-sdpo-megatron:/data/share/ziyi/venv/relax-sdpo/lib/python3.12/site-packages"
export PYTHONPATH="${PROJECT_ROOT}${MEGATRON:+:${MEGATRON}}"
export STUDENT_MODEL_PATH="/data/share/Qwen3-4B-Instruct-2507"
export TEACHER_MODEL_PATH="/data/share/Qwen3-4B-Instruct-2507"
export SDPO_DATA_ROOT="/data/L202500146/zengziyi_share/Data/SDPO"
