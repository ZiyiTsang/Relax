# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from relax.utils.opd.opd_opsd_worker import OpsdWorker
from relax.utils.types import Sample


def test_teacher_prefill_keeps_original_response_suffix_after_privileged_prompt() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[10, 11, 20, 21],
        response_length=2,
    )
    sample.teacher_tokens = [100, 101, 102, 20, 21]
    sample.teacher_prompt_length = 3

    worker = OpsdWorker(is_opsd=True)

    assert worker.teacher_input_ids(sample, response_length=2) == [100, 101, 102, 20, 21]
    assert worker.teacher_input_ids(sample, response_length=2)[-2:] == sample.tokens[-2:]
    assert worker.teacher_prompt_len(sample, response_length=2) == 3


def test_teacher_prefill_falls_back_to_rollout_tokens_without_dynamic_prompt() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[10, 11, 20, 21],
        response_length=2,
    )

    worker = OpsdWorker(is_opsd=False)

    assert worker.teacher_input_ids(sample, response_length=2) == sample.rollout_tokens
    assert worker.teacher_prompt_len(sample, response_length=2) == 2
