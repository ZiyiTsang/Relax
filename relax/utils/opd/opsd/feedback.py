# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OPSD feedback: the dataset-provided teacher prompt is authoritative."""

from __future__ import annotations

from typing import Any

from relax.utils.opd.feedback import EnvironmentFeedback
from relax.utils.types import Sample


class OPSDFeedback(EnvironmentFeedback):
    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        return

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        for sample in group:
            sample.opd_sample_mask = None


__all__ = [
    "OPSDFeedback",
]
