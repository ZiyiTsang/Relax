# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""OPSD feedback: the dataset-provided teacher prompt is the privilege.

The raw dataset column is rendered at ingestion and surfaced as
``metadata["opd_teacher_prompt"]``; this class owns the policy of assigning it
to ``sample.teacher_prompt``. Samples without the field fall back to the
student prompt via ``OpsdWorker``, matching plain OPD.
"""

from __future__ import annotations

from typing import Any

from relax.utils.opd.feedback import EnvironmentFeedback
from relax.utils.types import Sample


class OPSDFeedback(EnvironmentFeedback):
    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        for sample in group:
            sample.opd_sample_mask = None
            privileged = sample.metadata.get("opd_teacher_prompt") if isinstance(sample.metadata, dict) else None
            if privileged is not None:
                sample.teacher_prompt = privileged


__all__ = [
    "OPSDFeedback",
]
