# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Feedback strategies that own every OPD-flavored algorithm difference.

``EnvironmentFeedback`` is the polymorphism point of the OPD algorithm family
(OPD / MOPD / OPSD / SDPO). The base-class defaults reproduce plain OPD, so
the shared rollout path never branches on an algorithm name: a subclass only
overrides the hooks where its behavior differs. Hooks raise to hard-fail and
return to degrade.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

from relax.utils.opd.opd_opsd_worker import OpsdWorker
from relax.utils.types import Sample


class EnvironmentFeedback(ABC):
    @staticmethod
    def record(sample: Sample, text: str | None) -> None:
        if text:
            sample.metadata.setdefault("env_feedback", []).append(str(text))

    @abstractmethod
    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def create_opsd_worker(args: Any) -> OpsdWorker:
        return OpsdWorker.from_args(args)

    @classmethod
    def validate_launch_args(cls, args: Any) -> None:
        return None

    def extra_transfer_schema(self) -> list[str]:
        return []

    def produce_extra_transfer(self, samples: list[Sample], train_data: dict) -> None:
        return None

    def check_student_topk_ids(self, sample: Sample, top_k: int) -> None:
        return None

    def check_transfer_channels(self, sample: Sample, channels: dict, top_k: int) -> None:
        return None

    @staticmethod
    def _reward_feedback(reward: Any) -> str | None:
        if isinstance(reward, dict):
            for key in ("feedback", "error", "feedback_raw"):
                value = reward.get(key)
                if value:
                    if isinstance(value, str):
                        return value if value.strip() else None
                    return str(value)
        return None

    @staticmethod
    def feedback_text(sample: Sample, reward: Any) -> str:
        reward_feedback = EnvironmentFeedback._reward_feedback(reward)
        if reward_feedback:
            return reward_feedback
        values = sample.metadata.get("env_feedback", []) if isinstance(sample.metadata, dict) else []
        return "\n".join(str(value) for value in values if value)


class OPDFeedback(EnvironmentFeedback):
    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        return

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        for sample in group:
            sample.teacher_prompt = None
            sample.opd_sample_mask = None


def load_feedback_class(path: str | None) -> type[EnvironmentFeedback]:
    if not path:
        return OPDFeedback
    module_name, separator, class_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"Invalid feedback class path: {path!r}")
    cls = getattr(importlib.import_module(module_name), class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, EnvironmentFeedback):
        raise TypeError(f"{path!r} must name an EnvironmentFeedback subclass")
    return cls


__all__ = [
    "EnvironmentFeedback",
    "OPDFeedback",
    "load_feedback_class",
]
