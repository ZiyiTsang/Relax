# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small, explicit feedback implementations used by OPD/OPSD/SDPO."""

from __future__ import annotations

import copy
import importlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, ClassVar

from relax.utils.types import Sample


class EnvironmentFeedback(ABC):
    is_sdpo_feedback: ClassVar[bool] = False

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


class OPSDFeedback(EnvironmentFeedback):
    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        return

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        # The dataset-provided teacher_prompt is authoritative.
        for sample in group:
            sample.opd_sample_mask = None


def _record_sdpo_sample_feedback(sample: Sample, reward: Any) -> None:
    feedback = EnvironmentFeedback._reward_feedback(reward)
    if feedback:
        EnvironmentFeedback.record(sample, feedback)


def _render_sdpo_teacher_prompt(sample: Sample, additions: list[str]) -> str | list[dict[str, str]]:
    prompt = copy.deepcopy(sample.prompt)
    suffix = "\n\n".join(additions + (["Now produce the best answer to the original problem."] if additions else []))
    if isinstance(prompt, list):
        messages = prompt
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": ""})
        messages[-1]["content"] = f"{messages[-1].get('content', '')}\n\n{suffix}"
        return messages
    return f"{prompt}\n\n{suffix}"


def _set_sdpo_teacher_prompt(sample: Sample, additions: list[str]) -> None:
    sample.teacher_prompt = (
        _render_sdpo_teacher_prompt(sample, additions) if additions else copy.deepcopy(sample.prompt)
    )
    sample.opd_sample_mask = bool(additions)
    sample.teacher_tokens = None
    sample.teacher_prompt_length = None


def _is_successful_reward(reward: Any) -> bool:
    value = reward.get("score", reward.get("reward")) if isinstance(reward, dict) else reward
    try:
        return float(value) >= 1.0
    except (TypeError, ValueError):
        return False


def _sdpo_group_key(sample: Sample, position: int) -> Any:
    if sample.group_index is not None:
        return ("group", sample.group_index)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    uid = metadata.get("uid")
    return ("uid", uid) if uid is not None else ("singleton", position)


def _prepare_sdpo_teacher_prompts(group: list[Sample], rewards: list[Any]) -> None:
    if len(group) != len(rewards):
        raise ValueError(f"feedback requires one reward per sample: {len(group)} != {len(rewards)}")
    by_group: dict[Any, list[Sample]] = defaultdict(list)
    for position, sample in enumerate(group):
        by_group[_sdpo_group_key(sample, position)].append(sample)
    reward_by_id = {id(sample): reward for sample, reward in zip(group, rewards, strict=True)}
    successful = {
        key: [sample for sample in samples if _is_successful_reward(reward_by_id[id(sample)])]
        for key, samples in by_group.items()
    }
    for key, samples in by_group.items():
        for sample in samples:
            peer = next((candidate for candidate in successful[key] if candidate is not sample), None)
            source = peer or next((candidate for candidate in successful[key] if candidate is sample), None)
            additions = []
            if source is not None:
                additions.append(f"<successful_attempt>\n{source.response}\n</successful_attempt>")
            feedback = EnvironmentFeedback.feedback_text(sample, reward_by_id[id(sample)])
            if feedback:
                additions.append(f"<feedback>\n{feedback}\n</feedback>")
            _set_sdpo_teacher_prompt(sample, additions)


class SciKnowEvalSDPOFeedback(EnvironmentFeedback):
    is_sdpo_feedback: ClassVar[bool] = True

    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        _record_sdpo_sample_feedback(sample, reward)

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        _prepare_sdpo_teacher_prompts(group, rewards)


class ToolUseSDPOFeedback(EnvironmentFeedback):
    is_sdpo_feedback: ClassVar[bool] = True

    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        _record_sdpo_sample_feedback(sample, reward)

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        _prepare_sdpo_teacher_prompts(group, rewards)


class CodeSDPOFeedback(EnvironmentFeedback):
    is_sdpo_feedback: ClassVar[bool] = True

    def record_sample_feedback(self, sample: Sample, reward: Any) -> None:
        _record_sdpo_sample_feedback(sample, reward)

    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        _prepare_sdpo_teacher_prompts(group, rewards)


def load_feedback_class(path: str | None) -> type[EnvironmentFeedback]:
    if not path:
        raise ValueError("OPD feedback class is required; no default implementation exists")
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
    "OPSDFeedback",
    "SciKnowEvalSDPOFeedback",
    "ToolUseSDPOFeedback",
    "CodeSDPOFeedback",
    "load_feedback_class",
]
