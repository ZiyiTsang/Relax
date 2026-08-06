# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Small, explicit feedback implementations used by OPD/OPSD/SDPO."""

from __future__ import annotations

import copy
import importlib
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from relax.utils.types import Sample


class EnvironmentFeedback(ABC):
    @staticmethod
    def record(sample: Sample, text: str | None) -> None:
        if text:
            sample.metadata.setdefault("env_feedback", []).append(str(text))

    def record_reward(self, sample: Sample, reward: Any) -> None:
        if isinstance(reward, dict):
            for key in ("feedback", "error", "feedback_raw"):
                value = reward.get(key)
                if value:
                    self.record(sample, value if isinstance(value, str) else str(value))
                    return

    @abstractmethod
    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        raise NotImplementedError

    @staticmethod
    def feedback_text(sample: Sample, reward: Any) -> str:
        if isinstance(reward, dict):
            for key in ("feedback", "error", "feedback_raw"):
                value = reward.get(key)
                if value:
                    return value if isinstance(value, str) else str(value)
        values = sample.metadata.get("env_feedback", []) if isinstance(sample.metadata, dict) else []
        return "\n".join(str(value) for value in values if value)


class OPDFeedback(EnvironmentFeedback):
    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        for sample in group:
            sample.teacher_prompt = None


class OPSDFeedback(EnvironmentFeedback):
    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        # The dataset-provided teacher_prompt is authoritative.
        return


class SDPOFeedback(EnvironmentFeedback):
    def prepare_teacher_prompts(self, group: list[Sample], rewards: list[Any]) -> None:
        if len(group) != len(rewards):
            raise ValueError(f"feedback requires one reward per sample: {len(group)} != {len(rewards)}")
        by_group: dict[Any, list[Sample]] = defaultdict(list)
        for position, sample in enumerate(group):
            key = sample.group_index if sample.group_index is not None else ("singleton", position)
            by_group[key].append(sample)
        reward_by_id = {id(sample): reward for sample, reward in zip(group, rewards)}
        successful: dict[Any, list[Sample]] = {}
        for key, samples in by_group.items():
            successful[key] = [sample for sample in samples if self._success(reward_by_id[id(sample)], sample)]
        for key, samples in by_group.items():
            for sample in samples:
                reward = reward_by_id[id(sample)]
                solution = next(
                    (
                        candidate.response
                        for candidate in successful[key]
                        if candidate is not sample and candidate.response
                    ),
                    None,
                )
                feedback = self.feedback_text(sample, reward)
                sample.teacher_prompt = self._render(sample, solution, feedback) if solution or feedback else None
                sample.teacher_tokens = None
                sample.teacher_prompt_length = None

    @staticmethod
    def _success(reward: Any, sample: Sample) -> bool:
        value = reward.get("score", reward.get("reward")) if isinstance(reward, dict) else reward
        try:
            return float(value) >= 1.0 and bool(sample.response)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _render(sample: Sample, solution: str | None, feedback: str) -> str | list[dict[str, str]]:
        prompt = copy.deepcopy(sample.prompt)
        addition = []
        if solution:
            addition.append(f"<successful_attempt>\n{solution}\n</successful_attempt>")
        if feedback:
            addition.append(f"<feedback>\n{feedback}\n</feedback>")
        suffix = "\n\n".join(addition + (["Now produce the best answer to the original problem."] if addition else []))
        if isinstance(prompt, list):
            messages = prompt
            if not messages or messages[-1].get("role") != "user":
                messages.append({"role": "user", "content": ""})
            messages[-1]["content"] = f"{messages[-1].get('content', '')}\n\n{suffix}"
            return messages
        return f"{prompt}\n\n{suffix}"


class SciKnowEvalSDPOFeedback(SDPOFeedback):
    pass


class ToolUseSDPOFeedback(SDPOFeedback):
    pass


class MathSDPOFeedback(SDPOFeedback):
    pass


class CodeSDPOFeedback(SDPOFeedback):
    pass


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
    "SDPOFeedback",
    "SciKnowEvalSDPOFeedback",
    "ToolUseSDPOFeedback",
    "MathSDPOFeedback",
    "CodeSDPOFeedback",
    "load_feedback_class",
]
