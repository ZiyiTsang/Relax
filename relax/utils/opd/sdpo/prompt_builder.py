# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Composable routing components for static-teacher SDPO prompts."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from relax.utils.types import Sample


@dataclass(frozen=True)
class FeedbackRecord:
    """Normalized feedback returned by a reward or environment adapter."""

    score: float | None = None
    feedback: str = ""
    raw_feedback: Any = None
    error: str | None = None
    is_success: bool = False

    @property
    def has_feedback(self) -> bool:
        return bool(self.feedback)


class TeacherPromptRenderer:
    """Render privileged solution and feedback context for the teacher only."""

    def render(
        self,
        sample: Sample,
        *,
        solution: str | None = None,
        feedback: str | None = None,
    ) -> list[dict[str, Any]]:
        messages = self._base_messages(sample)
        original_content = _as_text(messages[-1].get("content"))

        sections = [original_content]
        if solution:
            sections.append(
                "A previous successful attempt is provided below. Use it as a demonstration, "
                "but solve the original problem yourself.\n\n"
                "<successful_attempt>\n"
                f"{solution}\n"
                "</successful_attempt>"
            )
        if feedback:
            sections.append(
                "Feedback from an earlier attempt is provided below. Correct the issue it describes.\n\n"
                "<feedback>\n"
                f"{feedback}\n"
                "</feedback>"
            )
        if solution or feedback:
            sections.append("Now produce the best answer to the original problem.")

        messages[-1]["content"] = "\n\n".join(section for section in sections if section)
        return messages

    @staticmethod
    def _base_messages(sample: Sample) -> list[dict[str, Any]]:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        raw_prompt = metadata.get("sdpo_prompt")
        if raw_prompt is None:
            raw_prompt = sample.prompt

        if isinstance(raw_prompt, list):
            messages = copy.deepcopy(raw_prompt)
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                    raise TypeError(f"SDPO text prompt message {index} must contain a string role")
                if "content" in message and not isinstance(message["content"], str):
                    raise TypeError("SDPO-lite only supports text chat message content")
            if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
                return messages
            return messages + [{"role": "user", "content": ""}]

        return [{"role": "user", "content": _as_text(raw_prompt)}]


@dataclass(frozen=True)
class SdpoPromptStats:
    """Counters produced while routing a rollout group into teacher prompts."""

    total_samples: int = 0
    valid_samples: int = 0
    feedback_available: int = 0
    feedback_used: int = 0
    successful_demonstrations: int = 0
    successful_groups: int = 0
    total_groups: int = 0

    def as_dict(self) -> dict[str, float]:
        total = max(self.total_samples, 1)
        return {
            "valid_teacher_context_ratio": self.valid_samples / total,
            "feedback_available_ratio": self.feedback_available / total,
            "feedback_used_ratio": self.feedback_used / total,
            "successful_demo_ratio": self.successful_demonstrations / total,
            "successful_group_ratio": self.successful_groups / max(1, self.total_groups),
        }


class SdpoPromptBuilder:
    """Build teacher-only SDPO prompts from a rollout batch."""

    def __init__(
        self,
        *,
        feedback_provider: Callable[[Sample], FeedbackRecord] | None = None,
        prompt_renderer: TeacherPromptRenderer | None = None,
        reward_key: str | None = None,
        success_reward_threshold: float = 1.0,
        exclude_self_success: bool = True,
        feedback_only_without_solution: bool = False,
    ) -> None:
        self.feedback_provider = feedback_provider
        self.prompt_renderer = prompt_renderer or TeacherPromptRenderer()
        self.reward_key = reward_key
        self.success_reward_threshold = success_reward_threshold
        self.exclude_self_success = exclude_self_success
        self.feedback_only_without_solution = feedback_only_without_solution

    def apply(self, samples: list[Sample]) -> SdpoPromptStats:
        """Apply teacher-context routing and write SDPO fields to samples."""

        sdpo_samples = [sample for sample in samples if _is_sdpo_sample(sample)]
        if not sdpo_samples:
            return SdpoPromptStats()

        groups = self._resolve_groups(sdpo_samples)
        records = {id(sample): self._get_feedback(sample) for sample in sdpo_samples}
        valid_samples = 0
        feedback_available = 0
        feedback_used = 0
        successful_demonstrations = 0
        successful_groups = 0

        for group in groups.values():
            successful = [
                sample
                for sample in group
                if records[id(sample)].is_success and bool(_as_text(sample.response))
            ]
            if successful:
                successful_groups += 1

            for sample in group:
                record = records[id(sample)]
                if record.has_feedback:
                    feedback_available += 1

                solution = self._select_successful_response(successful, sample)
                use_feedback = record.has_feedback and (
                    not self.feedback_only_without_solution or not solution
                )
                if use_feedback:
                    feedback_used += 1

                has_teacher_context = bool(solution or use_feedback)
                sample.sdpo_valid = has_teacher_context
                sample.teacher_prompt = self.prompt_renderer.render(
                    sample,
                    solution=solution,
                    feedback=record.feedback if use_feedback else "",
                )
                sample.teacher_tokens = None
                sample.teacher_prompt_length = None

                if has_teacher_context:
                    valid_samples += 1
                if solution:
                    successful_demonstrations += 1

        return SdpoPromptStats(
            total_samples=len(sdpo_samples),
            valid_samples=valid_samples,
            feedback_available=feedback_available,
            feedback_used=feedback_used,
            successful_demonstrations=successful_demonstrations,
            successful_groups=successful_groups,
            total_groups=len(groups),
        )

    def _get_feedback(self, sample: Sample) -> FeedbackRecord:
        if self.feedback_provider is not None:
            return self.feedback_provider(sample)

        reward = sample.reward
        reward_dict = reward if isinstance(reward, dict) else None
        score_value: Any = reward
        if reward_dict is not None:
            score_value = reward_dict.get(self.reward_key or "score")
            if score_value is None:
                score_value = reward_dict.get("reward")

        score = _to_float(score_value)
        feedback = ""
        raw_feedback = None
        error = None
        if reward_dict is not None:
            raw_feedback = reward_dict.get("feedback_raw")
            error_text = _as_text(reward_dict.get("error"))
            error = error_text or None
            for key in ("feedback", "feedback_raw", "error"):
                feedback = _as_text(reward_dict.get(key))
                if feedback:
                    break

        if not feedback:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            feedback = _as_text(metadata.get("feedback"))

        return FeedbackRecord(
            score=score,
            feedback=feedback,
            raw_feedback=raw_feedback,
            error=error,
            is_success=(
                score is not None
                and score >= self.success_reward_threshold
                and bool(_as_text(sample.response))
            ),
        )

    @staticmethod
    def _resolve_groups(samples: Iterable[Sample]) -> dict[Hashable, list[Sample]]:
        groups: dict[Hashable, list[Sample]] = defaultdict(list)
        for position, sample in enumerate(samples):
            # A missing group id must never allow unrelated prompts to share a
            # successful response. Such a sample therefore forms a singleton.
            key = (
                sample.group_index
                if sample.group_index is not None
                else ("sdpo-singleton", position)
            )
            groups[key].append(sample)
        return dict(groups)

    def _select_successful_response(self, candidates: list[Sample], current: Sample) -> str | None:
        for candidate in candidates:
            if self.exclude_self_success and candidate is current:
                continue
            response = _as_text(candidate.response)
            if response:
                return response
        return None


def prepare_sdpo_teacher_prompts(
    samples: list[Sample],
    *,
    reward_key: str | None = None,
    success_reward_threshold: float = 1.0,
    exclude_self_success: bool = True,
    feedback_only_without_solution: bool = False,
) -> SdpoPromptStats:
    """Build SDPO teacher prompts with the default reward feedback provider."""

    builder = SdpoPromptBuilder(
        reward_key=reward_key,
        success_reward_threshold=success_reward_threshold,
        exclude_self_success=exclude_self_success,
        feedback_only_without_solution=feedback_only_without_solution,
    )
    return builder.apply(samples)


def _is_sdpo_sample(sample: Sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return bool(metadata.get("sdpo", False))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "FeedbackRecord",
    "SdpoPromptBuilder",
    "SdpoPromptStats",
    "TeacherPromptRenderer",
    "prepare_sdpo_teacher_prompts",
]
