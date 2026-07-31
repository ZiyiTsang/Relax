# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reusable row normalizers for the minimal SDPO data pipeline.

The module separates dataset selection from row-format selection.  Dataset
normalizers are registered in :class:`DatasetNormalizerRegistry`; a dataset
normalizer may in turn delegate to format-specific strategies.  This keeps
the command-line entry point independent of individual data schemas.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable


TARGET_DOMAINS = frozenset({"Chemistry", "Physics", "Biology", "Material", "Materials"})


def canonical_domain(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized == "material":
        return "Materials"
    return normalized.capitalize()


def json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class NormalizationContext:
    source_split: str
    domain: str | None = None


class RowNormalizationStrategy(ABC):
    """Strategy for one concrete source-row schema."""

    @abstractmethod
    def supports(self, row: dict[str, Any]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any] | None:
        raise NotImplementedError


class DatasetNormalizer(ABC):
    """Normalizer for one logical dataset."""

    @abstractmethod
    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any] | None:
        raise NotImplementedError


class PublicSciKnowEvalStrategy(RowNormalizationStrategy):
    """Normalize the nested public SciKnowEval benchmark schema."""

    def supports(self, row: dict[str, Any]) -> bool:
        return isinstance(row.get("details"), dict) and (
            isinstance(row.get("prompt"), dict) or "question" in row
        )

    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any] | None:
        details = row.get("details") or {}
        if str(details.get("level", "")).upper() != "L3":
            return None

        domain = str(row.get("domain", ""))
        if domain not in TARGET_DOMAINS:
            return None

        choices = row.get("choices") or {}
        choice_lines = [
            f"{label}: {text}"
            for label, text in zip(choices.get("label") or [], choices.get("text") or [], strict=False)
        ]
        prompt_value = row.get("prompt", {})
        prompt_default = prompt_value.get("default", "") if isinstance(prompt_value, dict) else prompt_value
        question = str(row.get("question") or prompt_default).strip()
        prompt = question
        if choice_lines:
            prompt = f"{question}\n\n" + "\n".join(choice_lines)
        prompt += "\n\nReason carefully and provide the final answer."

        normalized_domain = "Materials" if domain == "Material" else domain
        answer = row.get("answerKey", row.get("answer", ""))
        metadata = {
            "sdpo": True,
            "data_source": "sciknoweval",
            "source_split": context.source_split,
            "domain": normalized_domain,
            "task_type": str(row.get("type", "unknown")),
            "answer_key": answer,
            "choices": choices,
            "sdpo_prompt": prompt,
        }
        return {"prompt": prompt, "label": json_text(answer), "metadata": metadata}


class ReferenceSciKnowEvalStrategy(RowNormalizationStrategy):
    """Normalize the flat prompt schema shipped with the SDPO reference code."""

    def supports(self, row: dict[str, Any]) -> bool:
        return row.get("dataset") == "sciknoweval" and isinstance(row.get("prompt"), str) and "answer" in row

    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any] | None:
        domain = canonical_domain(context.domain)
        if domain not in {"Chemistry", "Physics", "Biology", "Materials"}:
            return None

        prompt = str(row["prompt"]).strip()
        system = str(row.get("system") or "").strip()
        if system:
            prompt = f"{system}\n\n{prompt}"
        answer = row.get("answer", "")
        metadata = {
            "sdpo": True,
            "data_source": "sciknoweval",
            "source_split": context.source_split,
            "domain": domain,
            "task_type": str(row.get("kind", "mcq")),
            "answer_key": answer,
            "sdpo_prompt": prompt,
            "source_index": row.get("idx"),
        }
        return {"prompt": prompt, "label": json_text(answer), "metadata": metadata}


class SciKnowEvalNormalizer(DatasetNormalizer):
    """Facade selecting a SciKnowEval source-format strategy."""

    def __init__(self, strategies: Iterable[RowNormalizationStrategy] | None = None) -> None:
        self._strategies = tuple(
            strategies
            or (
                ReferenceSciKnowEvalStrategy(),
                PublicSciKnowEvalStrategy(),
            )
        )

    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any] | None:
        for strategy in self._strategies:
            if strategy.supports(row):
                return strategy.normalize(row, context)
        return None


class ToolAlpacaStrategy(RowNormalizationStrategy):
    """Normalize the public Ahren09/ToolAlpaca row schema."""

    def supports(self, row: dict[str, Any]) -> bool:
        return "golden_answer" in row and any(key in row for key in ("name", "nl_documentation", "instruction"))

    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any]:
        name = str(row.get("name", "")).strip()
        description = str(row.get("description", "")).strip()
        documentation = str(row.get("nl_documentation", "")).strip()
        instruction = str(row.get("instruction", row.get("prompt", ""))).strip()
        prompt = (
            "You are given an API specification and a user request. Select the correct tool and "
            "emit the tool call using exactly:\n"
            "Action: <tool name>\nAction Input: <JSON object>\n\n"
            f"Tool name: {name}\n"
            f"Tool description: {description}\n"
            f"Tool documentation:\n{documentation}\n\n"
            f"User request:\n{instruction}"
        )
        golden_answer = row.get("golden_answer") or []
        metadata = {
            "sdpo": True,
            "data_source": "toolalpaca",
            "source_split": context.source_split,
            "task_type": "tool_call",
            "golden_answer": golden_answer,
            "sdpo_prompt": prompt,
        }
        return {"prompt": prompt, "label": json_text(golden_answer), "metadata": metadata}


class ReferenceToolUseStrategy(RowNormalizationStrategy):
    """Normalize the flat tooluse schema shipped with the SDPO reference code."""

    def supports(self, row: dict[str, Any]) -> bool:
        return row.get("dataset") == "tooluse" and isinstance(row.get("prompt"), str) and "answer" in row

    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any]:
        answer = row.get("answer", "")
        try:
            golden_answer = json.loads(answer) if isinstance(answer, str) else answer
        except json.JSONDecodeError:
            golden_answer = answer
        prompt = str(row.get("prompt", "")).strip()
        metadata = {
            "sdpo": True,
            "data_source": "tooluse",
            "source_split": context.source_split,
            "task_type": str(row.get("kind", "tooluse")),
            "golden_answer": golden_answer,
            "sdpo_prompt": prompt,
            "source_index": row.get("idx"),
        }
        return {"prompt": prompt, "label": json_text(answer), "metadata": metadata}


class ToolAlpacaNormalizer(DatasetNormalizer):
    """Facade for public ToolAlpaca and reference SDPO tooluse rows."""

    def __init__(self, strategies: Iterable[RowNormalizationStrategy] | None = None) -> None:
        self._strategies = tuple(strategies or (ReferenceToolUseStrategy(), ToolAlpacaStrategy()))

    def normalize(self, row: dict[str, Any], context: NormalizationContext) -> dict[str, Any] | None:
        for strategy in self._strategies:
            if strategy.supports(row):
                return strategy.normalize(row, context)
        return None


class DatasetNormalizerRegistry:
    """Registry/factory for logical dataset normalizers."""

    def __init__(self, normalizers: dict[str, DatasetNormalizer] | None = None) -> None:
        self._normalizers = dict(
            normalizers
            or {
                "sciknoweval": SciKnowEvalNormalizer(),
                "toolalpaca": ToolAlpacaNormalizer(),
                "tooluse": ToolAlpacaNormalizer(),
            }
        )

    def register(self, name: str, normalizer: DatasetNormalizer) -> None:
        self._normalizers[name] = normalizer

    def resolve(self, name: str) -> DatasetNormalizer:
        try:
            return self._normalizers[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported SDPO dataset {name!r}; available={sorted(self._normalizers)}") from exc

    def normalize_rows(
        self,
        name: str,
        rows: Iterable[dict[str, Any]],
        *,
        source_split: str,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        normalizer = self.resolve(name)
        context = NormalizationContext(source_split=source_split, domain=domain)
        return [
            normalized
            for row in rows
            if (normalized := normalizer.normalize(row, context)) is not None
        ]


DEFAULT_NORMALIZER_REGISTRY = DatasetNormalizerRegistry()


def normalize_rows(
    dataset: str,
    rows: Iterable[dict[str, Any]],
    *,
    source_split: str,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Small facade retained for the CLI and focused unit tests."""

    return DEFAULT_NORMALIZER_REGISTRY.normalize_rows(
        dataset,
        rows,
        source_split=source_split,
        domain=domain,
    )


__all__ = [
    "DEFAULT_NORMALIZER_REGISTRY",
    "DatasetNormalizer",
    "DatasetNormalizerRegistry",
    "NormalizationContext",
    "PublicSciKnowEvalStrategy",
    "ReferenceToolUseStrategy",
    "ReferenceSciKnowEvalStrategy",
    "RowNormalizationStrategy",
    "SciKnowEvalNormalizer",
    "ToolAlpacaStrategy",
    "ToolAlpacaNormalizer",
    "canonical_domain",
    "normalize_rows",
]
