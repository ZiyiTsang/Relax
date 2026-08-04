# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Rule-based rewards and compact feedback for the SDPO examples."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


def _metadata(sample: Any) -> dict[str, Any]:
    value = getattr(sample, "metadata", None)
    return value if isinstance(value, dict) else {}


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _extract_answer(response: str) -> str:
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.IGNORECASE | re.DOTALL)
    candidate = tagged[-1] if tagged else response
    candidate = candidate.strip()
    match = re.search(r"\b([A-D])\b", candidate, flags=re.IGNORECASE)
    return match.group(1).upper() if match else candidate


def _score_sciknoweval(sample: Any) -> dict[str, Any]:
    metadata = _metadata(sample)
    expected = metadata.get("answer_key", getattr(sample, "label", ""))
    task_type = str(metadata.get("task_type", "")).casefold()
    response = getattr(sample, "response", "")
    predicted = _extract_answer(response)

    if "true_or_false" in task_type or "true/false" in task_type:
        correct = _normalize(predicted) in {_normalize(expected), _normalize(str(expected).replace(" ", ""))}
    else:
        correct = _normalize(predicted) == _normalize(expected)

    feedback = "" if correct else "The attempted answer is incorrect. Recheck the reasoning and final answer."
    return {
        "score": 1.0 if correct else 0.0,
        "predicted": predicted,
        "format_error": int(not bool(predicted)),
        "feedback": feedback,
    }


def _extract_tool_calls(response: str) -> tuple[list[str], dict[str, Any], bool]:
    actions = [
        action.strip()
        for action in re.findall(r"^\s*Action:\s*(.+?)\s*$", response, flags=re.IGNORECASE | re.MULTILINE)
    ]
    inputs: dict[str, Any] = {}
    format_ok = bool(re.search(r"Action:.*?\nAction Input:", response, flags=re.IGNORECASE | re.DOTALL))
    decoder = json.JSONDecoder()
    for match in re.finditer(r"Action Input:\s*", response, flags=re.IGNORECASE):
        try:
            value, _ = decoder.raw_decode(response, match.end())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            inputs.update(value)
    return actions, inputs, format_ok


def _golden_tool_call(metadata: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    golden = metadata.get("golden_answer") or []
    if isinstance(golden, str):
        try:
            golden = json.loads(golden)
        except json.JSONDecodeError:
            golden = []
    actions = [str(item.get("Action", "")).strip() for item in golden if isinstance(item, dict)]
    inputs: dict[str, Any] = {}
    for item in golden:
        if not isinstance(item, dict):
            continue
        value = item.get("Action_Input", {})
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {}
        if isinstance(value, dict):
            inputs.update(value)
    return actions, inputs


def _score_toolalpaca(sample: Any) -> dict[str, Any]:
    metadata = _metadata(sample)
    predicted_actions, predicted_inputs, format_ok = _extract_tool_calls(getattr(sample, "response", ""))
    expected_actions, expected_inputs = _golden_tool_call(metadata)
    actions_correct = Counter(predicted_actions) == Counter(expected_actions)
    inputs_correct = predicted_inputs == expected_inputs
    correct = format_ok and actions_correct and inputs_correct
    feedback_parts = []
    if not actions_correct:
        feedback_parts.append("The selected tool action does not match the request.")
    if not inputs_correct:
        feedback_parts.append("The tool input arguments do not match the request.")
    if not format_ok:
        feedback_parts.append("Use the required Action and Action Input format.")
    return {
        "score": 1.0 if correct else 0.0,
        "predicted_actions": predicted_actions,
        "predicted_inputs": predicted_inputs,
        "format_error": int(not format_ok),
        "feedback": " ".join(feedback_parts),
    }


def _score_one(sample: Any) -> dict[str, Any]:
    source = str(_metadata(sample).get("data_source", "")).casefold()
    if source == "sciknoweval":
        return _score_sciknoweval(sample)
    if source in {"toolalpaca", "tooluse"}:
        return _score_toolalpaca(sample)
    raise ValueError(f"Unsupported SDPO data_source: {source!r}")


def score(_args: Any, samples: Any) -> dict[str, Any] | list[dict[str, Any]]:
    """Custom reward entry point for ``--custom-rm-path``.

    Relax calls this function with a list in ``--group-rm`` mode and with one
    ``Sample`` otherwise.  Returning only reward payloads keeps the reward
    worker process isolated; the core rollout path later uses those payloads to
    build dynamic teacher prompts.
    """

    if isinstance(samples, list):
        return [_score_one(sample) for sample in samples]
    return _score_one(samples)
