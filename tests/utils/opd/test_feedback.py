# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TDD coverage for the explicit OPD/OPSD/SDPO feedback interface."""

from __future__ import annotations

import pytest

from relax.utils.types import Sample


def _sample(group: int, index: int, response: str, reward: object, **metadata: object) -> Sample:
    return Sample(
        group_index=group,
        index=index,
        prompt=f"question-{group}",
        response=response,
        response_length=len(response),
        reward=reward,
        metadata=dict(metadata),
    )


def test_record_appends_text_only_to_originating_sample() -> None:
    from relax.utils.opd.feedback import EnvironmentFeedback

    sample = _sample(1, 0, "bad", 0.0)
    other = _sample(1, 1, "good", 1.0)
    EnvironmentFeedback.record(sample, "failed test case")
    EnvironmentFeedback.record(sample, "retry with a shorter answer")

    assert sample.metadata["env_feedback"] == ["failed test case", "retry with a shorter answer"]
    assert "env_feedback" not in other.metadata


def test_sdpo_solution_is_shared_only_inside_group_and_feedback_stays_local() -> None:
    from relax.utils.opd.feedback import SDPOFeedback

    target = _sample(7, 0, "wrong", {"score": 0.0}, env_feedback=["fix arithmetic"])
    success = _sample(7, 1, "correct solution", {"score": 1.0}, env_feedback=["success details"])
    unrelated = _sample(8, 2, "other", {"score": 0.0}, env_feedback=["unrelated feedback"])

    feedback = SDPOFeedback()
    feedback.prepare_teacher_prompts([target, success, unrelated], [target.reward, success.reward, unrelated.reward])

    assert target.teacher_prompt is not None
    target_text = (
        target.teacher_prompt[-1]["content"] if isinstance(target.teacher_prompt, list) else target.teacher_prompt
    )
    assert "correct solution" in target_text
    assert "fix arithmetic" in target_text
    assert "success details" not in target_text
    unrelated_text = (
        unrelated.teacher_prompt[-1]["content"]
        if isinstance(unrelated.teacher_prompt, list)
        else unrelated.teacher_prompt
    )
    assert "correct solution" not in unrelated_text
    assert "unrelated feedback" in unrelated_text


def test_sdpo_falls_back_to_original_prompt_without_solution_or_feedback() -> None:
    from relax.utils.opd.feedback import SDPOFeedback

    sample = _sample(3, 0, "an answer", {"score": 0.0})
    SDPOFeedback().prepare_teacher_prompts([sample], [sample.reward])

    assert sample.teacher_prompt is None or sample.teacher_prompt == sample.prompt


def test_math_sdpo_uses_same_group_successful_rollout() -> None:
    from relax.utils.opd.feedback import MathSDPOFeedback

    failed = _sample(4, 0, "wrong", {"score": 0.0})
    solved = _sample(4, 1, "worked solution", {"score": 1.0})
    MathSDPOFeedback().prepare_teacher_prompts([failed, solved], [failed.reward, solved.reward])

    assert "worked solution" in str(failed.teacher_prompt)


@pytest.mark.parametrize(
    "name",
    [
        "EnvironmentFeedback",
        "OPDFeedback",
        "OPSDFeedback",
        "SDPOFeedback",
        "SciKnowEvalSDPOFeedback",
        "ToolUseSDPOFeedback",
        "MathSDPOFeedback",
        "CodeSDPOFeedback",
    ],
)
def test_all_feedback_classes_are_defined_in_one_public_module(name: str) -> None:
    import relax.utils.opd.feedback as feedback

    cls = getattr(feedback, name)
    assert cls.__module__ == feedback.__name__


def test_feedback_class_is_required_by_opd_validation() -> None:
    from argparse import Namespace

    from relax.utils.opd.opd_utils import validate_opd_args

    args = Namespace(use_opd=True, opd_type="sglang", opd_feedback_class=None)
    with pytest.raises(ValueError, match="feedback class"):
        validate_opd_args(args, is_sft=False)
