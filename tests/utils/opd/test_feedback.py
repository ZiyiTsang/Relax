# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""TDD coverage for the explicit OPD/OPSD/SDPO feedback interface."""

from __future__ import annotations

import pytest

from relax.utils.types import Sample


def _sample(group: int | None, index: int, response: str, reward: object, **metadata: object) -> Sample:
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


def test_record_sample_feedback_uses_feedback_specific_sample_stage() -> None:
    from relax.utils.opd.feedback import OPDFeedback, OPSDFeedback, ToolUseSDPOFeedback

    opd_sample = _sample(1, 0, "answer", {"score": 0.0, "feedback": "unused"})
    opsd_sample = _sample(1, 1, "answer", {"score": 0.0, "feedback": "unused"})
    sdpo_sample = _sample(1, 2, "answer", {"score": 0.0, "feedback": "fix the second step"})

    OPDFeedback().record_sample_feedback(opd_sample, opd_sample.reward)
    OPSDFeedback().record_sample_feedback(opsd_sample, opsd_sample.reward)
    ToolUseSDPOFeedback().record_sample_feedback(sdpo_sample, sdpo_sample.reward)

    assert "env_feedback" not in opd_sample.metadata
    assert "env_feedback" not in opsd_sample.metadata
    assert sdpo_sample.metadata["env_feedback"] == ["fix the second step"]


@pytest.mark.parametrize(
    "feedback_class_name",
    ["SciKnowEvalSDPOFeedback", "ToolUseSDPOFeedback", "CodeSDPOFeedback"],
)
def test_sdpo_solution_is_shared_only_inside_group_and_feedback_stays_local(feedback_class_name: str) -> None:
    import relax.utils.opd.feedback as feedback_module

    target = _sample(7, 0, "wrong", {"score": 0.0}, env_feedback=["fix arithmetic"])
    success = _sample(7, 1, "correct solution", {"score": 1.0}, env_feedback=["success details"])
    unrelated = _sample(8, 2, "other", {"score": 0.0}, env_feedback=["unrelated feedback"])

    feedback = getattr(feedback_module, feedback_class_name)()
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


@pytest.mark.parametrize(
    "feedback_class_name",
    ["SciKnowEvalSDPOFeedback", "ToolUseSDPOFeedback", "CodeSDPOFeedback"],
)
def test_sdpo_prefers_peer_solution_and_falls_back_to_self_success(feedback_class_name: str) -> None:
    import relax.utils.opd.feedback as feedback_module

    self_success = _sample(9, 0, "self solution", {"score": 1.0})
    peer_success = _sample(9, 1, "peer solution", {"score": 1.0})

    singleton_success = _sample(10, 0, "only solution", {"score": 1.0})
    getattr(feedback_module, feedback_class_name)().prepare_teacher_prompts(
        [self_success, peer_success, singleton_success],
        [self_success.reward, peer_success.reward, singleton_success.reward],
    )

    assert "peer solution" in str(self_success.teacher_prompt)
    assert "self solution" not in str(self_success.teacher_prompt)
    assert "self solution" in str(peer_success.teacher_prompt)
    assert "peer solution" not in str(peer_success.teacher_prompt)
    assert "only solution" in str(singleton_success.teacher_prompt)
    assert singleton_success.opd_sample_mask is True


def test_sdpo_falls_back_to_original_prompt_without_solution_or_feedback() -> None:
    from relax.utils.opd.feedback import ToolUseSDPOFeedback

    sample = _sample(3, 0, "an answer", {"score": 0.0})
    ToolUseSDPOFeedback().prepare_teacher_prompts([sample], [sample.reward])

    assert sample.teacher_prompt == sample.prompt
    assert sample.opd_sample_mask is False


def test_sdpo_fallback_copies_original_message_prompt() -> None:
    from relax.utils.opd.feedback import ToolUseSDPOFeedback

    prompt = [{"role": "user", "content": "question"}]
    sample = Sample(
        group_index=3,
        prompt=prompt,
        response="answer",
        response_length=1,
        reward={"score": 0.0},
    )

    ToolUseSDPOFeedback().prepare_teacher_prompts([sample], [sample.reward])

    assert sample.teacher_prompt == prompt
    assert sample.teacher_prompt is not prompt
    assert sample.opd_sample_mask is False


@pytest.mark.parametrize(
    "feedback_class_name",
    ["SciKnowEvalSDPOFeedback", "ToolUseSDPOFeedback", "CodeSDPOFeedback"],
)
def test_sdpo_uses_same_group_successful_rollout(feedback_class_name: str) -> None:
    import relax.utils.opd.feedback as feedback_module

    failed = _sample(4, 0, "wrong", {"score": 0.0})
    solved = _sample(4, 1, "worked solution", {"score": 1.0})
    getattr(feedback_module, feedback_class_name)().prepare_teacher_prompts(
        [failed, solved], [failed.reward, solved.reward]
    )

    assert "worked solution" in str(failed.teacher_prompt)
    assert failed.opd_sample_mask is True


@pytest.mark.parametrize("feedback_class_name", ["ToolUseSDPOFeedback", "CodeSDPOFeedback"])
def test_tool_and_code_use_uid_when_group_index_is_missing(feedback_class_name: str) -> None:
    import relax.utils.opd.feedback as feedback_module

    failed = _sample(None, 0, "failed attempt", {"score": 0.0}, uid="uid-a")
    solved = _sample(None, 1, "same uid solution", {"score": 1.0}, uid="uid-a")
    unrelated = _sample(None, 2, "other uid solution", {"score": 1.0}, uid="uid-b")

    getattr(feedback_module, feedback_class_name)().prepare_teacher_prompts(
        [failed, solved, unrelated], [failed.reward, solved.reward, unrelated.reward]
    )

    assert "same uid solution" in str(failed.teacher_prompt)
    assert "other uid solution" not in str(failed.teacher_prompt)
    assert failed.opd_sample_mask is True


def test_ordinary_opd_feedback_does_not_create_a_sample_mask() -> None:
    from relax.utils.opd.feedback import OPDFeedback, OPSDFeedback

    opd_sample = _sample(6, 0, "answer", 1.0)
    opsd_sample = _sample(6, 1, "answer", 1.0)
    opsd_sample.teacher_prompt = "dataset teacher prompt"

    OPDFeedback().prepare_teacher_prompts([opd_sample], [opd_sample.reward])
    OPSDFeedback().prepare_teacher_prompts([opsd_sample], [opsd_sample.reward])

    assert opd_sample.opd_sample_mask is None
    assert opsd_sample.opd_sample_mask is None
    assert opsd_sample.teacher_prompt == "dataset teacher prompt"


@pytest.mark.parametrize("feedback_class_name", ["ToolUseSDPOFeedback", "CodeSDPOFeedback"])
def test_tool_and_code_feedback_share_peer_solutions(feedback_class_name: str) -> None:
    import relax.utils.opd.feedback as feedback_module

    failed = _sample(11, 0, "failed attempt", {"score": 0.0, "feedback": "fix the tool call"})
    solved = _sample(11, 1, "successful attempt", {"score": 1.0})
    getattr(feedback_module, feedback_class_name)().prepare_teacher_prompts(
        [failed, solved], [failed.reward, solved.reward]
    )

    teacher_text = str(failed.teacher_prompt)
    assert "successful attempt" in teacher_text
    assert "fix the tool call" in teacher_text
    assert failed.opd_sample_mask is True


@pytest.mark.parametrize(
    "name",
    [
        "EnvironmentFeedback",
        "OPDFeedback",
        "OPSDFeedback",
        "SciKnowEvalSDPOFeedback",
        "ToolUseSDPOFeedback",
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
