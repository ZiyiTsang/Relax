# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import pytest

from relax.utils.opd.sdpo.prompt_builder import (
    FeedbackProvider,
    FeedbackRecord,
    SdpoPromptBuilder,
    SdpoPromptStats,
    TeacherPromptRenderer,
    prepare_sdpo_teacher_prompts,
    validate_sdpo_text_only,
)
from relax.utils.types import Sample


def _sample(group_index: int, index: int, response: str, reward: dict) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        prompt="rendered student prompt",
        response=response,
        tokens=[10, 11, 12],
        response_length=2,
        reward=reward,
        metadata={"sdpo_prompt": "original question"},
    )


def test_sdpo_package_exports_public_prompt_api() -> None:
    from relax.utils.opd import sdpo

    expected = {
        "FeedbackRecord",
        "FeedbackProvider",
        "SdpoPromptBuilder",
        "SdpoPromptStats",
        "TeacherPromptRenderer",
        "prepare_sdpo_teacher_prompts",
        "SDPO_TOKEN_SELECTION",
        "validate_sdpo_text_only",
    }

    assert expected.issubset(set(sdpo.__all__))
    for name in expected:
        assert hasattr(sdpo, name)


def test_prepare_sdpo_teacher_prompt_uses_same_group_and_preserves_student_tokens() -> None:
    failed = _sample(3, 0, "wrong response", {"score": 0.0, "feedback": "check the second step"})
    success = _sample(3, 1, "correct response", {"score": 1.0, "feedback": "success context"})
    other_group = _sample(4, 2, "unrelated response", {"score": 0.0, "feedback": "other fix"})
    original_tokens = list(failed.tokens)

    stats = prepare_sdpo_teacher_prompts([failed, success, other_group])

    assert stats.valid_samples == 3
    assert failed.teacher_prompt is not None
    assert success.teacher_prompt is not None
    assert "success context" in success.teacher_prompt[-1]["content"]
    assert other_group.teacher_prompt is not None
    assert failed.tokens == original_tokens
    assert failed.teacher_prompt != failed.prompt
    prompt_text = failed.teacher_prompt[-1]["content"]
    assert "correct response" in prompt_text
    assert "check the second step" in prompt_text
    assert "unrelated response" not in prompt_text


def test_prepare_sdpo_teacher_prompt_rejects_self_success_without_context() -> None:
    success = _sample(7, 0, "only success", {"score": 1.0, "feedback": ""})
    with pytest.raises(ValueError, match="privileged teacher context"):
        prepare_sdpo_teacher_prompts([success], exclude_self_success=True)


def test_prepare_sdpo_teacher_prompt_can_use_feedback_without_solution() -> None:
    sample = _sample(9, 0, "bad", {"score": 0.0, "feedback": "format is invalid"})

    stats = prepare_sdpo_teacher_prompts([sample])

    assert stats.feedback_available == 1
    assert stats.feedback_used == 1
    assert sample.teacher_prompt is not None
    assert "format is invalid" in sample.teacher_prompt[-1]["content"]


def test_prompt_builder_accepts_scalar_reward_as_success() -> None:
    failed = _sample(10, 0, "wrong", {"score": 0.0})
    successful = Sample(
        group_index=10,
        index=1,
        prompt="rendered student prompt",
        response="correct",
        reward=1.0,
        metadata={"sdpo_prompt": "original question"},
    )

    stats = prepare_sdpo_teacher_prompts([failed, successful])

    assert stats.successful_demonstrations == 1
    assert failed.teacher_prompt is not None
    assert "correct" in failed.teacher_prompt[-1]["content"]


def test_prepare_sdpo_teacher_prompt_does_not_cross_group_when_group_id_missing() -> None:
    first = _sample(1, 0, "first", {"score": 0.0, "feedback": "fix"})
    second = _sample(2, 1, "second", {"score": 0.0, "feedback": "second fix"})
    first.group_index = None
    second.group_index = None

    prepare_sdpo_teacher_prompts([first, second])

    assert first.teacher_prompt is not None
    assert "second" not in first.teacher_prompt[-1]["content"]
    assert second.teacher_prompt is not None


def test_sdpo_components_can_be_composed_with_environment_feedback() -> None:
    class EnvironmentFeedback:
        def extract(self, sample: Sample) -> FeedbackRecord:
            if sample.index == 1:
                return FeedbackRecord(score=1.0, feedback="success context", is_success=True)
            return FeedbackRecord(score=0.0, feedback="environment says to retry")

    failed = _sample(11, 0, "failed", {"score": 0.0})
    success = _sample(11, 1, "successful", {"score": 1.0})
    builder = SdpoPromptBuilder(
        feedback_provider=FeedbackProvider(extractor=EnvironmentFeedback().extract),
        prompt_renderer=TeacherPromptRenderer(),
    )

    stats = builder.apply([failed, success])

    assert isinstance(stats, SdpoPromptStats)
    assert stats.valid_samples == 2
    assert failed.teacher_prompt is not None
    assert "successful" in failed.teacher_prompt[-1]["content"]
    assert "environment says to retry" in failed.teacher_prompt[-1]["content"]
    assert success.teacher_prompt is not None
    assert "success context" in success.teacher_prompt[-1]["content"]


def test_prompt_builder_skips_current_success() -> None:
    current = _sample(12, 0, "first successful", {"score": 1.0})
    next_success = _sample(12, 1, "second successful", {"score": 1.0})

    prepare_sdpo_teacher_prompts([current, next_success])

    assert "second successful" in current.teacher_prompt[-1]["content"]
    assert "first successful" in next_success.teacher_prompt[-1]["content"]


def test_teacher_prompt_renderer_preserves_source_chat_prompt() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    sample = Sample(
        prompt="student prompt",
        response="response",
        metadata={"sdpo_prompt": messages},
    )

    rendered = TeacherPromptRenderer().render(sample, solution="solution", feedback="feedback")

    assert messages[-1]["content"] == "question"
    assert rendered[0] == messages[0]
    assert "solution" in rendered[-1]["content"]
    assert "feedback" in rendered[-1]["content"]


def test_sdpo_prompt_stats_use_context_and_group_denominators() -> None:
    failed = _sample(13, 0, "failed", {"score": 0.0, "feedback": "retry"})
    success = _sample(13, 1, "success", {"score": 1.0, "feedback": "success context"})
    other = _sample(14, 2, "other", {"score": 0.0, "feedback": "other fix"})

    stats = prepare_sdpo_teacher_prompts([failed, success, other])

    metrics = stats.as_dict()
    assert metrics["valid_teacher_context_ratio"] == 1.0
    assert metrics["successful_group_ratio"] == 1 / 2


def test_sdpo_rejects_image_fields_before_prompt_routing() -> None:
    sample = _sample(15, 0, "response", {"score": 0.0, "feedback": "retry"})
    sample.multimodal_inputs = {"images": [b"image"]}

    with pytest.raises(ValueError, match="SDPO only supports text inputs"):
        validate_sdpo_text_only(sample)
    with pytest.raises(ValueError, match="SDPO only supports text inputs"):
        prepare_sdpo_teacher_prompts([sample])


def test_sdpo_rejects_structured_image_message_content() -> None:
    sample = _sample(16, 0, "response", {"score": 0.0, "feedback": "retry"})
    sample.prompt = [{"role": "user", "content": [{"type": "image", "image": "x"}]}]

    with pytest.raises(ValueError, match="string message content"):
        prepare_sdpo_teacher_prompts([sample])


def test_sdpo_accepts_empty_media_placeholders() -> None:
    sample = _sample(17, 0, "response", {"score": 0.0, "feedback": "retry"})
    sample.multimodal_inputs = {"images": [], "videos": [], "audio": []}
    sample.multimodal_train_inputs = {"image_grid_thw": []}

    validate_sdpo_text_only(sample)
