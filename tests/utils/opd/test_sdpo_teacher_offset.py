# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import asyncio
from types import SimpleNamespace

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from relax.utils.opd.teacher_prefill_builder import TeacherPrefillBuilder, TeacherPrefillMode
from relax.utils.types import Sample


def test_teacher_prefill_keeps_original_response_suffix_after_privileged_prompt() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[10, 11, 20, 21],
        response_length=2,
    )
    sample.teacher_tokens = [100, 101, 102, 20, 21]
    sample.teacher_prompt_length = 3

    worker = TeacherPrefillBuilder(TeacherPrefillMode.OPSD)

    assert worker.teacher_input_ids(sample, response_length=2) == [100, 101, 102, 20, 21]
    assert worker.teacher_input_ids(sample, response_length=2)[-2:] == sample.tokens[-2:]
    assert worker.teacher_prompt_length(sample, response_length=2) == 3


def test_sdpo_teacher_prefill_rejects_changed_response_suffix() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[10, 11, 20, 21],
        response_length=2,
        teacher_tokens=[100, 101, 102, 30, 31],
        teacher_prompt_length=3,
    )

    worker = TeacherPrefillBuilder(TeacherPrefillMode.SDPO)

    with pytest.raises(ValueError, match="response suffix"):
        worker.teacher_input_ids(sample, response_length=2)


@pytest.mark.parametrize("mode", [TeacherPrefillMode.PLAIN_OPD, TeacherPrefillMode.OPSD])
def test_teacher_prefill_rejects_response_longer_than_input(mode: TeacherPrefillMode) -> None:
    sample = Sample(tokens=[20, 21], rollout_tokens=[1, 2], response_length=3)

    with pytest.raises(ValueError, match="shorter than the response"):
        TeacherPrefillBuilder(mode).teacher_input_ids(sample, response_length=3)


def test_teacher_prefill_falls_back_to_rollout_tokens_without_dynamic_prompt() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[10, 11, 20, 21],
        response_length=2,
    )

    worker = TeacherPrefillBuilder(TeacherPrefillMode.PLAIN_OPD)

    assert worker.teacher_input_ids(sample, response_length=2) == sample.rollout_tokens
    assert worker.teacher_prompt_length(sample, response_length=2) == 2


def test_teacher_prompt_length_preserves_ordinary_opd_fallback() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[1, 2, 3, 4, 20, 21],
        response_length=2,
    )

    worker = TeacherPrefillBuilder(TeacherPrefillMode.PLAIN_OPD)

    assert worker.teacher_input_ids(sample, response_length=2) == [1, 2, 3, 4, 20, 21]
    assert worker.teacher_prompt_length(sample, response_length=2) == 2


def test_teacher_prefill_inputs_preserve_ordinary_opd_input_and_offset() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[1, 2, 3, 4, 20, 21],
        response_length=2,
    )

    inputs = asyncio.run(TeacherPrefillBuilder(TeacherPrefillMode.PLAIN_OPD).build(sample, response_length=2))

    assert inputs.input_ids == sample.rollout_tokens
    assert inputs.prompt_length == 2
    assert inputs.logprob_start_len == 1
    assert inputs.image_data is None


def test_teacher_prefill_inputs_preserve_preexpanded_opsd_offset() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        response_length=2,
        teacher_tokens=[100, 101, 102, 20, 21],
        teacher_image_b64_list=["image"],
        teacher_image_grid_thw=[[1, 2, 2]],
    )

    inputs = asyncio.run(TeacherPrefillBuilder(TeacherPrefillMode.OPSD).build(sample, response_length=2))

    assert inputs.input_ids == sample.teacher_tokens
    assert inputs.prompt_length == 3
    assert inputs.logprob_start_len == 2
    assert inputs.image_data == [
        {
            "format": "opd_preexpanded_raw",
            "images_b64": ["image"],
            "image_grid_thw": [[1, 2, 2]],
        }
    ]


def test_sdpo_teacher_prefill_input_rejects_image_payload() -> None:
    sample = Sample(
        tokens=[10, 20],
        response_length=1,
        teacher_prompt="privileged context",
        teacher_tokens=[100, 20],
        teacher_image_b64_list=["stale"],
        teacher_image_grid_thw=[[1, 1, 1]],
    )

    with pytest.raises(ValueError, match="SDPO only supports text inputs"):
        asyncio.run(TeacherPrefillBuilder(TeacherPrefillMode.SDPO).build(sample, response_length=1))


def test_ordinary_worker_ignores_stale_privileged_fields() -> None:
    sample = Sample(
        tokens=[10, 11, 20, 21],
        rollout_tokens=[1, 2, 3, 4, 20, 21],
        response_length=2,
        teacher_tokens=[100, 101, 102, 20, 21],
        teacher_image_b64_list=["stale"],
        teacher_image_grid_thw=[[1, 1, 1]],
    )

    worker = TeacherPrefillBuilder(TeacherPrefillMode.PLAIN_OPD)

    assert worker.teacher_input_ids(sample, response_length=2) == sample.rollout_tokens
    assert worker.teacher_prompt_length(sample, response_length=2) == 2


def test_sdpo_worker_rejects_image_before_processing() -> None:
    sample = Sample(
        prompt="question",
        tokens=[10, 20],
        response_length=1,
        multimodal_inputs={"images": [b"image"]},
    )

    worker = TeacherPrefillBuilder(TeacherPrefillMode.SDPO)

    with pytest.raises(ValueError, match="SDPO only supports text inputs"):
        asyncio.run(worker.prepare(object(), sample))


def test_sdpo_dynamic_teacher_input_uses_text_path_only() -> None:
    class Tokenizer:
        def encode(self, prompt, add_special_tokens=False):
            assert prompt == "privileged context"
            return [100, 101]

    sample = Sample(
        prompt="question",
        teacher_prompt="privileged context",
        tokens=[10, 20],
        response_length=1,
    )
    worker = TeacherPrefillBuilder(TeacherPrefillMode.SDPO, tokenizer=Tokenizer())

    asyncio.run(worker.prepare(object(), sample))

    assert sample.teacher_tokens == [100, 101, 20]
    assert sample.teacher_prompt_length == 2


def test_sdpo_chat_template_normalizes_batch_encoding_to_token_ids() -> None:
    class Tokenizer:
        def apply_chat_template(self, prompt, **kwargs):
            assert kwargs["return_dict"] is False
            return BatchEncoding({"input_ids": [100, 101], "attention_mask": [1, 1]})

    sample = Sample(
        prompt="question",
        teacher_prompt=[{"role": "user", "content": "privileged context"}],
        tokens=[10, 20],
        response_length=1,
    )
    worker = TeacherPrefillBuilder(TeacherPrefillMode.SDPO, tokenizer=Tokenizer())

    asyncio.run(worker.prepare(object(), sample))

    assert sample.teacher_tokens == [100, 101, 20]
    assert all(isinstance(token_id, int) for token_id in sample.teacher_tokens)


@pytest.mark.parametrize("encoding_attribute", ["input_ids", "ids"])
def test_sdpo_dynamic_teacher_input_normalizes_tokenizer_encoding(encoding_attribute: str) -> None:
    class Encoding:
        def __init__(self) -> None:
            setattr(self, encoding_attribute, [100, 101])

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            assert messages == [{"role": "user", "content": "privileged context"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return Encoding()

    sample = Sample(
        prompt="question",
        teacher_prompt=[{"role": "user", "content": "privileged context"}],
        tokens=[10, 20],
        response_length=1,
    )
    worker = TeacherPrefillBuilder(TeacherPrefillMode.SDPO, tokenizer=Tokenizer())

    asyncio.run(worker.prepare(object(), sample))

    assert sample.teacher_tokens == [100, 101, 20]
    assert all(isinstance(token_id, int) for token_id in sample.teacher_tokens)


def test_sdpo_dynamic_teacher_input_normalizes_real_tokenizers_encoding() -> None:
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "privileged": 100, "context": 101}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    class ChatTokenizer:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            assert messages == [{"role": "user", "content": "privileged context"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return tokenizer.encode("privileged context")

    sample = Sample(
        prompt="question",
        teacher_prompt=[{"role": "user", "content": "privileged context"}],
        tokens=[10, 20],
        response_length=1,
    )
    worker = TeacherPrefillBuilder(TeacherPrefillMode.SDPO, tokenizer=ChatTokenizer())

    asyncio.run(worker.prepare(object(), sample))

    assert sample.teacher_tokens == [100, 101, 20]
    assert all(isinstance(token_id, int) for token_id in sample.teacher_tokens)


@pytest.mark.parametrize(
    ("args", "expected_mode"),
    [
        (
            SimpleNamespace(opd_loss_mode="opd", opd_teacher_prompt_key=None, opd_teacher_image_key=None),
            TeacherPrefillMode.PLAIN_OPD,
        ),
        (
            SimpleNamespace(opd_loss_mode="opd", opd_teacher_prompt_key="teacher_prompt", opd_teacher_image_key=None),
            TeacherPrefillMode.OPSD,
        ),
        (
            SimpleNamespace(opd_loss_mode="sdpo", opd_teacher_prompt_key=None, opd_teacher_image_key=None),
            TeacherPrefillMode.SDPO,
        ),
    ],
)
def test_teacher_prefill_builder_derives_all_modes_from_args(args, expected_mode) -> None:
    assert TeacherPrefillBuilder.from_args(args).mode is expected_mode


@pytest.mark.parametrize(
    "mode",
    [TeacherPrefillMode.PLAIN_OPD, TeacherPrefillMode.OPSD, TeacherPrefillMode.SDPO],
)
def test_teacher_prefill_builder_modes_share_one_input_contract(mode) -> None:
    sample = Sample(
        tokens=[10, 11, 20],
        rollout_tokens=[1, 2, 3, 20],
        response_length=1,
        teacher_prompt="privileged context" if mode is TeacherPrefillMode.SDPO else None,
        teacher_tokens=[100, 101, 20] if mode is not TeacherPrefillMode.PLAIN_OPD else [999, 20],
    )

    teacher_input = asyncio.run(TeacherPrefillBuilder(mode).build(sample, response_length=1))

    assert isinstance(teacher_input.input_ids, list)
    assert isinstance(teacher_input.prompt_length, int)
    assert teacher_input.logprob_start_len == max(teacher_input.prompt_length - 1, 0)
    assert teacher_input.image_data is None
    if mode is TeacherPrefillMode.PLAIN_OPD:
        assert teacher_input.input_ids == sample.rollout_tokens
        assert teacher_input.prompt_length == len(sample.tokens) - sample.response_length
    else:
        assert teacher_input.input_ids == sample.teacher_tokens
        assert teacher_input.prompt_length == len(sample.teacher_tokens) - sample.response_length
