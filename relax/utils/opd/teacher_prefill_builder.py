# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from relax.utils.types import Sample


class TeacherPrefillMode(str, Enum):
    PLAIN_OPD = "plain_opd"
    OPSD = "opsd"
    SDPO = "sdpo"


@dataclass(frozen=True)
class TeacherPrefillInput:
    input_ids: list[int]
    prompt_length: int
    image_data: list | None = None

    @property
    def logprob_start_len(self) -> int:
        return max(self.prompt_length - 1, 0)


def get_original_response_ids(sample: "Sample", response_length: int) -> list[int]:
    if response_length < 0:
        raise ValueError(f"response_length must be non-negative, got {response_length}")
    if response_length == 0:
        return []
    if len(sample.tokens) < response_length:
        raise ValueError(
            f"Original sample is shorter than the response: input={len(sample.tokens)}, response={response_length}"
        )
    return list(sample.tokens[-response_length:])


class TeacherPrefillBuilder:
    """Build teacher requests for plain OPD, OPSD, and SDPO."""

    def __init__(self, mode: TeacherPrefillMode, tokenizer: Any | None = None):
        self.mode = TeacherPrefillMode(mode)
        self._tokenizer = tokenizer

    @classmethod
    def from_args(cls, args) -> "TeacherPrefillBuilder":
        if getattr(args, "opd_loss_mode", "opd") == "sdpo":
            return cls(TeacherPrefillMode.SDPO)
        if (
            getattr(args, "opd_teacher_prompt_key", None) is not None
            or getattr(args, "opd_teacher_image_key", None) is not None
        ):
            return cls(TeacherPrefillMode.OPSD)
        return cls(TeacherPrefillMode.PLAIN_OPD)

    async def prepare(self, args, sample: "Sample") -> None:
        if self.mode is TeacherPrefillMode.PLAIN_OPD:
            return
        if self.mode is TeacherPrefillMode.SDPO:
            await self._prepare_sdpo(args, sample)
            return
        await self._prepare_opsd(args, sample)

    async def build(self, sample: "Sample", response_length: int) -> TeacherPrefillInput:
        if self.mode is TeacherPrefillMode.SDPO:
            from relax.utils.opd.sdpo import validate_sdpo_text_only

            validate_sdpo_text_only(sample)

        teacher_input_ids = self.teacher_input_ids(sample, response_length)
        prompt_length = self.teacher_prompt_length(sample, response_length)
        image_data = None
        if self.mode is TeacherPrefillMode.OPSD:
            from relax.utils.opd.opd_utils import build_teacher_preexpanded_image_data

            image_data = await build_teacher_preexpanded_image_data(sample)
        return TeacherPrefillInput(
            input_ids=teacher_input_ids,
            prompt_length=prompt_length,
            image_data=image_data,
        )

    def teacher_input_ids(self, sample: "Sample", response_length: int) -> list[int]:
        original_response = get_original_response_ids(sample, response_length)
        teacher_tokens = getattr(sample, "teacher_tokens", None)
        if self.mode is not TeacherPrefillMode.PLAIN_OPD and teacher_tokens is not None:
            input_ids = list(teacher_tokens)
        else:
            input_ids = list(sample.rollout_tokens or sample.tokens)

        if len(input_ids) < len(original_response):
            raise ValueError(
                "Teacher input is shorter than the original response: "
                f"input={len(input_ids)}, response={len(original_response)}"
            )
        if self.mode is TeacherPrefillMode.SDPO and original_response:
            if input_ids[-response_length:] != original_response:
                raise ValueError(
                    "Teacher input response suffix does not match the original response; "
                    "privileged context must only shift the prompt prefix."
                )
        return input_ids

    def teacher_prompt_length(self, sample: "Sample", response_length: int) -> int:
        input_ids = self.teacher_input_ids(sample, response_length)
        if self.mode is not TeacherPrefillMode.PLAIN_OPD and getattr(sample, "teacher_tokens", None) is not None:
            prompt_length = len(input_ids) - response_length
        else:
            prompt_length = len(sample.tokens) - response_length
        if prompt_length < 0:
            raise AssertionError(
                "Teacher prompt/response offset is inconsistent: "
                f"prompt={prompt_length}, response={response_length}, input={len(input_ids)}"
            )
        return prompt_length

    async def _prepare_sdpo(self, args, sample: "Sample") -> None:
        from relax.utils.opd.sdpo import validate_sdpo_text_only

        validate_sdpo_text_only(sample)
        if not sample.tokens or int(sample.response_length or 0) <= 0 or sample.teacher_prompt is None:
            return

        tokenizer = self._tokenizer
        if tokenizer is None:
            from relax.engine.rollout.sglang_rollout import GenerateState

            tokenizer = GenerateState(args).tokenizer
        if isinstance(sample.teacher_prompt, str):
            teacher_prompt_ids = tokenizer.encode(sample.teacher_prompt, add_special_tokens=False)
        else:
            teacher_prompt_ids = tokenizer.apply_chat_template(
                sample.teacher_prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        max_prompt_length = getattr(args, "rollout_max_prompt_len", None)
        if max_prompt_length is not None and int(max_prompt_length) > 0:
            teacher_prompt_ids = list(teacher_prompt_ids[: int(max_prompt_length)])
        response_ids = get_original_response_ids(sample, int(sample.response_length))
        sample.teacher_tokens = list(teacher_prompt_ids) + response_ids
        sample.teacher_prompt_length = len(teacher_prompt_ids)

    async def _prepare_opsd(self, args, sample: "Sample") -> None:
        teacher_prompt = getattr(sample, "teacher_prompt", None)
        teacher_mm = getattr(sample, "teacher_multimodal_inputs", None)
        teacher_has_media = teacher_mm is not None and bool(teacher_mm.get("images"))

        if teacher_prompt is None and not teacher_has_media:
            if sample.tokens and int(sample.response_length or 0) > 0:
                student_mm_train_inputs = getattr(sample, "multimodal_train_inputs", None)
                student_grid_thw = (student_mm_train_inputs or {}).get("image_grid_thw")
                student_mm_in = sample.multimodal_inputs or {}
                student_raw_images = student_mm_in.get("images") or []
                if student_grid_thw is not None and student_raw_images:
                    cached = student_mm_in.get("_teacher_image_b64_cache")
                    if cached is None:
                        from relax.utils.data.processing_utils import async_encode_image_for_rollout_engine

                        cached = list(await _gather_encode(student_raw_images, async_encode_image_for_rollout_engine))
                        student_mm_in["_teacher_image_b64_cache"] = cached
                    sample.teacher_image_b64_list = cached
                    sample.teacher_image_grid_thw = student_grid_thw
            return

        if not sample.tokens or int(sample.response_length or 0) <= 0:
            return

        from relax.engine.rollout.sglang_rollout import GenerateState, _run_image_processor

        state = GenerateState(args)
        teacher_prompt_for_tokenize = teacher_prompt if teacher_prompt is not None else sample.prompt

        if state.processor is not None and teacher_has_media:
            teacher_prompt_ids, teacher_mm_train_inputs, _ = await _run_image_processor(
                state, args, teacher_prompt_for_tokenize, teacher_mm
            )
            if teacher_mm_train_inputs:
                sample.teacher_image_grid_thw = teacher_mm_train_inputs.get("image_grid_thw")
            teacher_raw_images = (teacher_mm or {}).get("images") or []
            if teacher_raw_images:
                cached = teacher_mm.get("_teacher_image_b64_cache") if teacher_mm else None
                if cached is None:
                    from relax.utils.data.processing_utils import async_encode_image_for_rollout_engine

                    cached = list(await _gather_encode(teacher_raw_images, async_encode_image_for_rollout_engine))
                    if teacher_mm is not None:
                        teacher_mm["_teacher_image_b64_cache"] = cached
                sample.teacher_image_b64_list = cached
        elif isinstance(teacher_prompt_for_tokenize, str):
            teacher_prompt_ids = state.tokenizer.encode(teacher_prompt_for_tokenize, add_special_tokens=False)
        else:
            teacher_prompt_ids = state.tokenizer.apply_chat_template(
                teacher_prompt_for_tokenize,
                tokenize=True,
                add_generation_prompt=True,
            )

        response_ids = get_original_response_ids(sample, int(sample.response_length))
        sample.teacher_tokens = list(teacher_prompt_ids) + response_ids
        sample.teacher_prompt_length = len(teacher_prompt_ids)


async def _gather_encode(images: list, encode_fn) -> list:
    return await asyncio.gather(*(encode_fn(image) for image in images))
