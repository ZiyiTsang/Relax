# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Context-parallel layout adapters for OPD Top-K rollout fields."""

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from relax.utils.types import RolloutBatch


def get_opd_local_response_lengths(
    rollout_data: RolloutBatch,
    args: Namespace,
    *,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
) -> list[int]:
    """Return response-row counts owned by the current CP rank.

    The same offsets are consumed by Top-K payload slicing, loss reduction, and
    SDPO.  Keeping this calculation in one adapter prevents a CP rank from
    pairing a local student row with a full-response mask.
    """

    from relax.backends.megatron.cp_utils import get_logits_and_tokens_offset_with_cp

    if dynamic_cp_size is None:
        from megatron.core import mpu

        cp_size = mpu.get_context_parallel_world_size()
    else:
        cp_size = dynamic_cp_size
    response_lengths = [int(length) for length in rollout_data["response_lengths"]]
    if cp_size <= 1:
        return response_lengths

    total_lengths = rollout_data["total_lengths"]
    max_seq_lens = rollout_data.get("max_seq_lens")
    padded_total_lengths = rollout_data.get("padded_total_lengths")
    local_lengths: list[int] = []
    for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=True)):
        _, _, logits_offsets, _ = get_logits_and_tokens_offset_with_cp(
            int(total_length),
            response_length,
            args.qkv_format,
            max_seq_lens[i] if max_seq_lens is not None else None,
            padded_total_lengths[i] if padded_total_lengths is not None else None,
            dynamic_cp_size=dynamic_cp_size,
            dynamic_cp_rank=dynamic_cp_rank,
        )
        local_lengths.append(sum(end - start for start, end in logits_offsets))
    return local_lengths


def slice_opd_topk_rollout_fields(
    rollout_data: RolloutBatch,
    args: Namespace,
    *,
    dynamic_cp_size: int | None = None,
    dynamic_cp_rank: int | None = None,
) -> None:
    """Slice full-response OPD Top-K fields to the current CP response rows."""

    if args.opd_token_selection not in ("student_topk", "teacher_topk", "union"):
        return

    from relax.backends.megatron.cp_utils import slice_log_prob_with_cp
    from relax.utils.opd.opd_main_worker import TopkWorker

    if dynamic_cp_size is None:
        from megatron.core import mpu

        cp_size = mpu.get_context_parallel_world_size()
    else:
        cp_size = dynamic_cp_size
    if cp_size <= 1:
        return
    if getattr(args, "allgather_cp", False):
        raise NotImplementedError(
            "OPD Top-K fields require zig-zag CP slicing; allgather_cp=True is not supported yet."
        )

    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    max_seq_lens = rollout_data.get("max_seq_lens")
    padded_total_lengths = rollout_data.get("padded_total_lengths")
    local_response_lengths = get_opd_local_response_lengths(
        rollout_data,
        args,
        dynamic_cp_size=dynamic_cp_size,
        dynamic_cp_rank=dynamic_cp_rank,
    )

    fields = (
        TopkWorker.TRANSFER_TOKEN_IDS,
        TopkWorker.TRANSFER_STUDENT_LOG_PROBS,
        TopkWorker.TRANSFER_TEACHER_LOG_PROBS,
    )
    for field in fields:
        values = rollout_data.get(field)
        if values is None:
            continue
        if len(values) != len(total_lengths):
            raise ValueError(f"OPD Top-K field {field!r} sample count does not match total_lengths")

        sliced_values = []
        for i, value in enumerate(values):
            if value is None:
                sliced_values.append(None)
                continue
            tensor = torch.as_tensor(value)
            if tensor.numel() == 0:
                k_size = tensor.shape[-1] if tensor.ndim > 1 else 0
                sliced_values.append(tensor.reshape(0, k_size))
                continue
            if tensor.ndim != 2:
                raise ValueError(f"OPD Top-K field {field!r} must have shape [response, K], got {tensor.shape}")
            if tensor.size(1) == 0:
                raise ValueError(f"OPD Top-K field {field!r} has zero K for non-empty response rows")

            columns = [
                slice_log_prob_with_cp(
                    tensor[:, column],
                    int(total_lengths[i]),
                    int(response_lengths[i]),
                    args.qkv_format,
                    max_seq_lens[i] if max_seq_lens is not None else None,
                    padded_total_length=padded_total_lengths[i] if padded_total_lengths is not None else None,
                    dynamic_cp_size=dynamic_cp_size,
                    dynamic_cp_rank=dynamic_cp_rank,
                )
                for column in range(tensor.size(1))
            ]
            sliced_tensor = torch.stack(columns, dim=1)
            expected_rows = local_response_lengths[i]
            if sliced_tensor.size(0) != expected_rows:
                raise ValueError(
                    "OPD Top-K CP row mismatch: "
                    f"field={field!r}, sample={i}, rows={sliced_tensor.size(0)}, expected={expected_rows}"
                )
            sliced_values.append(sliced_tensor)
        rollout_data[field] = sliced_values

    if args.opd_token_selection != "union":
        return

    values = rollout_data.get(TopkWorker.TRANSFER_K_LENGTHS)
    if values is None:
        return
    if len(values) != len(total_lengths):
        raise ValueError("OPD union Top-K lengths sample count does not match total_lengths")

    sliced_lengths = []
    for i, value in enumerate(values):
        if value is None:
            sliced_lengths.append(None)
            continue
        tensor = torch.as_tensor(value)
        if tensor.numel() == 0:
            sliced_lengths.append(tensor.reshape(0))
            continue
        sliced = slice_log_prob_with_cp(
            tensor,
            int(total_lengths[i]),
            int(response_lengths[i]),
            args.qkv_format,
            max_seq_lens[i] if max_seq_lens is not None else None,
            padded_total_length=padded_total_lengths[i] if padded_total_lengths is not None else None,
            dynamic_cp_size=dynamic_cp_size,
            dynamic_cp_rank=dynamic_cp_rank,
        )
        expected_rows = local_response_lengths[i]
        if len(sliced) != expected_rows:
            raise ValueError(f"OPD union Top-K row mismatch: sample={i}, rows={len(sliced)}, expected={expected_rows}")
        sliced_lengths.append(sliced)
    rollout_data[TopkWorker.TRANSFER_K_LENGTHS] = sliced_lengths


__all__ = ["get_opd_local_response_lengths", "slice_opd_topk_rollout_fields"]
