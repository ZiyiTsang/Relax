# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Static-teacher SDPO criterion built on Relax's OPD Top-K primitives."""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from relax.utils.opd.topk_mask import OPDDistributionTopKMaskEstimator


if TYPE_CHECKING:
    from relax.utils.types import RolloutBatch


def compute_sdpo_topk_divergence(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    kl_type: str,
    jsd_alpha: float,
    norm_mode: str = "tail",
    log_prob_min_clamp: float | None = None,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one SDPO divergence value for each response position.

    The inputs contain *unnormalized* vocabulary probabilities for the same
    student-selected token ids. ``tail`` appends one bucket containing all
    omitted vocabulary mass, which is the approximation used by the SDPO
    reference implementation. ``support_mask`` is only needed for padded
    support rows; it is lifted to the tail-augmented distribution by the shared
    OPD mask estimator. The teacher is a detached target; the student remains
    differentiable. SDPO uses explicit endpoint aliases: ``jsd_alpha=0`` is
    ``KL(teacher || student)`` and ``jsd_alpha=1`` is
    ``KL(student || teacher)``. This is intentionally separate from ordinary
    OPD's alpha convention.
    """

    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError(
            "SDPO Top-K shape mismatch: "
            f"student={tuple(student_log_probs.shape)}, teacher={tuple(teacher_log_probs.shape)}"
        )
    if student_log_probs.ndim != 2:
        raise ValueError(f"SDPO Top-K log-probs must have shape [response, K], got {student_log_probs.shape}")
    if not 0.0 <= jsd_alpha <= 1.0:
        raise ValueError(f"jsd_alpha must be in [0, 1], got {jsd_alpha}")
    if support_mask is not None and support_mask.shape != student_log_probs.shape:
        raise ValueError(
            "SDPO Top-K support mask shape mismatch: "
            f"mask={tuple(support_mask.shape)}, values={tuple(student_log_probs.shape)}"
        )
    if norm_mode == "trunc":
        raise ValueError("SDPO does not support truncation without renormalization; use tail or norm")
    if norm_mode not in ("tail", "norm"):
        raise ValueError(f"Unknown SDPO norm mode: {norm_mode}")

    if torch.isnan(teacher_log_probs).any() or torch.isposinf(teacher_log_probs).any():
        raise ValueError("SDPO teacher log-probs may contain -inf, but not NaN or +inf")

    student = student_log_probs.float()
    teacher = teacher_log_probs.float().detach()
    support_estimate = OPDDistributionTopKMaskEstimator.estimate(
        support_mask,
        norm_mode=norm_mode,
        include_tail_bucket=norm_mode == "tail",
    )
    if support_mask is not None:
        student = student.masked_fill(~support_mask, float("-inf"))
        teacher = teacher.masked_fill(~support_mask, float("-inf"))
    if log_prob_min_clamp is not None:
        student = student.clamp_min(log_prob_min_clamp)
        teacher = teacher.clamp_min(log_prob_min_clamp)
        if support_mask is not None:
            student = student.masked_fill(~support_mask, float("-inf"))
            teacher = teacher.masked_fill(~support_mask, float("-inf"))

    def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
        log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True).clamp(max=-1e-7)
        tail = torch.log(-torch.expm1(log_mass))
        return torch.cat([log_probs, tail], dim=-1)

    if norm_mode == "tail":
        student_dist = add_tail(student)
        teacher_dist = add_tail(teacher)
    elif norm_mode == "norm":
        student_dist = student - torch.logsumexp(student, dim=-1, keepdim=True)
        teacher_dist = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)

    distribution_mask = support_estimate.distribution_mask
    if distribution_mask is not None:
        student_for_loss = torch.where(distribution_mask, student_dist, torch.zeros_like(student_dist))
        teacher_for_loss = torch.where(distribution_mask, teacher_dist, torch.zeros_like(teacher_dist))
    else:
        student_for_loss = student_dist
        teacher_for_loss = teacher_dist

    def masked_sum(values: torch.Tensor) -> torch.Tensor:
        if distribution_mask is not None:
            values = values.masked_fill(~distribution_mask, 0.0)
        return values.sum(dim=-1)

    def weighted_log_ratio(
        log_weight: torch.Tensor,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
    ) -> torch.Tensor:
        values = log_weight.exp() * (log_p - log_q)
        return torch.where(torch.isneginf(log_weight), torch.zeros_like(values), values)

    if kl_type == "forward_kl":
        # Reference SDPO convention: alpha=0 is KL(teacher || student).
        return masked_sum(weighted_log_ratio(teacher_for_loss, teacher_for_loss, student_for_loss))
    if kl_type == "reverse_kl":
        return masked_sum(weighted_log_ratio(student_for_loss, student_for_loss, teacher_for_loss))
    if kl_type != "jsd":
        raise ValueError(f"Unknown SDPO KL type: {kl_type}")

    if jsd_alpha == 0.0:
        return masked_sum(weighted_log_ratio(teacher_for_loss, teacher_for_loss, student_for_loss))
    if jsd_alpha == 1.0:
        return masked_sum(weighted_log_ratio(student_for_loss, student_for_loss, teacher_for_loss))

    log_alpha = torch.log(torch.tensor(jsd_alpha, dtype=student_dist.dtype, device=student_dist.device))
    log_one_minus_alpha = torch.log(
        torch.tensor(1.0 - jsd_alpha, dtype=student_dist.dtype, device=student_dist.device)
    )
    mixture = torch.logsumexp(
        torch.stack(
            [
                student_dist + log_one_minus_alpha,
                teacher_dist + log_alpha,
            ],
            dim=0,
        ),
        dim=0,
    )
    student_kl = masked_sum(weighted_log_ratio(student_for_loss, student_for_loss, mixture))
    teacher_kl = masked_sum(weighted_log_ratio(teacher_for_loss, teacher_for_loss, mixture))
    return (1.0 - jsd_alpha) * student_kl + jsd_alpha * teacher_kl


def _context_parallel_size(batch: RolloutBatch) -> int:
    dynamic_cp_size = batch.get("dynamic_cp_size")
    if dynamic_cp_size is not None:
        return int(dynamic_cp_size)
    try:
        from megatron.core import mpu

        return int(mpu.get_context_parallel_world_size())
    except (ImportError, RuntimeError):
        return 1


def validate_sdpo_topk_payload(
    *,
    token_ids: object,
    teacher_log_probs: object,
    response_rows: int,
    top_k: int,
    sample_index: int,
) -> None:
    """Validate the complete student-top-k teacher payload for one sample."""

    if response_rows == 0:
        return
    if token_ids is None or teacher_log_probs is None:
        raise ValueError(f"SDPO sample {sample_index} is missing its complete teacher Top-K payload")

    token_ids_tensor = torch.as_tensor(token_ids)
    teacher_tensor = torch.as_tensor(teacher_log_probs)
    expected_shape = (response_rows, top_k)
    if tuple(token_ids_tensor.shape) != expected_shape:
        raise ValueError(
            f"SDPO sample {sample_index} token-id payload shape {tuple(token_ids_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if tuple(teacher_tensor.shape) != expected_shape:
        raise ValueError(
            f"SDPO sample {sample_index} teacher payload shape {tuple(teacher_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if torch.isnan(teacher_tensor).any() or torch.isposinf(teacher_tensor).any():
        raise ValueError(f"SDPO sample {sample_index} teacher Top-K payload contains NaN or +inf")


def validate_sdpo_student_topk_ids(
    *,
    token_ids: object,
    response_rows: int,
    top_k: int,
    sample_index: int,
) -> None:
    """Validate student rollout ids before sending the teacher request."""

    if token_ids is None:
        raise ValueError(f"SDPO sample {sample_index} is missing student Top-K token ids")
    if response_rows <= 0 or top_k <= 0:
        raise ValueError(
            f"SDPO sample {sample_index} has invalid Top-K dimensions: rows={response_rows}, top_k={top_k}"
        )

    token_ids_tensor = torch.as_tensor(token_ids)
    expected_shape = (response_rows, top_k)
    if tuple(token_ids_tensor.shape) != expected_shape:
        raise ValueError(
            f"SDPO sample {sample_index} student Top-K id shape {tuple(token_ids_tensor.shape)} "
            f"does not match {expected_shape}"
        )
    if token_ids_tensor.dtype == torch.bool or token_ids_tensor.dtype.is_floating_point:
        raise ValueError(f"SDPO sample {sample_index} student Top-K ids must be integer token ids")
    if token_ids_tensor.numel() and bool((token_ids_tensor < 0).any()):
        raise ValueError(f"SDPO sample {sample_index} student Top-K ids contain a negative token id")


def _local_response_lengths(batch: RolloutBatch, args: Namespace) -> list[int]:
    response_lengths = [int(length) for length in batch["response_lengths"]]
    cp_size = _context_parallel_size(batch)
    if cp_size <= 1:
        return response_lengths

    from relax.backends.megatron.cp_utils import get_logits_and_tokens_offset_with_cp

    dynamic_cp_size = batch.get("dynamic_cp_size")
    dynamic_cp_rank = batch.get("dynamic_cp_rank")
    max_seq_lens = batch.get("max_seq_lens")
    padded_total_lengths = batch.get("padded_total_lengths")
    return [
        sum(
            end - start
            for start, end in get_logits_and_tokens_offset_with_cp(
                int(total_length),
                response_length,
                args.qkv_format,
                max_seq_lens[i] if max_seq_lens is not None else None,
                padded_total_lengths[i] if padded_total_lengths is not None else None,
                dynamic_cp_size=dynamic_cp_size,
                dynamic_cp_rank=dynamic_cp_rank,
            )[2]
        )
        for i, (total_length, response_length) in enumerate(zip(batch["total_lengths"], response_lengths, strict=True))
    ]


def _reduce_sdpo_values(
    values: torch.Tensor,
    batch: RolloutBatch,
    args: Namespace,
    reducer: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    local_lengths = _local_response_lengths(batch, args)
    if sum(local_lengths) != values.numel():
        raise ValueError(
            "SDPO response rows do not match the student forward: "
            f"values={values.numel()}, expected={sum(local_lengths)}"
        )
    loss_masks = batch["loss_masks"]
    if len(loss_masks) != len(local_lengths):
        raise ValueError("SDPO loss mask and response length sample counts differ")

    if reducer is None:
        from relax.backends.megatron.cp_utils import get_sum_of_sample_mean

        reducer = get_sum_of_sample_mean(
            batch["total_lengths"],
            batch["response_lengths"],
            loss_masks,
            getattr(args, "calculate_per_token_loss", False),
            args.qkv_format,
            batch.get("max_seq_lens"),
            batch.get("padded_total_lengths"),
            dynamic_cp_size=batch.get("dynamic_cp_size"),
            dynamic_cp_rank=batch.get("dynamic_cp_rank"),
        )
    return reducer(values)


def compute_sdpo_loss(
    *,
    args: Namespace,
    batch: RolloutBatch,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    log_probs_and_entropy: dict[str, list[torch.Tensor]],
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    """Compute SDPO's student-top-k distillation loss.

    Every non-empty response must have a complete teacher payload. Invalid
    samples are rejected before the loss is reduced.
    """

    coefficient = float(getattr(args, "opd_loss_coef", 0.0) or 0.0)
    if coefficient == 0.0:
        return None, {}
    if getattr(args, "opd_token_selection", None) != "student_topk":
        raise ValueError("SDPO criterion only supports student_topk token selection")
    if not getattr(args, "calculate_per_token_loss", False):
        raise ValueError("SDPO requires --calculate-per-token-loss for reference token-mean reduction")

    sample_count = len(batch["response_lengths"])
    if len(batch["loss_masks"]) != sample_count:
        raise ValueError("SDPO loss mask and response length sample counts differ")
    if any(int(response_length) <= 0 for response_length in batch["response_lengths"]):
        raise ValueError("SDPO requires a non-empty response for every training sample")
    local_lengths = _local_response_lengths(batch, args)
    if sum(local_lengths) != log_probs.numel():
        raise ValueError(
            "SDPO response rows do not match the student forward: "
            f"student={log_probs.numel()}, expected={sum(local_lengths)}"
        )

    teacher_rows = batch.get("opd_topk_teacher_log_probs")
    token_id_rows = batch.get("opd_topk_token_ids")
    student_rows = log_probs_and_entropy.get("topk_log_probs")
    if not all(
        isinstance(rows, list) and len(rows) == sample_count for rows in (teacher_rows, token_id_rows, student_rows)
    ):
        raise ValueError("SDPO requires student top-k, teacher top-k, and token-id payloads for every batch sample")

    device = log_probs.device
    per_token_values: list[torch.Tensor] = []
    covered_mass = log_probs.new_zeros(())
    covered_rows = log_probs.new_zeros(())
    for index, expected_rows in enumerate(local_lengths):
        student = student_rows[index]
        teacher = teacher_rows[index]
        token_ids = token_id_rows[index]

        if expected_rows == 0:
            per_token_values.append(log_probs.new_zeros(0))
            continue

        top_k = int(getattr(args, "opd_log_prob_top_k", 0) or 0)
        if top_k <= 0 and token_ids is not None:
            token_ids_shape = torch.as_tensor(token_ids).shape
            top_k = int(token_ids_shape[1]) if len(token_ids_shape) == 2 else 0
        validate_sdpo_topk_payload(
            token_ids=token_ids,
            teacher_log_probs=teacher,
            response_rows=expected_rows,
            top_k=top_k,
            sample_index=index,
        )
        if student is None:
            raise ValueError(f"SDPO sample {index} is missing student Top-K log-probs")
        student = torch.as_tensor(student, device=device)
        if student.ndim != 2 or tuple(student.shape) != tuple(torch.as_tensor(teacher).shape):
            raise ValueError(
                f"SDPO sample {index} student payload shape {tuple(student.shape)} "
                f"does not match teacher shape {tuple(torch.as_tensor(teacher).shape)}"
            )
        teacher = torch.as_tensor(teacher, device=device)
        token_ids = torch.as_tensor(token_ids, device=device)
        student = student.float()
        teacher = teacher.to(dtype=student.dtype).detach()
        if token_ids.shape != student.shape:
            raise ValueError(
                "SDPO Top-K row alignment mismatch: "
                f"sample={index}, expected_rows={expected_rows}, "
                f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}, ids={tuple(token_ids.shape)}"
            )
        per_token_values.append(
            compute_sdpo_topk_divergence(
                student,
                teacher,
                kl_type=args.opd_kl_type,
                jsd_alpha=float(args.opd_jsd_alpha),
                norm_mode=getattr(args, "opd_norm_mode", "tail"),
                log_prob_min_clamp=getattr(args, "opd_log_prob_min_clamp", None),
            ).to(dtype=log_probs.dtype)
        )
        covered_rows = covered_rows + expected_rows
        covered_mass = covered_mass + teacher.exp().sum()

    values = torch.cat(per_token_values, dim=0) if per_token_values else log_probs.new_zeros((0,))

    per_token_clip = getattr(args, "opd_per_token_clip", None)
    reported: dict[str, torch.Tensor] = {}
    if per_token_clip is not None and values.numel() > 0:
        limit = float(per_token_clip)
        before_clip = values
        values = values.clamp(max=limit)
        reported["opd_per_token_clip_frac"] = (before_clip > limit).float().mean().detach()

    is_clip = getattr(args, "opd_is_clip", None)
    if is_clip is not None:
        if old_log_probs.numel() != values.numel():
            raise ValueError("SDPO importance-sampling inputs do not match response rows")
        ratio = torch.exp(log_probs.detach() - old_log_probs.detach()).clamp(max=float(is_clip))
        values = values * ratio.to(dtype=values.dtype)

    loss = _reduce_sdpo_values(values, batch, args, sum_of_sample_mean)
    loss = loss + log_probs.sum() * 0.0
    coverage = covered_mass / torch.clamp_min(covered_rows, 1)
    reported["sdpo_topk_coverage"] = coverage.detach()
    reported["opd_kl"] = _reduce_sdpo_values(values.detach(), batch, args, sum_of_sample_mean)
    reported["__sdpo_topk_coverage_denominator"] = covered_rows.detach()
    return coefficient * loss, reported


__all__ = [
    "compute_sdpo_loss",
    "compute_sdpo_topk_divergence",
    "validate_sdpo_student_topk_ids",
    "validate_sdpo_topk_payload",
]
