# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Strict teacher-payload contracts for the SDPO training criterion."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from relax.utils.opd.sdpo.loss import compute_sdpo_loss


def _args() -> Namespace:
    return Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="jsd",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_topk",
        opd_log_prob_top_k=2,
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
        calculate_per_token_loss=True,
        qkv_format="thd",
    )


def _batch(
    *,
    token_ids: torch.Tensor,
    teacher: torch.Tensor,
) -> dict:
    batch = {
        "total_lengths": [4],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
        "opd_topk_token_ids": [token_ids],
        "opd_topk_teacher_log_probs": [teacher],
    }
    return batch


def _compute(batch: dict, student: torch.Tensor):
    log_probs = torch.zeros(2, requires_grad=True)
    loss, metrics = compute_sdpo_loss(
        args=_args(),
        batch=batch,
        log_probs=log_probs,
        old_log_probs=log_probs.detach(),
        log_probs_and_entropy={"topk_log_probs": [student]},
    )
    return loss, metrics, log_probs


@pytest.mark.parametrize(
    ("token_ids", "teacher"),
    [
        (torch.empty((0, 2), dtype=torch.long), torch.empty((0, 2))),
        (torch.ones((2, 2), dtype=torch.long), torch.empty((0, 2))),
        (torch.empty((0, 2), dtype=torch.long), torch.ones((2, 2))),
    ],
)
def test_sdpo_rejects_empty_and_partial_payloads(
    token_ids: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    student = torch.empty((2, 0), requires_grad=True)
    with pytest.raises(ValueError, match="payload shape|complete teacher"):
        _compute(_batch(token_ids=token_ids, teacher=teacher), student)


@pytest.mark.parametrize(
    ("token_ids", "teacher"),
    [
        (torch.empty((0, 2), dtype=torch.long), torch.empty((0, 2))),
        (torch.ones((2, 2), dtype=torch.long), torch.empty((0, 2))),
        (torch.empty((0, 2), dtype=torch.long), torch.ones((2, 2))),
    ],
)
def test_sdpo_rejects_missing_or_malformed_payload(
    token_ids: torch.Tensor,
    teacher: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="payload shape|complete teacher"):
        _compute(
            _batch(token_ids=token_ids, teacher=teacher),
            torch.empty((2, 0), requires_grad=True),
        )


def test_sdpo_sample_keeps_teacher_and_student_topk_shape_contract() -> None:
    student = torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]], requires_grad=True))
    teacher = torch.log(torch.tensor([[0.1, 0.4], [0.2, 0.3]]))
    token_ids = torch.ones((2, 2), dtype=torch.long)

    loss, _, _ = _compute(
        _batch(token_ids=token_ids, teacher=teacher),
        student,
    )

    assert loss is not None and torch.isfinite(loss)


def test_sdpo_accepts_zero_teacher_mass_without_nan() -> None:
    student = torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]], requires_grad=True))
    teacher = torch.tensor(
        [[torch.log(torch.tensor(0.1)), float("-inf")], [float("-inf"), torch.log(torch.tensor(0.3))]]
    )
    token_ids = torch.ones((2, 2), dtype=torch.long)

    loss, _, _ = _compute(
        _batch(token_ids=token_ids, teacher=teacher),
        student,
    )

    assert loss is not None and torch.isfinite(loss)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_sdpo_rejects_invalid_teacher_log_probs(invalid_value: float) -> None:
    student = torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]], requires_grad=True))
    teacher = torch.tensor([[invalid_value, -1.0], [-1.0, -1.0]])
    token_ids = torch.ones((2, 2), dtype=torch.long)

    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        _compute(_batch(token_ids=token_ids, teacher=teacher), student)
