# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Unit tests for reusable Top-K mask estimation."""

import pytest
import torch

from relax.utils.opd.topk_mask import OPDDistributionTopKMaskEstimator  # noqa: E402


def test_topk_mask_estimator_preserves_legacy_mask_layout_by_default() -> None:
    mask = torch.tensor([[True, False, True]])

    estimate = OPDDistributionTopKMaskEstimator.estimate(mask, norm_mode="tail")

    assert estimate.topk_mask is mask
    assert torch.equal(estimate.distribution_mask, mask)


def test_topk_mask_estimator_can_add_an_explicit_tail_bucket() -> None:
    mask = torch.tensor([[True, False, True]])

    estimate = OPDDistributionTopKMaskEstimator.estimate(
        mask,
        norm_mode="tail",
        include_tail_bucket=True,
    )

    assert estimate.topk_mask is mask
    assert torch.equal(estimate.distribution_mask, torch.tensor([[True, False, True, True]]))


@pytest.mark.parametrize("norm_mode", ["norm", "trunc"])
def test_topk_mask_estimator_keeps_shape_without_tail(norm_mode: str) -> None:
    mask = torch.tensor([[True, False]])

    estimate = OPDDistributionTopKMaskEstimator.estimate(mask, norm_mode=norm_mode)

    assert estimate.topk_mask is mask
    assert estimate.distribution_mask is mask


def test_topk_mask_estimator_handles_fixed_topk_without_mask() -> None:
    estimate = OPDDistributionTopKMaskEstimator.estimate(None, norm_mode="tail")

    assert estimate.topk_mask is None
    assert estimate.distribution_mask is None


def test_topk_mask_estimator_does_not_mutate_support_mask() -> None:
    mask = torch.tensor([[True, False], [False, True]])
    original = mask.clone()

    estimate = OPDDistributionTopKMaskEstimator.estimate(mask, norm_mode="tail", include_tail_bucket=True)

    assert torch.equal(mask, original)
    assert estimate.distribution_mask.shape == (2, 3)
    assert torch.equal(estimate.distribution_mask[:, -1], torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (2, 3, 4)])
def test_topk_mask_estimator_handles_empty_and_arbitrary_leading_dimensions(shape: tuple[int, ...]) -> None:
    mask = torch.zeros(shape, dtype=torch.bool)

    estimate = OPDDistributionTopKMaskEstimator.estimate(
        mask,
        norm_mode="tail",
        include_tail_bucket=True,
    )

    assert estimate.distribution_mask.shape == (*shape[:-1], shape[-1] + 1)
    assert torch.equal(estimate.topk_mask, mask)


def test_topk_mask_estimator_preserves_noncontiguous_support_layout() -> None:
    mask = torch.tensor([[True, False], [False, True], [True, True]])
    noncontiguous = mask.t()

    estimate = OPDDistributionTopKMaskEstimator.estimate(
        noncontiguous,
        norm_mode="tail",
        include_tail_bucket=True,
    )

    assert not noncontiguous.is_contiguous()
    assert torch.equal(estimate.topk_mask, noncontiguous)
    assert torch.equal(estimate.distribution_mask[..., -1], torch.ones(2, dtype=torch.bool))
