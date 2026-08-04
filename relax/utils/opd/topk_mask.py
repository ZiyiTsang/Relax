# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Reusable masks for truncated Top-K distributions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class OPDDistributionTopKMaskEstimate:
    """Top-K support and the mask aligned with the divergence distribution."""

    topk_mask: torch.Tensor | None
    distribution_mask: torch.Tensor | None


class OPDDistributionTopKMaskEstimator:
    """Build a Top-K support mask and its divergence-column layout."""

    @staticmethod
    def estimate(
        mask: torch.Tensor | None,
        *,
        norm_mode: str,
        include_tail_bucket: bool = False,
    ) -> OPDDistributionTopKMaskEstimate:
        """Return masks for the raw Top-K and divergence supports.

        ``include_tail_bucket`` is explicit because ordinary OPD historically
        excludes the appended tail column when a ragged support mask is
        present, while SDPO includes it in its full-support approximation.
        Keeping that choice at the call site lets both algorithms share the
        layout logic without changing ordinary OPD numerics.
        """
        if mask is None or norm_mode != "tail":
            return OPDDistributionTopKMaskEstimate(topk_mask=mask, distribution_mask=mask)

        if not include_tail_bucket:
            return OPDDistributionTopKMaskEstimate(topk_mask=mask, distribution_mask=mask)

        # SDPO tail support adds one valid bucket after the Top-K columns.
        tail_mask = torch.ones((*mask.shape[:-1], 1), dtype=mask.dtype, device=mask.device)
        distribution_mask = torch.cat([mask, tail_mask], dim=-1)
        return OPDDistributionTopKMaskEstimate(topk_mask=mask, distribution_mask=distribution_mask)


__all__ = ["OPDDistributionTopKMaskEstimate", "OPDDistributionTopKMaskEstimator"]
