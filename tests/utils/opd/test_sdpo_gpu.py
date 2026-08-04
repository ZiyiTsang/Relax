# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Single-device CUDA smoke tests; TP/CP acceptance remains a distributed
run."""

import pytest
import torch

from relax.utils.opd.sdpo.loss import compute_sdpo_topk_divergence  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the SDPO GPU smoke test")
def test_sdpo_jsd_cuda_keeps_student_gradient_and_teacher_detached() -> None:
    student = torch.log(torch.tensor([[0.2, 0.3]], device="cuda", requires_grad=True))
    teacher = torch.log(torch.tensor([[0.1, 0.4]], device="cuda", requires_grad=True))
    student.retain_grad()
    teacher.retain_grad()

    loss = compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.5,
        norm_mode="tail",
    ).sum()
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
