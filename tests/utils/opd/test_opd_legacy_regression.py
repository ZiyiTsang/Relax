# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Independent numerical oracles for the unchanged ordinary OPD path."""

from argparse import Namespace

import pytest
import torch

from relax.utils.opd import opd_utils  # noqa: E402


def _legacy_oracle(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    kl_type: str,
    jsd_alpha: float = 0.5,
    norm_mode: str = "tail",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """A copy-independent oracle of upstream/main's OPD equations."""

    student = student.float()
    teacher = teacher.float()
    if mask is not None:
        student = student.masked_fill(~mask, float("-inf"))
        teacher = teacher.masked_fill(~mask, float("-inf"))

    if kl_type == "reverse_kl":
        return student - teacher
    if kl_type == "low_var_kl":
        log_ratio = student - teacher
        return (torch.exp(-log_ratio) - 1.0 + log_ratio).sum(dim=-1)
    if kl_type != "jsd":
        raise ValueError(kl_type)

    def add_tail(values: torch.Tensor) -> torch.Tensor:
        mass = torch.logsumexp(values, dim=-1, keepdim=True).clamp(max=-1e-7)
        return torch.cat([values, torch.log(-torch.expm1(mass))], dim=-1)

    if norm_mode == "tail":
        student = add_tail(student)
        teacher = add_tail(teacher)
    elif norm_mode == "norm":
        student = student - torch.logsumexp(student, dim=-1, keepdim=True)
        teacher = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
    elif norm_mode != "trunc":
        raise ValueError(norm_mode)

    if jsd_alpha == 0.0:
        values = student.exp() * (student - teacher)
        if mask is not None:
            values = values[..., : mask.size(-1)].masked_fill(~mask, 0.0)
        return values.sum(dim=-1)
    if jsd_alpha == 1.0:
        values = teacher.exp() * (teacher - student)
        if mask is not None:
            values = values[..., : mask.size(-1)].masked_fill(~mask, 0.0)
        return values.sum(dim=-1)

    mixture = torch.logsumexp(
        torch.stack(
            [
                student + torch.log(torch.tensor(1.0 - jsd_alpha)),
                teacher + torch.log(torch.tensor(jsd_alpha)),
            ]
        ),
        dim=0,
    )
    student_kl = student.exp() * (student - mixture)
    teacher_kl = teacher.exp() * (teacher - mixture)
    if mask is not None:
        student_kl = student_kl[..., : mask.size(-1)].masked_fill(~mask, 0.0)
        teacher_kl = teacher_kl[..., : mask.size(-1)].masked_fill(~mask, 0.0)
    return (1.0 - jsd_alpha) * student_kl.sum(-1) + jsd_alpha * teacher_kl.sum(-1)


@pytest.mark.parametrize("norm_mode", ["tail", "norm", "trunc"])
@pytest.mark.parametrize("jsd_alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_ordinary_opd_matches_independent_upstream_oracle(
    norm_mode: str,
    jsd_alpha: float,
) -> None:
    student = torch.log(torch.tensor([[0.2, 0.1, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.2, 0.4]]))
    support_mask = torch.tensor([[True, False, True]])

    actual = opd_utils.compute_opd_kl(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=jsd_alpha,
        norm_mode=norm_mode,
        mask=support_mask,
    )
    expected = _legacy_oracle(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=jsd_alpha,
        norm_mode=norm_mode,
        mask=support_mask,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_ordinary_low_var_kl_matches_independent_upstream_oracle() -> None:
    student = torch.tensor([[-0.7, -0.8]])
    teacher = torch.tensor([[-0.4, -0.5]])

    actual = opd_utils.compute_opd_kl(student, teacher, kl_type="low_var_kl")
    expected = _legacy_oracle(student, teacher, kl_type="low_var_kl")

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_ordinary_sampled_reverse_kl_matches_independent_upstream_oracle() -> None:
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_sampled",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
    )
    student = torch.tensor([-0.7, -0.8], requires_grad=True)
    teacher = torch.tensor([-0.4, -0.5])
    batch = {
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
        "teacher_log_probs": [teacher],
    }

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=student,
        old_log_probs=student.detach(),
        log_probs_and_entropy={},
    )

    expected = _legacy_oracle(student, teacher, kl_type="reverse_kl").mean()
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.allclose(student.grad, torch.full_like(student, 0.5))


def test_ordinary_opd_ignores_unrelated_batch_metadata() -> None:
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_topk",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
    )
    student = [torch.log(torch.tensor([[0.2, 0.3]]))]
    teacher = [torch.log(torch.tensor([[0.1, 0.4]]))]
    common = {
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
        "opd_topk_teacher_log_probs": teacher,
    }
    plain_loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=common,
        log_probs=torch.zeros(1),
        old_log_probs=torch.zeros(1),
        log_probs_and_entropy={"topk_log_probs": student},
    )
    common_with_metadata = dict(common, unrelated_metadata=[False])
    metadata_loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=common_with_metadata,
        log_probs=torch.zeros(1),
        old_log_probs=torch.zeros(1),
        log_probs_and_entropy={"topk_log_probs": student},
    )

    assert torch.equal(plain_loss, metadata_loss)


def test_ordinary_cp_reducer_preserves_sample_mean_mode() -> None:
    pytest.importorskip("megatron.core")
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_topk",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        context_parallel_size=2,
    )
    teacher = [torch.log(torch.tensor([[0.1, 0.4], [0.2, 0.3]]))]
    student = [torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]]))]
    batch = {
        "total_lengths": [7],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "dynamic_cp_size": 2,
        "dynamic_cp_rank": 1,
        "opd_topk_teacher_log_probs": teacher,
    }

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=torch.zeros(2),
        old_log_probs=torch.zeros(2),
        log_probs_and_entropy={"topk_log_probs": student},
    )
    expected_per_token = opd_utils.compute_opd_kl_topk(student[0], teacher[0], kl_type="reverse_kl")

    assert torch.allclose(loss, expected_per_token.mean())


def test_opd_train_data_schema_does_not_duplicate_rollout_log_probs():
    args = Namespace(
        use_opd=True,
        opd_type="sglang",
        opd_loss_mode="opd",
        opd_token_selection="student_sampled",
        opd_kl_coef=1.0,
        opd_loss_coef=0.0,
    )
    fields = ["rollout_log_probs"]

    opd_utils.consume_opd_train_data(fields, args)

    assert fields.count("rollout_log_probs") == 1
    assert fields.count("teacher_log_probs") == 1
