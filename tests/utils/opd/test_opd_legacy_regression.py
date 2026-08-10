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


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_ordinary_opd_keeps_upstream_reducer_with_production_reducer(
    calculate_per_token_loss: bool,
) -> None:
    args = _sampled_loss_args()
    args.calculate_per_token_loss = calculate_per_token_loss
    student = torch.tensor([-0.7, -0.8, -0.3], requires_grad=True)
    teacher = torch.tensor([-0.4, -0.5, -0.2])
    batch = _sampled_loss_batch()
    batch["teacher_log_probs"] = [teacher[:2], teacher[2:]]

    def production_reducer(values: torch.Tensor) -> torch.Tensor:
        return values.sum() + 17.0

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=student,
        old_log_probs=student.detach(),
        log_probs_and_entropy={},
        sum_of_sample_mean=production_reducer,
    )

    expected_values = torch.cat([student[:2] - teacher[:2], student[2:] - teacher[2:]])
    expected = expected_values.sum() / expected_values.numel()
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize("kl_type, jsd_alpha", [("reverse_kl", 0.0), ("forward_kl", 1.0)])
def test_topk_endpoint_aliases_match_upstream_oracle(kl_type: str, jsd_alpha: float) -> None:
    student = torch.log(torch.tensor([[0.2, 0.3], [0.4, 0.1]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4], [0.3, 0.2]]))

    actual = opd_utils.compute_opd_kl_topk(
        student,
        teacher,
        kl_type=kl_type,
        norm_mode="norm",
    )
    expected = _legacy_oracle(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=jsd_alpha,
        norm_mode="norm",
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("jsd_alpha", [0.0, 1.0])
def test_sdpo_policy_loss_uses_upstream_endpoint_direction(jsd_alpha: float) -> None:
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="jsd",
        opd_jsd_alpha=jsd_alpha,
        opd_norm_mode="norm",
        opd_token_selection="student_topk",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
    )
    student = torch.log(torch.tensor([[0.2, 0.3]])).detach().requires_grad_()
    teacher = torch.log(torch.tensor([[0.1, 0.4]])).detach().requires_grad_()
    batch = {
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
        "opd_sample_mask": [True],
        "opd_topk_teacher_log_probs": [teacher],
    }

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=torch.zeros(1),
        old_log_probs=torch.zeros(1),
        log_probs_and_entropy={"topk_log_probs": [student]},
    )
    expected = _legacy_oracle(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=jsd_alpha,
        norm_mode="norm",
    ).mean()

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None


def _sampled_loss_args() -> Namespace:
    return Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_sampled",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
    )


def _sampled_loss_batch(sample_mask: list[bool] | None = None) -> dict:
    batch = {
        "response_lengths": [2, 1],
        "loss_masks": [torch.ones(2), torch.ones(1)],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
        "teacher_log_probs": [torch.tensor([-0.4, -0.5]), torch.tensor([-0.2])],
    }
    if sample_mask is not None:
        batch["opd_sample_mask"] = sample_mask
    return batch


def test_opd_sample_mask_keeps_all_one_loss_and_gradient_unchanged() -> None:
    args = _sampled_loss_args()
    student_plain = torch.tensor([-0.7, -0.8, -0.3], requires_grad=True)
    student_masked = student_plain.detach().clone().requires_grad_(True)

    plain_loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=_sampled_loss_batch(),
        log_probs=student_plain,
        old_log_probs=student_plain.detach(),
        log_probs_and_entropy={},
    )
    masked_loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=_sampled_loss_batch([True, True]),
        log_probs=student_masked,
        old_log_probs=student_masked.detach(),
        log_probs_and_entropy={},
    )

    torch.testing.assert_close(masked_loss, plain_loss)
    plain_loss.backward()
    masked_loss.backward()
    torch.testing.assert_close(student_masked.grad, student_plain.grad)


def test_opd_sample_mask_removes_unprivileged_sample_from_loss_and_gradient() -> None:
    args = _sampled_loss_args()
    student = torch.tensor([-0.7, -0.8, -0.3], requires_grad=True)

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=_sampled_loss_batch([True, False]),
        log_probs=student,
        old_log_probs=student.detach(),
        log_probs_and_entropy={},
    )

    expected = torch.tensor(((-0.7 + 0.4) + (-0.8 + 0.5)) / 2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(student.grad, torch.tensor([0.5, 0.5, 0.0]))


def test_opd_sample_mask_all_false_has_zero_loss_and_gradient() -> None:
    args = _sampled_loss_args()
    student = torch.tensor([-0.7, -0.8, -0.3], requires_grad=True)

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=_sampled_loss_batch([False, False]),
        log_probs=student,
        old_log_probs=student.detach(),
        log_probs_and_entropy={},
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student))


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_opd_sample_mask_uses_active_denominator_with_production_reducer(
    calculate_per_token_loss: bool,
) -> None:
    args = _sampled_loss_args()
    args.calculate_per_token_loss = calculate_per_token_loss
    student = torch.tensor([-0.7, -0.8, -0.3], requires_grad=True)
    teacher = torch.tensor([-0.4, -0.5, -0.2])
    batch = _sampled_loss_batch([True, False])
    batch["teacher_log_probs"] = [teacher[:2], teacher[2:]]

    def production_reducer(values: torch.Tensor) -> torch.Tensor:
        chunks = values.split(batch["response_lengths"], dim=0)
        if calculate_per_token_loss:
            return sum(chunk.sum() for chunk in chunks)
        return sum(chunk.mean() for chunk in chunks)

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=student,
        old_log_probs=student.detach(),
        log_probs_and_entropy={},
        sum_of_sample_mean=production_reducer,
    )

    active_sum = torch.tensor(-0.6 if calculate_per_token_loss else -0.3)
    full_denominator = torch.tensor(3.0 if calculate_per_token_loss else 2.0)
    active_denominator = torch.tensor(2.0 if calculate_per_token_loss else 1.0)
    expected = active_sum * full_denominator / active_denominator
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected_gradient = 1.5 if calculate_per_token_loss else 1.0
    torch.testing.assert_close(student.grad, torch.tensor([expected_gradient, expected_gradient, 0.0]))


@pytest.mark.parametrize("calculate_per_token_loss", [False, True])
def test_opd_sample_mask_prefers_step_level_denominator_scale(calculate_per_token_loss: bool) -> None:
    args = _sampled_loss_args()
    args.calculate_per_token_loss = calculate_per_token_loss
    student = torch.tensor([-0.7, -0.8, -0.3], requires_grad=True)
    teacher = torch.tensor([-0.4, -0.5, -0.2])
    batch = _sampled_loss_batch([True, False])
    batch[opd_utils.OPD_SAMPLE_MASK_DENOMINATOR_SCALE] = [torch.tensor(7.0), torch.tensor(7.0)]
    batch["teacher_log_probs"] = [teacher[:2], teacher[2:]]

    def production_reducer(values: torch.Tensor) -> torch.Tensor:
        chunks = values.split(batch["response_lengths"], dim=0)
        if calculate_per_token_loss:
            return sum(chunk.sum() for chunk in chunks)
        return sum(chunk.mean() for chunk in chunks)

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=student,
        old_log_probs=student.detach(),
        log_probs_and_entropy={},
        sum_of_sample_mean=production_reducer,
    )

    active_sum = torch.tensor(-0.6 if calculate_per_token_loss else -0.3)
    torch.testing.assert_close(loss, active_sum * 7)


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


def test_opd_sample_mask_cp_reducer_masks_response_rows() -> None:
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
    student = [torch.tensor([[-1.2, -1.0]], requires_grad=True)]
    teacher = [torch.tensor([[-0.9, -0.8]])]
    batch = {
        "total_lengths": [8],
        "response_lengths": [4],
        "loss_masks": [torch.ones(4)],
        "dynamic_cp_size": 2,
        "dynamic_cp_rank": 0,
        "opd_sample_mask": [False],
        "opd_topk_teacher_log_probs": teacher,
    }

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=torch.zeros(1),
        old_log_probs=torch.zeros(1),
        log_probs_and_entropy={"topk_log_probs": student},
    )

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    loss.backward()
    torch.testing.assert_close(student[0].grad, torch.zeros_like(student[0]))
