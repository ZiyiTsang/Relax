# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Regression tests for OPD aggregation and masked Top-K divergence semantics."""

import importlib
import sys
from argparse import Namespace
from types import ModuleType

import pytest


def _install_fake_megatron(monkeypatch):
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")

    mpu.get_context_parallel_world_size = lambda: 1
    core.mpu = mpu
    megatron.core = core

    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)


@pytest.fixture()
def opd_utils_module(monkeypatch):
    torch = pytest.importorskip("torch", exc_type=ImportError)
    _install_fake_megatron(monkeypatch)
    sys.modules.pop("relax.utils.opd.opd_utils", None)
    module = importlib.import_module("relax.utils.opd.opd_utils")
    yield module, torch
    sys.modules.pop("relax.utils.opd.opd_utils", None)


def test_opd_loss_token_mean(opd_utils_module):
    opd_utils, torch = opd_utils_module
    values = torch.tensor([1.0, 1.0, 10.0, 6.0])
    batch = {
        "total_lengths": [4, 5],
        "response_lengths": [2, 2],
        "loss_masks": [
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 0.0]),
        ],
    }

    # token_mean: (1+1+10+0) / (1+1+1+0) = 12 / 3
    assert torch.isclose(opd_utils.reduce_opd_loss(batch, values), torch.tensor(12.0 / 3.0))


def test_sdpo_valid_mask_excludes_invalid_sample_from_opd_denominator(opd_utils_module):
    opd_utils, torch = opd_utils_module
    values = torch.tensor([1.0, 2.0, 10.0, 20.0])
    batch = {
        "total_lengths": [4, 5],
        "response_lengths": [2, 2],
        "loss_masks": [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])],
        "sdpo_valid": [torch.tensor(True), torch.tensor(False)],
    }

    masked = opd_utils._mask_opd_values_by_sample(
        values,
        batch,
        Namespace(qkv_format="thd"),
        torch.tensor([True, False]),
    )

    assert torch.equal(masked, torch.tensor([1.0, 2.0, 0.0, 0.0]))
    # Invalid SDPO samples are excluded from both OPD numerator and denominator.
    assert torch.isclose(
        opd_utils.reduce_opd_loss(batch, masked, torch.tensor([True, False])),
        torch.tensor(1.5),
    )


def test_sdpo_valid_mask_handles_different_response_lengths(opd_utils_module):
    opd_utils, torch = opd_utils_module
    values = torch.tensor([10.0, 1.0, 2.0, 3.0, 4.0])
    batch = {
        "response_lengths": [1, 4],
        "loss_masks": [torch.ones(1), torch.ones(4)],
    }

    loss = opd_utils.reduce_opd_loss(batch, values, torch.tensor([False, True]))

    assert torch.isclose(loss, torch.tensor(2.5))


def test_sdpo_valid_mask_all_invalid_is_differentiable_zero(opd_utils_module):
    opd_utils, torch = opd_utils_module
    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    batch = {
        "response_lengths": [1, 2],
        "loss_masks": [torch.ones(1), torch.ones(2)],
    }

    loss = opd_utils.reduce_opd_loss(batch, values, torch.tensor([False, False]))

    assert torch.equal(loss, torch.zeros_like(loss))
    loss.backward()
    assert torch.equal(values.grad, torch.zeros_like(values))


def test_compute_policy_opd_loss_uses_sdpo_valid_for_denominator(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_topk",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
        qkv_format="thd",
    )
    batch = {
        "response_lengths": [1, 2],
        "total_lengths": [1, 2],
        "loss_masks": [torch.ones(1), torch.ones(2)],
        "sdpo_valid": [True, False],
        "opd_topk_teacher_log_probs": [
            torch.log(torch.tensor([[0.1, 0.4]])),
            torch.log(torch.tensor([[0.4, 0.1], [0.1, 0.4]])),
        ],
    }
    student_topk = [
        torch.log(torch.tensor([[0.2, 0.3]], requires_grad=True)),
        torch.log(torch.tensor([[0.3, 0.2], [0.2, 0.3]], requires_grad=True)),
    ]

    loss, metrics = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=torch.zeros(3),
        old_log_probs=torch.zeros(3),
        log_probs_and_entropy={"topk_log_probs": student_topk},
    )

    expected_values = torch.cat(
        [
            opd_utils.compute_sdpo_topk_loss(
                student_topk[0],
                batch["opd_topk_teacher_log_probs"][0],
                kl_type="reverse_kl",
                jsd_alpha=0.5,
                norm_mode="tail",
            ),
            opd_utils.compute_sdpo_topk_loss(
                student_topk[1],
                batch["opd_topk_teacher_log_probs"][1],
                kl_type="reverse_kl",
                jsd_alpha=0.5,
                norm_mode="tail",
            ),
        ]
    )
    expected = opd_utils.reduce_opd_loss(batch, expected_values, torch.tensor([True, False]))

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["opd_kl"], expected.detach())


@pytest.mark.parametrize(
    ("calculate_per_token_loss", "expected_scale"),
    [(False, 0.25), (True, 1.0)],
)
@pytest.mark.parametrize(("dynamic_cp_rank", "valid_rows"), [(0, 1), (1, 3)])
def test_sdpo_valid_mask_uses_cp_local_reducer(
    opd_utils_module,
    calculate_per_token_loss,
    expected_scale,
    dynamic_cp_rank,
    valid_rows,
):
    opd_utils, torch = opd_utils_module
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_topk",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
        calculate_per_token_loss=calculate_per_token_loss,
        qkv_format="thd",
    )
    batch = {
        "response_lengths": [4, 4],
        "total_lengths": [8, 8],
        "loss_masks": [torch.ones(4), torch.ones(4)],
        "sdpo_valid": [False, True],
        "dynamic_cp_size": 2,
        "dynamic_cp_rank": dynamic_cp_rank,
        "opd_topk_teacher_log_probs": [
            torch.log(torch.tensor([[0.1, 0.4]]).repeat(valid_rows, 1)),
            torch.log(torch.tensor([[0.1, 0.4]]).repeat(valid_rows, 1)),
        ],
    }
    student_topk = [
        torch.log(torch.tensor([[0.4, 0.1]]).repeat(valid_rows, 1)),
        torch.log(torch.tensor([[0.2, 0.3]]).repeat(valid_rows, 1)),
    ]

    loss, metrics = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=torch.zeros(2 * valid_rows),
        old_log_probs=torch.zeros(2 * valid_rows),
        log_probs_and_entropy={"topk_log_probs": student_topk},
    )

    valid_values = opd_utils.compute_sdpo_topk_loss(
        student_topk[1],
        batch["opd_topk_teacher_log_probs"][1],
        kl_type="reverse_kl",
        jsd_alpha=0.5,
        norm_mode="tail",
    )
    expected = valid_values.sum() * expected_scale

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["opd_kl"], expected.detach())


def test_union_jsd_tail_includes_tail_bin(opd_utils_module):
    opd_utils, torch = opd_utils_module
    student = torch.log(torch.tensor([[0.2, 0.3, 0.0]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4, 0.0]]))
    mask = torch.tensor([[True, True, False]])

    masked = opd_utils.compute_opd_kl_topk(
        student,
        teacher,
        kl_type="jsd",
        norm_mode="tail",
        mask=mask,
    )
    fixed = opd_utils.compute_opd_kl_topk(
        student[:, :2],
        teacher[:, :2],
        kl_type="jsd",
        norm_mode="tail",
    )
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.1, 0.4, 0.5])
    midpoint = (p + q) / 2
    expected = 0.5 * (p * torch.log(p / midpoint)).sum() + 0.5 * (q * torch.log(q / midpoint)).sum()

    assert torch.allclose(masked, fixed)
    assert torch.allclose(fixed, expected.reshape(1))


def test_sdpo_topk_reverse_kl_uses_student_probability_weight(opd_utils_module):
    opd_utils, torch = opd_utils_module
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))

    loss = opd_utils.compute_sdpo_topk_loss(
        student,
        teacher,
        kl_type="reverse_kl",
        jsd_alpha=0.5,
        norm_mode="tail",
    )
    expected = (
        torch.tensor([0.2, 0.3, 0.5])
        * torch.log(torch.tensor([0.2, 0.3, 0.5]) / torch.tensor([0.1, 0.4, 0.5]))
    ).sum()

    assert torch.allclose(loss, expected.reshape(1))


def test_sdpo_topk_forward_kl_and_jsd_match_reference_alpha_convention(opd_utils_module):
    opd_utils, torch = opd_utils_module
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.1, 0.4, 0.5])
    midpoint = (p + q) / 2

    forward = opd_utils.compute_sdpo_topk_loss(
        student,
        teacher,
        kl_type="forward_kl",
        jsd_alpha=0.5,
        norm_mode="tail",
    )
    reverse = opd_utils.compute_sdpo_topk_loss(
        student,
        teacher,
        kl_type="reverse_kl",
        jsd_alpha=0.5,
        norm_mode="tail",
    )
    jsd = opd_utils.compute_sdpo_topk_loss(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.5,
        norm_mode="tail",
    )
    expected_forward = (q * torch.log(q / p)).sum()
    expected_reverse = (p * torch.log(p / q)).sum()
    expected_jsd = (
        0.5 * (p * torch.log(p / midpoint)).sum()
        + 0.5 * (q * torch.log(q / midpoint)).sum()
    )

    assert torch.allclose(forward, expected_forward.reshape(1))
    assert torch.allclose(reverse, expected_reverse.reshape(1))
    assert torch.allclose(jsd, expected_jsd.reshape(1))

    alpha_zero = opd_utils.compute_sdpo_topk_loss(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.0,
        norm_mode="tail",
    )
    alpha_one = opd_utils.compute_sdpo_topk_loss(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=1.0,
        norm_mode="tail",
    )
    assert torch.allclose(alpha_zero, expected_forward.reshape(1))
    assert torch.allclose(alpha_one, expected_reverse.reshape(1))


def test_sdpo_topk_loss_detaches_teacher_and_keeps_student_gradient(opd_utils_module):
    opd_utils, torch = opd_utils_module
    student = torch.log(torch.tensor([[0.2, 0.3]], requires_grad=True))
    student.retain_grad()
    teacher = torch.log(torch.tensor([[0.1, 0.4]], requires_grad=True))
    teacher.retain_grad()

    loss = opd_utils.compute_sdpo_topk_loss(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.5,
        norm_mode="tail",
    ).sum()
    loss.backward()

    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_sdpo_policy_loss_with_no_valid_sample_is_differentiable_zero(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="reverse_kl",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_topk",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
        qkv_format="thd",
    )
    log_probs = torch.tensor([-0.7, -0.8], requires_grad=True)
    batch = {
        "response_lengths": [2],
        "total_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "sdpo_valid": [False],
        "opd_topk_teacher_log_probs": [torch.log(torch.tensor([[0.1, 0.4], [0.2, 0.3]]))],
    }
    student_topk = torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]], requires_grad=True))
    student_topk.retain_grad()

    loss, metrics = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=log_probs,
        old_log_probs=log_probs.detach(),
        log_probs_and_entropy={"topk_log_probs": [student_topk]},
    )

    assert loss is not None
    assert torch.equal(loss, torch.zeros_like(loss))
    assert loss.requires_grad
    loss.backward()
    assert student_topk.grad is not None
    assert torch.equal(student_topk.grad, torch.zeros_like(student_topk))
    assert metrics["sdpo_valid_ratio"].item() == 0.0
    assert metrics["opd_kl"].item() == 0.0


def test_sdpo_policy_loss_rejects_topk_rows_misaligned_with_response(opd_utils_module):
    opd_utils, torch = opd_utils_module
    args = Namespace(
        opd_loss_coef=1.0,
        opd_kl_type="jsd",
        opd_jsd_alpha=0.5,
        opd_norm_mode="tail",
        opd_token_selection="student_topk",
        opd_log_prob_min_clamp=None,
        opd_per_token_clip=None,
        opd_is_clip=None,
        qkv_format="thd",
    )
    batch = {
        "response_lengths": [2, 3],
        "total_lengths": [4, 5],
        "loss_masks": [torch.ones(2), torch.ones(3)],
        "opd_topk_teacher_log_probs": [
            torch.log(torch.full((2, 2), 0.25)),
            torch.log(torch.full((2, 2), 0.25)),
        ],
    }
    student_topk = [
        torch.log(torch.full((2, 2), 0.25)),
        torch.log(torch.full((3, 2), 0.25)),
    ]

    with pytest.raises(ValueError, match="OPD teacher top-k row mismatch"):
        opd_utils.compute_policy_opd_loss(
            args=args,
            batch=batch,
            log_probs=torch.zeros(5),
            old_log_probs=torch.zeros(5),
            log_probs_and_entropy={"topk_log_probs": student_topk},
        )
