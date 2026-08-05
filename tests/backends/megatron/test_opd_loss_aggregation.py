# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Focused regression tests for legacy OPD and the isolated SDPO criterion."""

import importlib
from argparse import Namespace

import pytest
import torch


@pytest.fixture()
def opd_modules():
    opd_utils = importlib.import_module("relax.utils.opd.opd_utils")
    sdpo_loss = importlib.import_module("relax.utils.opd.sdpo.loss")
    yield opd_utils, sdpo_loss, torch


def _legacy_args(**overrides) -> Namespace:
    values = {
        "opd_loss_coef": 1.0,
        "opd_kl_type": "reverse_kl",
        "opd_jsd_alpha": 0.5,
        "opd_norm_mode": "tail",
        "opd_token_selection": "student_topk",
        "opd_log_prob_min_clamp": None,
        "opd_per_token_clip": None,
        "opd_is_clip": None,
        "calculate_per_token_loss": True,
        "qkv_format": "thd",
    }
    values.update(overrides)
    return Namespace(**values)


def _batch(response_lengths: list[int], *, topk_teacher=None, topk_ids=None) -> dict:
    return {
        "total_lengths": [length + 2 for length in response_lengths],
        "response_lengths": response_lengths,
        "loss_masks": [torch.ones(length) for length in response_lengths],
        "dynamic_cp_size": 1,
        "dynamic_cp_rank": 0,
        **({"opd_topk_teacher_log_probs": topk_teacher} if topk_teacher is not None else {}),
        **({"opd_topk_token_ids": topk_ids} if topk_ids is not None else {}),
    }


def _import_megatron_loss():
    pytest.importorskip("megatron.core")
    return importlib.import_module("relax.backends.megatron.loss")


def test_legacy_opd_reducer_keeps_token_mean(opd_modules):
    opd_utils, _, torch = opd_modules
    values = torch.tensor([1.0, 1.0, 10.0, 6.0])
    batch = {
        "response_lengths": [2, 2],
        "loss_masks": [torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])],
    }

    assert torch.isclose(opd_utils.reduce_opd_loss(batch, values), torch.tensor(12.0 / 3.0))


def test_legacy_topk_opd_does_not_use_sdpo_criterion(opd_modules):
    opd_utils, sdpo_loss, torch = opd_modules
    args = _legacy_args(opd_kl_type="reverse_kl")
    teacher = [torch.log(torch.tensor([[0.1, 0.4], [0.2, 0.3]]))]
    student = [torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]], requires_grad=True))]
    batch = _batch([2], topk_teacher=teacher)

    loss, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=torch.zeros(2),
        old_log_probs=torch.zeros(2),
        log_probs_and_entropy={"topk_log_probs": student},
    )
    expected_values = opd_utils.compute_opd_kl_topk(student[0], teacher[0], kl_type="reverse_kl", norm_mode="tail")
    expected = opd_utils.reduce_opd_loss(batch, expected_values)

    assert torch.allclose(loss, expected)
    assert sdpo_loss.compute_sdpo_topk_divergence is not opd_utils.compute_opd_kl_topk


def test_legacy_topk_tail_keeps_legacy_mask_layout_with_union_mask(opd_modules):
    opd_utils, _, torch = opd_modules
    support_mask = torch.tensor([[True, False]])
    student = torch.log(torch.tensor([[0.2, 0.1]]))
    teacher = torch.log(torch.tensor([[0.1, 0.2]]))

    value = opd_utils.compute_opd_kl_topk(
        student,
        teacher,
        kl_type="reverse_kl",
        norm_mode="tail",
        mask=support_mask,
    )
    expected = 0.2 * torch.log(torch.tensor(2.0))
    expected = expected.reshape(1)

    assert torch.allclose(value, expected)


def test_sdpo_jsd_matches_symmetric_definition_and_detaches_teacher(opd_modules):
    _, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]], requires_grad=True))
    teacher = torch.log(torch.tensor([[0.1, 0.4]], requires_grad=True))
    student.retain_grad()
    teacher.retain_grad()

    value = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.5,
        norm_mode="tail",
    )
    p = torch.tensor([0.2, 0.3, 0.5])
    q = torch.tensor([0.1, 0.4, 0.5])
    midpoint = (p + q) / 2
    expected = 0.5 * (p * torch.log(p / midpoint)).sum() + 0.5 * (q * torch.log(q / midpoint)).sum()

    assert torch.allclose(value, expected.reshape(1))
    value.sum().backward()
    assert student.grad is not None
    assert teacher.grad is None


@pytest.mark.parametrize("kl_type", ["forward_kl", "reverse_kl"])
def test_sdpo_kl_direction_matches_independent_oracle(opd_modules, kl_type):
    _, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))
    student_distribution = torch.tensor([0.2, 0.3, 0.5])
    teacher_distribution = torch.tensor([0.1, 0.4, 0.5])

    actual = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type=kl_type,
        jsd_alpha=0.5,
        norm_mode="tail",
    )
    if kl_type == "forward_kl":
        expected = (teacher_distribution * (teacher_distribution / student_distribution).log()).sum()
    else:
        expected = (student_distribution * (student_distribution / teacher_distribution).log()).sum()

    assert torch.allclose(actual, expected.reshape(1))


@pytest.mark.parametrize(
    ("alpha", "kl_type"),
    [(0.0, "forward_kl"), (1.0, "reverse_kl")],
)
def test_sdpo_jsd_alpha_boundaries_match_reference_kl_direction(opd_modules, alpha, kl_type):
    _, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]], requires_grad=True))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))

    jsd_boundary = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=alpha,
        norm_mode="tail",
    )
    reference = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type=kl_type,
        jsd_alpha=0.5,
        norm_mode="tail",
    )

    assert torch.allclose(jsd_boundary, reference)


def test_ordinary_and_sdpo_jsd_endpoint_conventions_are_explicit(opd_modules):
    opd_utils, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))
    student_distribution = torch.tensor([0.2, 0.3, 0.5])
    teacher_distribution = torch.tensor([0.1, 0.4, 0.5])
    reverse_kl = (student_distribution * (student_distribution / teacher_distribution).log()).sum()
    forward_kl = (teacher_distribution * (teacher_distribution / student_distribution).log()).sum()

    ordinary_alpha_zero = opd_utils.compute_opd_kl_topk(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.0,
        norm_mode="tail",
    )
    ordinary_alpha_one = opd_utils.compute_opd_kl_topk(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=1.0,
        norm_mode="tail",
    )
    sdpo_alpha_zero = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.0,
        norm_mode="tail",
    )
    sdpo_alpha_one = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=1.0,
        norm_mode="tail",
    )

    assert torch.allclose(ordinary_alpha_zero, reverse_kl.reshape(1))
    assert torch.allclose(ordinary_alpha_one, forward_kl.reshape(1))
    assert torch.allclose(sdpo_alpha_zero, forward_kl.reshape(1))
    assert torch.allclose(sdpo_alpha_one, reverse_kl.reshape(1))
    assert not torch.allclose(ordinary_alpha_zero, sdpo_alpha_zero)


def test_sdpo_criterion_validates_kl_type_and_jsd_alpha(opd_modules):
    _, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))

    with pytest.raises(ValueError, match="Unknown SDPO KL type"):
        sdpo_loss.compute_sdpo_topk_divergence(
            student,
            teacher,
            kl_type="low_var_kl",
            jsd_alpha=0.5,
        )
    with pytest.raises(ValueError, match=r"jsd_alpha must be in \[0, 1\]"):
        sdpo_loss.compute_sdpo_topk_divergence(
            student,
            teacher,
            kl_type="jsd",
            jsd_alpha=1.1,
        )


def test_sdpo_support_mask_uses_one_tail_bucket(opd_modules):
    _, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))
    support_mask = torch.tensor([[True, False]])

    value = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="reverse_kl",
        jsd_alpha=0.5,
        norm_mode="tail",
        support_mask=support_mask,
    )
    expected = 0.2 * torch.log(torch.tensor(0.2 / 0.1)) + 0.8 * torch.log(torch.tensor(0.8 / 0.9))

    assert torch.allclose(value, expected.reshape(1))


def test_sdpo_jsd_masks_support_columns_but_keeps_tail(opd_modules):
    _, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]], requires_grad=True))
    teacher = torch.log(torch.tensor([[0.1, 0.4]], requires_grad=True))
    student.retain_grad()
    teacher.retain_grad()
    support_mask = torch.tensor([[True, False]])

    value = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.5,
        norm_mode="tail",
        support_mask=support_mask,
    )
    p = torch.tensor([0.2, 0.8])
    q = torch.tensor([0.1, 0.9])
    midpoint = (p + q) / 2
    expected = 0.5 * (p * torch.log(p / midpoint)).sum() + 0.5 * (q * torch.log(q / midpoint)).sum()

    assert torch.allclose(value, expected.reshape(1))
    value.sum().backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_sdpo_norm_support_mask_keeps_topk_shape(opd_modules):
    _, sdpo_loss, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))
    support_mask = torch.tensor([[True, False]])

    value = sdpo_loss.compute_sdpo_topk_divergence(
        student,
        teacher,
        kl_type="jsd",
        jsd_alpha=0.5,
        norm_mode="norm",
        support_mask=support_mask,
    )

    assert torch.allclose(value, torch.zeros(1))


def test_legacy_fixed_topk_loss_matches_concatenated_pre_refactor_path(opd_modules):
    opd_utils, _, torch = opd_modules
    args = _legacy_args(opd_kl_type="reverse_kl")
    student = [
        torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]], requires_grad=True)),
        torch.log(torch.tensor([[0.4, 0.1]], requires_grad=True)),
    ]
    teacher = [
        torch.log(torch.tensor([[0.1, 0.4], [0.2, 0.3]])),
        torch.log(torch.tensor([[0.3, 0.2]])),
    ]
    batch = _batch([2, 1], topk_teacher=teacher)
    batch["loss_masks"][1] = torch.zeros(1)
    log_probs = torch.zeros(3)

    actual, _ = opd_utils.compute_policy_opd_loss(
        args=args,
        batch=batch,
        log_probs=log_probs,
        old_log_probs=log_probs,
        log_probs_and_entropy={"topk_log_probs": student},
    )
    reference_values = torch.cat(
        [
            opd_utils.compute_opd_kl_topk(student[0], teacher[0], kl_type="reverse_kl"),
            opd_utils.compute_opd_kl_topk(student[1], teacher[1], kl_type="reverse_kl"),
        ]
    )
    reference = opd_utils.reduce_opd_loss(batch, reference_values)

    assert torch.allclose(actual, reference)


def test_legacy_topk_kl_names_keep_upstream_direction(opd_modules):
    opd_utils, _, torch = opd_modules
    student = torch.log(torch.tensor([[0.2, 0.3]]))
    teacher = torch.log(torch.tensor([[0.1, 0.4]]))
    student_dist = torch.tensor([0.2, 0.3, 0.5])
    teacher_dist = torch.tensor([0.1, 0.4, 0.5])

    reverse = opd_utils.compute_opd_kl_topk(student, teacher, kl_type="reverse_kl", norm_mode="tail")
    forward = opd_utils.compute_opd_kl_topk(student, teacher, kl_type="forward_kl", norm_mode="tail")
    expected_reverse = (student_dist * (student_dist / teacher_dist).log()).sum()
    expected_forward = (teacher_dist * (teacher_dist / student_dist).log()).sum()

    assert torch.allclose(reverse, expected_reverse.reshape(1))
    assert torch.allclose(forward, expected_forward.reshape(1))


@pytest.mark.parametrize(("dynamic_cp_size", "expected"), [(1, 2.0), (2, 2.0)])
def test_sdpo_token_normalizer_uses_original_loss_masks(opd_modules, monkeypatch, dynamic_cp_size, expected):
    _, _, torch = opd_modules
    megatron_loss = _import_megatron_loss()

    batch = _batch([2, 1], topk_teacher=[torch.ones((2, 2)), torch.ones((1, 2))])
    batch["opd_topk_token_ids"] = [torch.ones((2, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long)]
    monkeypatch.setattr(
        megatron_loss,
        "get_cp_local_num_tokens",
        lambda *args, **kwargs: torch.tensor(expected),
    )
    batch["dynamic_cp_size"] = dynamic_cp_size

    actual = megatron_loss._get_loss_num_tokens(batch, _legacy_args())

    assert actual.item() == expected


def test_sdpo_loss_function_preserves_rows_and_metric_denominators(opd_modules, monkeypatch):
    _, _, torch = opd_modules
    megatron_loss = _import_megatron_loss()

    observed = {}

    def fake_policy_loss(args, batch, logits, sum_of_sample_mean):
        observed["batch"] = batch
        return logits.sum() * 0 + 2.0, {
            "sdpo_topk_coverage": torch.tensor(1.0),
            "opd_kl": torch.tensor(3.0),
            "__sdpo_topk_coverage_denominator": torch.tensor(3.0),
        }

    monkeypatch.setattr(megatron_loss, "policy_loss_function", fake_policy_loss)
    monkeypatch.setattr(megatron_loss, "get_sum_of_sample_mean", lambda *args, **kwargs: object())
    args = Namespace(
        loss_type="policy_loss",
        opd_loss_mode="sdpo",
        calculate_per_token_loss=True,
        qkv_format="thd",
        allgather_cp=False,
        global_batch_size=2,
        recompute_loss_function=False,
    )
    batch = _batch(
        [2, 1],
        topk_teacher=[torch.ones((2, 2)), torch.ones((1, 2))],
        topk_ids=[torch.ones((2, 2), dtype=torch.long), torch.ones((1, 2), dtype=torch.long)],
    )
    logits = torch.ones(3, requires_grad=True)

    loss, normalizer, logging = megatron_loss.loss_function(args, batch, num_microbatches=1, logits=logits)

    assert loss.detach().item() == 2.0
    assert normalizer.item() == 3.0
    assert torch.equal(observed["batch"]["loss_masks"][0], torch.ones(2))
    assert torch.equal(observed["batch"]["loss_masks"][1], torch.ones(1))
    assert logging["keys"] == ["sdpo_topk_coverage", "opd_kl"]
    assert torch.equal(logging["values"], torch.tensor([3.0, 3.0, 3.0]))
    assert torch.equal(logging["metric_denominators"], torch.tensor([3.0, -1.0]))


def test_ordinary_loss_logging_keeps_legacy_payload_shape(opd_modules, monkeypatch):
    _, _, torch = opd_modules
    megatron_loss = _import_megatron_loss()
    monkeypatch.setattr(
        megatron_loss,
        "policy_loss_function",
        lambda args, batch, logits, sum_of_sample_mean: (torch.tensor(2.0), {"opd_kl": torch.tensor(4.0)}),
    )
    monkeypatch.setattr(megatron_loss, "get_sum_of_sample_mean", lambda *args, **kwargs: object())
    args = Namespace(
        loss_type="policy_loss",
        opd_loss_mode="opd",
        calculate_per_token_loss=False,
        qkv_format="thd",
        allgather_cp=False,
        global_batch_size=2,
        recompute_loss_function=False,
    )
    batch = _batch([1, 1])

    _, _, logging = megatron_loss.loss_function(args, batch, num_microbatches=1, logits=torch.ones(2))

    assert logging["keys"] == ["opd_kl"]
    assert torch.equal(logging["values"], torch.tensor([2.0, 4.0]))
    assert "metric_denominators" not in logging


def test_sdpo_reducer_preserves_sample_and_token_reduction_modes(opd_modules):
    _, sdpo_loss, torch = opd_modules
    batch = _batch([1, 2])
    values = torch.tensor([1.0, 1.0, 10.0], requires_grad=True)

    sample_mean_args = _legacy_args(calculate_per_token_loss=False)
    token_sum_args = _legacy_args(calculate_per_token_loss=True)
    sample_mean = sdpo_loss._reduce_sdpo_values(values, batch, sample_mean_args)
    token_sum = sdpo_loss._reduce_sdpo_values(values, batch, token_sum_args)

    assert torch.allclose(sample_mean, torch.tensor(6.5))
    assert torch.allclose(token_sum, torch.tensor(12.0))
