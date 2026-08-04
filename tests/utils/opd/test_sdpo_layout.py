# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Focused tests for SDPO Top-K context-parallel response layout."""

from argparse import Namespace

import pytest
import torch

from relax.utils.opd.opd_utils import (  # noqa: E402
    slice_opd_topk_rollout_fields as slice_legacy_opd_topk_rollout_fields,
)
from relax.utils.opd.topk_layout import (  # noqa: E402
    get_opd_local_response_lengths,
    slice_opd_topk_rollout_fields,
)


def _args(**overrides) -> Namespace:
    values = {
        "opd_token_selection": "student_topk",
        "qkv_format": "thd",
        "allgather_cp": False,
    }
    values.update(overrides)
    return Namespace(**values)


def _rollout_data() -> dict:
    rows = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    values = {
        "opd_topk_token_ids": [rows.to(dtype=torch.long).repeat(1, 2)],
        "opd_topk_student_log_probs": [rows.repeat(1, 2)],
        "opd_topk_teacher_log_probs": [(rows + 10).repeat(1, 2)],
    }
    return {
        "total_lengths": [16],
        "response_lengths": [8],
        **values,
    }


def test_sdpo_layout_is_noop_for_cp_one() -> None:
    rollout_data = _rollout_data()
    original = {key: value for key, value in rollout_data.items() if key.startswith("opd_")}

    slice_opd_topk_rollout_fields(rollout_data, _args(), dynamic_cp_size=1, dynamic_cp_rank=0)

    for key, value in original.items():
        assert rollout_data[key] is value


def test_sdpo_layout_slices_response_rows_but_not_topk_columns() -> None:
    rollout_data = _rollout_data()

    slice_opd_topk_rollout_fields(rollout_data, _args(), dynamic_cp_size=2, dynamic_cp_rank=0)

    for key, value in rollout_data.items():
        if key.startswith("opd_"):
            assert value[0].shape == (3, 2)
            offset = 10 if key.endswith("teacher_log_probs") else 0
            expected = torch.tensor([5 + offset, 6 + offset, 7 + offset], dtype=value[0].dtype)
            assert torch.equal(value[0][:, 0], expected)
            assert torch.equal(value[0][:, 1], expected)


def test_sdpo_layout_slices_the_other_cp_rank_with_global_row_order() -> None:
    rollout_data = _rollout_data()

    assert get_opd_local_response_lengths(rollout_data, _args(), dynamic_cp_size=2, dynamic_cp_rank=1) == [5]
    slice_opd_topk_rollout_fields(rollout_data, _args(), dynamic_cp_size=2, dynamic_cp_rank=1)

    for key, value in rollout_data.items():
        if key.startswith("opd_"):
            assert value[0].shape == (5, 2)
            offset = 10 if key.endswith("teacher_log_probs") else 0
            expected = torch.tensor([offset, 1 + offset, 2 + offset, 3 + offset, 4 + offset], dtype=value[0].dtype)
            assert torch.equal(value[0][:, 0], expected)


def test_cp_two_reassembles_the_same_topk_rows_as_cp_one() -> None:
    cp_one = _rollout_data()
    cp_rank_one = _rollout_data()
    cp_rank_zero = _rollout_data()

    slice_opd_topk_rollout_fields(cp_rank_one, _args(), dynamic_cp_size=2, dynamic_cp_rank=1)
    slice_opd_topk_rollout_fields(cp_rank_zero, _args(), dynamic_cp_size=2, dynamic_cp_rank=0)

    for key in ("opd_topk_token_ids", "opd_topk_student_log_probs", "opd_topk_teacher_log_probs"):
        reassembled = torch.cat([cp_rank_one[key][0], cp_rank_zero[key][0]], dim=0)
        torch.testing.assert_close(reassembled, cp_one[key][0])


def _sdpo_tail_oracle(student: torch.Tensor, teacher: torch.Tensor, alpha: float) -> torch.Tensor:
    student = student.float()
    teacher = teacher.float()
    student_tail = torch.log1p(-student.exp().sum(dim=-1, keepdim=True))
    teacher_tail = torch.log1p(-teacher.exp().sum(dim=-1, keepdim=True))
    student = torch.cat([student, student_tail], dim=-1)
    teacher = torch.cat([teacher, teacher_tail], dim=-1)
    if alpha == 0.0:
        return (teacher.exp() * (teacher - student)).sum(dim=-1)
    if alpha == 1.0:
        return (student.exp() * (student - teacher)).sum(dim=-1)
    mixture = torch.log((1.0 - alpha) * student.exp() + alpha * teacher.exp())
    student_kl = (student.exp() * (student - mixture)).sum(dim=-1)
    teacher_kl = (teacher.exp() * (teacher - mixture)).sum(dim=-1)
    return (1.0 - alpha) * student_kl + alpha * teacher_kl


@pytest.mark.parametrize("jsd_alpha", [0.0, 0.5, 1.0])
def test_cp_two_preserves_sdpo_divergence_rows(jsd_alpha: float) -> None:
    from relax.utils.opd.sdpo.loss import compute_sdpo_topk_divergence

    student = torch.log(torch.tensor([[0.20, 0.30], [0.25, 0.15]]).repeat(4, 1))
    teacher = torch.log(torch.tensor([[0.10, 0.40], [0.30, 0.20]]).repeat(4, 1))
    expected = _sdpo_tail_oracle(student, teacher, jsd_alpha)

    local_values = []
    for cp_rank in (1, 0):
        rollout_data = _rollout_data()
        rollout_data["opd_topk_student_log_probs"] = [student.clone()]
        rollout_data["opd_topk_teacher_log_probs"] = [teacher.clone()]
        slice_opd_topk_rollout_fields(rollout_data, _args(), dynamic_cp_size=2, dynamic_cp_rank=cp_rank)
        local_values.append(
            compute_sdpo_topk_divergence(
                rollout_data["opd_topk_student_log_probs"][0],
                rollout_data["opd_topk_teacher_log_probs"][0],
                kl_type="jsd",
                jsd_alpha=jsd_alpha,
                norm_mode="tail",
            )
        )

    torch.testing.assert_close(torch.cat(local_values), expected)


def test_sdpo_layout_uses_padded_lengths_and_preserves_empty_cp_shard() -> None:
    rollout_data = _rollout_data()
    rollout_data["total_lengths"] = [13]
    rollout_data["response_lengths"] = [8]
    rollout_data["padded_total_lengths"] = [16]

    assert get_opd_local_response_lengths(rollout_data, _args(), dynamic_cp_size=2, dynamic_cp_rank=0) == [0]
    slice_opd_topk_rollout_fields(rollout_data, _args(), dynamic_cp_size=2, dynamic_cp_rank=0)

    for key, value in rollout_data.items():
        if key.startswith("opd_"):
            assert value[0].shape == (0, 2)

    full_rank = _rollout_data()
    full_rank["total_lengths"] = [13]
    full_rank["padded_total_lengths"] = [16]
    slice_opd_topk_rollout_fields(full_rank, _args(), dynamic_cp_size=2, dynamic_cp_rank=1)
    for key, value in full_rank.items():
        if key.startswith("opd_"):
            assert value[0].shape == (8, 2)


def test_legacy_opd_entrypoint_injects_shared_layout(monkeypatch) -> None:
    observed = {}

    def fake_layout(data, args, **kwargs):
        observed["data"] = data
        observed["args"] = args
        observed["kwargs"] = kwargs

    monkeypatch.setattr("relax.utils.opd.topk_layout.slice_opd_topk_rollout_fields", fake_layout)
    rollout_data = _rollout_data()
    args = _args()

    slice_legacy_opd_topk_rollout_fields(rollout_data, args, dynamic_cp_size=2, dynamic_cp_rank=0)

    assert observed == {"data": rollout_data, "args": args, "kwargs": {"dynamic_cp_size": 2, "dynamic_cp_rank": 0}}


def test_sdpo_layout_preserves_empty_rows() -> None:
    rollout_data = _rollout_data()
    for key in (
        "opd_topk_token_ids",
        "opd_topk_student_log_probs",
        "opd_topk_teacher_log_probs",
    ):
        rollout_data[key] = [torch.empty((0, 2))]

    slice_opd_topk_rollout_fields(rollout_data, _args(), dynamic_cp_size=2, dynamic_cp_rank=0)

    for key in rollout_data:
        if key.startswith("opd_"):
            assert rollout_data[key][0].shape == (0, 2)


def test_sdpo_layout_rejects_allgather_cp() -> None:
    with pytest.raises(NotImplementedError, match="allgather_cp"):
        slice_opd_topk_rollout_fields(_rollout_data(), _args(allgather_cp=True), dynamic_cp_size=2, dynamic_cp_rank=0)


def test_sdpo_layout_ignores_non_student_topk() -> None:
    rollout_data = _rollout_data()
    original = {key: value for key, value in rollout_data.items() if key.startswith("opd_")}

    slice_opd_topk_rollout_fields(
        rollout_data, _args(opd_token_selection="student_sampled"), dynamic_cp_size=2, dynamic_cp_rank=0
    )

    for key, value in original.items():
        assert rollout_data[key] is value


def test_sdpo_union_layout_slices_row_lengths_with_the_same_offsets() -> None:
    rollout_data = _rollout_data()
    rollout_data["opd_topk_ksz"] = [torch.arange(8, dtype=torch.long) + 1]

    slice_opd_topk_rollout_fields(
        rollout_data,
        _args(opd_token_selection="union"),
        dynamic_cp_size=2,
        dynamic_cp_rank=0,
    )

    assert rollout_data["opd_topk_ksz"][0].tolist() == [6, 7, 8]
    assert rollout_data["opd_topk_token_ids"][0].shape == (3, 2)


def test_sdpo_layout_rejects_mismatched_sample_count() -> None:
    rollout_data = _rollout_data()
    rollout_data["opd_topk_teacher_log_probs"] = [torch.ones(8, 2), torch.ones(8, 2)]

    with pytest.raises(ValueError, match="sample count"):
        slice_opd_topk_rollout_fields(
            rollout_data,
            _args(),
            dynamic_cp_size=2,
            dynamic_cp_rank=0,
        )
