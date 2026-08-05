# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-rank CPU collective tests for the OPD/SDPO TP and CP contracts."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch
import torch.distributed as dist


WORLD_SIZE = 2
TP_CP_WORLD_SIZE = 4


def _init_process_group(init_file: str, rank: int) -> None:
    _init_process_group_with_size(init_file, rank, WORLD_SIZE)


def _init_process_group_with_size(init_file: str, rank: int, world_size: int) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _spawn_two_ranks(worker, init_file: str) -> None:
    torch.multiprocessing.spawn(worker, args=(init_file,), nprocs=WORLD_SIZE, join=True)


def _spawn_four_ranks(worker, init_file: str) -> None:
    torch.multiprocessing.spawn(worker, args=(init_file,), nprocs=TP_CP_WORLD_SIZE, join=True)


def _sdpo_tail_reference(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    student = student.float()
    teacher = teacher.float()
    student = torch.cat([student, torch.log1p(-student.exp().sum(dim=-1, keepdim=True))], dim=-1)
    teacher = torch.cat([teacher, torch.log1p(-teacher.exp().sum(dim=-1, keepdim=True))], dim=-1)
    mixture = torch.log(0.5 * student.exp() + 0.5 * teacher.exp())
    return 0.5 * (
        (student.exp() * (student - mixture)).sum(dim=-1) + (teacher.exp() * (teacher - mixture)).sum(dim=-1)
    )


def _tp_global_vocab_worker(rank: int, init_file: str) -> None:
    _init_process_group(init_file, rank)
    try:
        from megatron.core import mpu

        mpu.initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)
        from relax.utils.opd.opd_utils import compute_log_probs_on_topk_token_ids

        local_logits = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0]] if rank == 0 else [[4.0, 5.0, 6.0, 7.0]],
            requires_grad=True,
        )
        global_token_ids = torch.tensor([[0, 4, 7, -1]], dtype=torch.long)
        actual = compute_log_probs_on_topk_token_ids(local_logits, global_token_ids, dist.group.WORLD)
        global_logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
        expected = global_logits[:, [0, 4, 7]] - torch.logsumexp(global_logits, dim=-1, keepdim=True)
        torch.testing.assert_close(actual[:, :3], expected)
        assert torch.isneginf(actual[:, 3]).all()
        assert torch.isfinite(actual[:, :3]).all()

        gathered = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, actual.detach().tolist())
        assert gathered[0] == gathered[1]
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _cp_fixed_topk_worker(rank: int, init_file: str) -> None:
    _init_process_group(init_file, rank)
    try:
        from argparse import Namespace

        from relax.utils.opd.topk_layout import slice_opd_topk_rollout_fields

        rows = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        rollout_data = {
            "total_lengths": [16],
            "response_lengths": [8],
            "opd_topk_token_ids": [rows.to(dtype=torch.long).repeat(1, 3)],
            "opd_topk_student_log_probs": [rows.repeat(1, 3)],
            "opd_topk_teacher_log_probs": [(rows + 10).repeat(1, 3)],
        }
        args = Namespace(opd_token_selection="student_topk", qkv_format="thd", allgather_cp=False)
        slice_opd_topk_rollout_fields(
            rollout_data,
            args,
            dynamic_cp_size=WORLD_SIZE,
            dynamic_cp_rank=rank,
        )

        expected = torch.arange(5, dtype=torch.float32) if rank == 1 else torch.arange(5, 8, dtype=torch.float32)
        actual = rollout_data["opd_topk_student_log_probs"][0]
        assert actual.shape == (expected.numel(), 3)
        torch.testing.assert_close(actual[:, 0], expected)
        assert actual.shape[-1] == 3
        torch.testing.assert_close(rollout_data["opd_topk_token_ids"][0][:, 0].float(), expected)
    finally:
        dist.destroy_process_group()


def _cp_union_padded_worker(rank: int, init_file: str) -> None:
    _init_process_group(init_file, rank)
    try:
        from argparse import Namespace

        from relax.utils.opd.topk_layout import slice_opd_topk_rollout_fields

        first = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
        second = (torch.arange(8, dtype=torch.float32) + 20).unsqueeze(1).repeat(1, 2)
        rollout_data = {
            "total_lengths": [16, 13],
            "response_lengths": [8, 8],
            "padded_total_lengths": [16, 16],
            "opd_topk_token_ids": [first.to(dtype=torch.long), second.to(dtype=torch.long)],
            "opd_topk_student_log_probs": [first, second],
            "opd_topk_teacher_log_probs": [first + 10, second + 10],
            "opd_topk_ksz": [torch.arange(8, dtype=torch.long) + 1, torch.arange(8, dtype=torch.long) + 11],
        }
        args = Namespace(opd_token_selection="union", qkv_format="thd", allgather_cp=False)
        slice_opd_topk_rollout_fields(
            rollout_data,
            args,
            dynamic_cp_size=WORLD_SIZE,
            dynamic_cp_rank=rank,
        )

        first_expected = torch.arange(5, dtype=torch.float32) if rank == 1 else torch.arange(5, 8, dtype=torch.float32)
        first_actual = rollout_data["opd_topk_student_log_probs"][0]
        assert first_actual.shape == (first_expected.numel(), 2)
        torch.testing.assert_close(first_actual[:, 0], first_expected)

        second_actual = rollout_data["opd_topk_student_log_probs"][1]
        second_lengths = rollout_data["opd_topk_ksz"][1]
        if rank == 0:
            assert second_actual.shape == (0, 2)
            assert second_lengths.numel() == 0
        else:
            assert second_actual.shape == (8, 2)
            torch.testing.assert_close(second_actual[:, 0], torch.arange(8, dtype=torch.float32) + 20)
            torch.testing.assert_close(second_lengths, torch.arange(8, dtype=torch.long) + 11)
        assert first_actual.shape[-1] == second_actual.shape[-1] == 2
    finally:
        dist.destroy_process_group()


def _cp_reducer_boundary_worker(rank: int, init_file: str) -> None:
    _init_process_group(init_file, rank)
    try:
        from argparse import Namespace

        from relax.utils.opd.opd_utils import compute_policy_opd_loss

        local_values = torch.empty(0, requires_grad=True) if rank == 0 else torch.tensor([10.0, 20.0])
        local_teacher = torch.empty(0) if rank == 0 else torch.zeros(2)
        args = Namespace(
            opd_loss_coef=1.0,
            opd_kl_type="reverse_kl",
            opd_jsd_alpha=0.5,
            opd_norm_mode="tail",
            opd_token_selection="student_sampled",
            opd_log_prob_min_clamp=None,
            opd_per_token_clip=None,
            opd_is_clip=None,
            calculate_per_token_loss=True,
            qkv_format="thd",
        )
        batch = {
            "total_lengths": [7],
            "response_lengths": [2],
            "loss_masks": [torch.ones(2)],
            "dynamic_cp_size": WORLD_SIZE,
            "dynamic_cp_rank": rank,
            "teacher_log_probs": [local_teacher],
        }

        actual, _ = compute_policy_opd_loss(
            args=args,
            batch=batch,
            log_probs=local_values,
            old_log_probs=local_values.detach(),
            log_probs_and_entropy={},
        )
        reduced = actual.detach().clone()
        dist.all_reduce(reduced)
        torch.testing.assert_close(reduced, torch.tensor(30.0))
    finally:
        dist.destroy_process_group()


def _cp_topk_reducer_boundary_worker(rank: int, init_file: str) -> None:
    _init_process_group(init_file, rank)
    try:
        from argparse import Namespace

        from relax.utils.opd.opd_utils import compute_opd_kl_topk, compute_policy_opd_loss
        from relax.utils.opd.topk_layout import slice_opd_topk_rollout_fields

        full_student = torch.log(torch.tensor([[0.2, 0.3], [0.3, 0.2]]))
        full_teacher = torch.log(torch.tensor([[0.1, 0.4], [0.2, 0.3]]))
        rollout_data = {
            "total_lengths": [7],
            "response_lengths": [2],
            "opd_topk_token_ids": [torch.ones((2, 2), dtype=torch.long)],
            "opd_topk_student_log_probs": [full_student],
            "opd_topk_teacher_log_probs": [full_teacher],
        }
        args = Namespace(opd_token_selection="student_topk", qkv_format="thd", allgather_cp=False)
        slice_opd_topk_rollout_fields(
            rollout_data,
            args,
            dynamic_cp_size=WORLD_SIZE,
            dynamic_cp_rank=rank,
        )

        local_student = rollout_data["opd_topk_student_log_probs"][0].requires_grad_()
        local_teacher = rollout_data["opd_topk_teacher_log_probs"][0]
        loss_args = Namespace(
            opd_loss_coef=1.0,
            opd_kl_type="reverse_kl",
            opd_jsd_alpha=0.5,
            opd_norm_mode="tail",
            opd_token_selection="student_topk",
            opd_log_prob_min_clamp=None,
            opd_per_token_clip=None,
            opd_is_clip=None,
            calculate_per_token_loss=True,
            qkv_format="thd",
        )
        batch = {
            "total_lengths": [7],
            "response_lengths": [2],
            "loss_masks": [torch.ones(2)],
            "dynamic_cp_size": WORLD_SIZE,
            "dynamic_cp_rank": rank,
            "opd_topk_teacher_log_probs": [local_teacher],
        }
        actual, _ = compute_policy_opd_loss(
            args=loss_args,
            batch=batch,
            log_probs=torch.zeros(local_student.size(0)),
            old_log_probs=torch.zeros(local_student.size(0)),
            log_probs_and_entropy={"topk_log_probs": [local_student]},
        )
        reduced = actual.detach().clone()
        dist.all_reduce(reduced)
        expected = compute_opd_kl_topk(full_student, full_teacher, kl_type="reverse_kl").sum()
        torch.testing.assert_close(reduced, expected)
    finally:
        dist.destroy_process_group()


def _tp_cp_topk_worker(rank: int, init_file: str) -> None:
    _init_process_group_with_size(init_file, rank, TP_CP_WORLD_SIZE)
    model_parallel_initialized = False
    try:
        from megatron.core import mpu

        from relax.utils.opd.opd_utils import compute_log_probs_on_topk_token_ids
        from relax.utils.opd.sdpo.loss import compute_sdpo_topk_divergence
        from relax.utils.opd.topk_layout import slice_opd_topk_rollout_fields

        mpu.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
        model_parallel_initialized = True
        tp_rank = mpu.get_tensor_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        tp_group = mpu.get_tensor_model_parallel_group()

        row_ids = torch.arange(5) if cp_rank == 1 else torch.arange(5, 8)
        vocab_size = 8
        topk_ids = torch.tensor([[0, 4, 7]] * 8, dtype=torch.long)
        global_logits = torch.arange(64, dtype=torch.float32).reshape(8, vocab_size) / 10.0
        full_student = torch.log_softmax(global_logits, dim=-1).gather(dim=-1, index=topk_ids)
        full_teacher = torch.log(torch.tensor([[0.10, 0.20, 0.30]] * 8, dtype=torch.float32))
        rollout_data = {
            "total_lengths": [16],
            "response_lengths": [8],
            "opd_topk_token_ids": [topk_ids.clone()],
            "opd_topk_student_log_probs": [full_student.clone()],
            "opd_topk_teacher_log_probs": [full_teacher.clone()],
        }
        args = Namespace(opd_token_selection="student_topk", qkv_format="thd", allgather_cp=False)
        slice_opd_topk_rollout_fields(
            rollout_data,
            args,
            dynamic_cp_size=2,
            dynamic_cp_rank=cp_rank,
        )

        local_ids = rollout_data["opd_topk_token_ids"][0]
        local_teacher = rollout_data["opd_topk_teacher_log_probs"][0]
        local_logits = global_logits[row_ids, tp_rank * (vocab_size // 2) : (tp_rank + 1) * (vocab_size // 2)]
        actual_student = compute_log_probs_on_topk_token_ids(local_logits, local_ids, tp_group)
        expected_student = full_student[row_ids]
        torch.testing.assert_close(actual_student, expected_student)
        torch.testing.assert_close(rollout_data["opd_topk_student_log_probs"][0], expected_student)
        torch.testing.assert_close(local_ids, topk_ids[row_ids])

        tp_consistent = actual_student.clone()
        dist.all_reduce(tp_consistent, group=tp_group)
        torch.testing.assert_close(tp_consistent / 2, expected_student)

        local_loss = compute_sdpo_topk_divergence(
            actual_student,
            local_teacher,
            kl_type="jsd",
            jsd_alpha=0.5,
            norm_mode="tail",
        ).sum()
        global_loss = local_loss.clone()
        dist.all_reduce(global_loss)
        expected_loss = _sdpo_tail_reference(full_student, full_teacher).sum()
        torch.testing.assert_close(global_loss / 2, expected_loss)

        gathered = [None] * TP_CP_WORLD_SIZE
        dist.all_gather_object(gathered, (cp_rank, tp_rank, actual_student.tolist()))
        if rank == 0:
            by_cp = {}
            for gathered_cp_rank, gathered_tp_rank, values in gathered:
                by_cp.setdefault(gathered_cp_rank, {})[gathered_tp_rank] = torch.tensor(values)
            for values in by_cp.values():
                torch.testing.assert_close(values[0], values[1])
            reassembled = torch.cat([by_cp[1][0], by_cp[0][0]], dim=0)
            torch.testing.assert_close(reassembled, full_student)
    finally:
        if model_parallel_initialized:
            mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("worker", [_cp_fixed_topk_worker, _cp_union_padded_worker])
def test_two_rank_opd_sdpo_layout_contracts(tmp_path, worker) -> None:
    _spawn_two_ranks(worker, str(tmp_path / f"init-{worker.__name__}"))


def test_two_rank_tp_global_vocab_contract(tmp_path) -> None:
    pytest.importorskip("megatron.core")
    _spawn_two_ranks(_tp_global_vocab_worker, str(tmp_path / "init-tp-global-vocab"))


@pytest.mark.parametrize("worker", [_cp_reducer_boundary_worker, _cp_topk_reducer_boundary_worker])
def test_two_rank_opd_cp_reducer_boundaries(tmp_path, worker) -> None:
    _spawn_two_ranks(worker, str(tmp_path / f"init-{worker.__name__}"))


def test_four_rank_tp_two_cp_two_topk_contract(tmp_path) -> None:
    pytest.importorskip("megatron.core")
    _spawn_four_ranks(_tp_cp_topk_worker, str(tmp_path / "init-tp2-cp2"))
