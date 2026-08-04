# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Two-GPU NCCL contracts for OPD Top-K and SDPO CP paths."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch
import torch.distributed as dist


WORLD_SIZE = 2


def _tp_worker(rank: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        from megatron.core import mpu

        mpu.initialize_model_parallel(tensor_model_parallel_size=WORLD_SIZE)
        from relax.utils.opd.opd_utils import compute_log_probs_on_topk_token_ids

        local_logits = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0]] if rank == 0 else [[4.0, 5.0, 6.0, 7.0]],
            device="cuda",
        )
        global_token_ids = torch.tensor([[0, 4, 7, -1]], dtype=torch.long, device="cuda")
        actual = compute_log_probs_on_topk_token_ids(local_logits, global_token_ids, dist.group.WORLD)

        global_logits = torch.arange(8, dtype=torch.float32, device="cuda").reshape(1, 8)
        expected = global_logits[:, [0, 4, 7]] - torch.logsumexp(global_logits, dim=-1, keepdim=True)
        torch.testing.assert_close(actual[:, :3], expected)
        assert torch.isneginf(actual[:, 3]).all()

        gathered = [torch.empty_like(actual) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered, actual)
        torch.testing.assert_close(gathered[0], gathered[1])
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _cp_layout_worker(rank: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        from megatron.core import mpu

        from relax.utils.opd.topk_layout import slice_opd_topk_rollout_fields

        mpu.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=WORLD_SIZE)

        rows = torch.arange(8, dtype=torch.float32, device="cuda").unsqueeze(1).repeat(1, 2)
        rollout_data = {
            "total_lengths": [16],
            "response_lengths": [8],
            "opd_topk_token_ids": [rows.to(dtype=torch.long)],
            "opd_topk_student_log_probs": [rows],
            "opd_topk_teacher_log_probs": [rows + 10],
        }
        args = Namespace(opd_token_selection="student_topk", qkv_format="thd", allgather_cp=False)
        slice_opd_topk_rollout_fields(
            rollout_data,
            args,
            dynamic_cp_size=WORLD_SIZE,
            dynamic_cp_rank=rank,
        )

        expected = (
            torch.arange(5, dtype=torch.float32, device="cuda")
            if rank == 1
            else torch.arange(5, 8, dtype=torch.float32, device="cuda")
        )
        actual = rollout_data["opd_topk_student_log_probs"][0]
        assert actual.is_cuda
        assert actual.shape == (expected.numel(), 2)
        torch.testing.assert_close(actual[:, 0], expected)

        local_sum = actual[:, 0].sum()
        dist.all_reduce(local_sum)
        torch.testing.assert_close(local_sum, torch.tensor(28.0, device="cuda"))
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _sdpo_cp_loss_worker(rank: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD_SIZE,
    )
    try:
        from megatron.core import mpu

        from relax.utils.opd.sdpo.loss import compute_sdpo_loss, compute_sdpo_topk_divergence

        mpu.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=WORLD_SIZE)

        local_rows = 0 if rank == 0 else 8
        student_seed = torch.tensor([[0.2, 0.3]], dtype=torch.float32, device="cuda", requires_grad=True)
        student = torch.log(student_seed).repeat(local_rows, 1)
        student.retain_grad()
        teacher = torch.log(torch.tensor([[0.1, 0.4]], dtype=torch.float32, device="cuda")).repeat(local_rows, 1)
        token_ids = torch.ones((local_rows, 2), dtype=torch.long, device="cuda")
        log_probs = torch.zeros(local_rows, dtype=torch.float32, device="cuda", requires_grad=True)
        args = Namespace(
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
        batch = {
            "total_lengths": [13],
            "response_lengths": [8],
            "padded_total_lengths": [16],
            "loss_masks": [torch.ones(8, dtype=torch.float32, device="cuda")],
            "dynamic_cp_size": WORLD_SIZE,
            "dynamic_cp_rank": rank,
            "opd_topk_token_ids": [token_ids],
            "opd_topk_teacher_log_probs": [teacher],
        }

        loss, metrics = compute_sdpo_loss(
            args=args,
            batch=batch,
            log_probs=log_probs,
            old_log_probs=log_probs.detach(),
            log_probs_and_entropy={"topk_log_probs": [student]},
        )
        assert loss is not None and torch.isfinite(loss)
        assert torch.isfinite(metrics["sdpo_topk_coverage"])
        loss.backward()
        assert log_probs.grad is not None
        if rank == 0:
            assert student.numel() == 0
        else:
            assert student.grad is not None and torch.isfinite(student.grad).all()

        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        full_student = torch.log(torch.tensor([[0.2, 0.3]] * 8, dtype=torch.float32, device="cuda"))
        full_teacher = torch.log(torch.tensor([[0.1, 0.4]] * 8, dtype=torch.float32, device="cuda"))
        expected = compute_sdpo_topk_divergence(
            full_student,
            full_teacher,
            kl_type="jsd",
            jsd_alpha=0.5,
            norm_mode="tail",
        ).sum()
        torch.testing.assert_close(global_loss, expected)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
def test_two_gpu_nccl_opd_topk_global_ids(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _tp_worker,
        args=(str(tmp_path / "nccl-init"),),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
def test_two_gpu_nccl_opd_cp_response_rows(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _cp_layout_worker,
        args=(str(tmp_path / "nccl-cp-init"),),
        nprocs=WORLD_SIZE,
        join=True,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE or not dist.is_nccl_available(),
    reason="requires two CUDA devices and NCCL",
)
def test_two_gpu_nccl_sdpo_cp_loss_handles_empty_local_shard(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _sdpo_cp_loss_worker,
        args=(str(tmp_path / "nccl-sdpo-cp-init"),),
        nprocs=WORLD_SIZE,
        join=True,
    )
