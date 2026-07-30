import importlib
import sys
from types import ModuleType

import pytest


def _run_static_cp_dr_grpo_worker(rank: int, init_method: str, result_path: str) -> None:
    import torch
    import torch.distributed as dist

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_rank = lambda: rank
    mpu.get_context_parallel_world_size = lambda: 2
    core.mpu = mpu
    megatron.core = core
    sys.modules["megatron"] = megatron
    sys.modules["megatron.core"] = core
    sys.modules["megatron.core.mpu"] = mpu
    sys.modules.pop("relax.backends.megatron.cp_utils", None)
    cp_utils = importlib.import_module("relax.backends.megatron.cp_utils")

    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        local_values = torch.tensor([1.0] if rank == 0 else [2.0, 3.0, 4.0], requires_grad=True)
        reducer = cp_utils.get_sequence_loss_aggregator(
            "seq-mean-token-sum-norm",
            total_lengths=[6],
            response_lengths=[4],
            loss_masks=[torch.ones(4)],
            scale_factor=8,
            dynamic_cp_size=2,
            dynamic_cp_rank=rank,
        )
        local_loss = reducer(local_values)
        step_token_normalizer = torch.tensor(float(local_values.numel()))
        dist.all_reduce(step_token_normalizer, op=dist.ReduceOp.SUM)
        scale = cp_utils.get_per_token_loss_scale(
            num_microbatches=1,
            global_batch_size=1,
            data_parallel_world_size=2,
            step_token_normalizer=step_token_normalizer,
        )
        (local_loss * scale).backward()

        fixed_scale_gradient = local_values.grad / (step_token_normalizer * 2)
        assert torch.equal(fixed_scale_gradient, torch.full_like(local_values, 1.0 / 8.0))

        total_loss = local_loss.detach().clone()
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        if rank == 0:
            torch.save(
                torch.stack([total_loss, step_token_normalizer, fixed_scale_gradient.mean()]),
                result_path,
            )
    finally:
        dist.destroy_process_group()


@pytest.fixture()
def cp_utils_module(monkeypatch):
    pytest.importorskip("torch")
    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    mpu = ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_context_parallel_world_size = lambda: 1
    core.mpu = mpu
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    monkeypatch.setitem(sys.modules, "megatron.core.mpu", mpu)
    sys.modules.pop("relax.backends.megatron.cp_utils", None)
    module = importlib.import_module("relax.backends.megatron.cp_utils")
    yield module
    sys.modules.pop("relax.backends.megatron.cp_utils", None)


def test_response_length_normalization_preserves_existing_behavior(cp_utils_module):
    import torch

    reducer = cp_utils_module.get_sum_of_sample_mean(
        total_lengths=[4, 5],
        response_lengths=[2, 3],
        loss_masks=[torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])],
    )

    value = reducer(torch.tensor([2.0, 11.0, 3.0, 5.0, 13.0]))

    assert torch.isclose(value, torch.tensor(6.0))


def test_seq_mean_token_sum_norm_uses_one_scale_factor_for_all_responses(cp_utils_module):
    import torch

    reducer = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm",
        total_lengths=[4, 5],
        response_lengths=[2, 3],
        loss_masks=[torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0, 0.0])],
        scale_factor=4,
    )
    values = torch.tensor([2.0, 11.0, 3.0, 5.0, 13.0], requires_grad=True)

    loss = reducer(values)
    loss.backward()

    assert torch.isclose(loss, torch.tensor(2.5))
    assert torch.equal(values.grad, torch.tensor([0.25, 0.0, 0.25, 0.25, 0.0]))


def test_seq_mean_token_sum_norm_requires_positive_scale_factor(cp_utils_module):
    import torch

    with pytest.raises(ValueError, match="scale_factor must be positive"):
        cp_utils_module.get_sequence_loss_aggregator(
            "seq-mean-token-sum-norm",
            total_lengths=[2],
            response_lengths=[1],
            loss_masks=[torch.tensor([1.0])],
            scale_factor=0,
        )


def test_seq_mean_token_sum_norm_cp_shards_sum_to_single_rank_value(cp_utils_module):
    import torch

    kwargs = {
        "total_lengths": [6],
        "response_lengths": [4],
        "loss_masks": [torch.ones(4)],
        "scale_factor": 8,
        "dynamic_cp_size": 2,
    }
    rank_zero = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm", dynamic_cp_rank=0, **kwargs
    )
    rank_one = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm", dynamic_cp_rank=1, **kwargs
    )

    cp_sum = rank_zero(torch.tensor([1.0])) + rank_one(torch.tensor([2.0, 3.0, 4.0]))

    assert torch.isclose(cp_sum, torch.tensor(10.0 / 8.0))


def test_per_token_finalizer_scale_recovers_fixed_dr_grpo_denominator(cp_utils_module):
    import torch

    first_microbatch = torch.tensor([2.0, 3.0], requires_grad=True)
    second_microbatch = torch.tensor([5.0], requires_grad=True)
    fixed_scale_factor = 8
    global_batch_size = 2
    step_token_normalizer = 3
    scale = cp_utils_module.get_per_token_loss_scale(
        num_microbatches=2,
        global_batch_size=global_batch_size,
        data_parallel_world_size=1,
        step_token_normalizer=step_token_normalizer,
    )

    megatron_loss = (
        (first_microbatch.sum() / fixed_scale_factor) * scale / 2
        + (second_microbatch.sum() / fixed_scale_factor) * scale / 2
    ) / step_token_normalizer
    megatron_loss.backward()

    assert torch.isclose(megatron_loss, torch.tensor(10.0 / (global_batch_size * fixed_scale_factor)))
    expected_gradient = 1.0 / (global_batch_size * fixed_scale_factor)
    assert torch.equal(first_microbatch.grad, torch.full_like(first_microbatch, expected_gradient))
    assert torch.equal(second_microbatch.grad, torch.full_like(second_microbatch, expected_gradient))


def test_per_token_finalizer_cp_shards_recover_fixed_dr_grpo_denominator(cp_utils_module):
    import torch

    fixed_scale_factor = 8
    step_token_normalizer = 4
    rank_zero = torch.tensor([1.0])
    rank_one = torch.tensor([2.0, 3.0, 4.0])
    scale = cp_utils_module.get_per_token_loss_scale(
        num_microbatches=1,
        global_batch_size=1,
        data_parallel_world_size=2,
        step_token_normalizer=step_token_normalizer,
    )

    final_loss = (
        (rank_zero.sum() / fixed_scale_factor + rank_one.sum() / fixed_scale_factor) * scale / 2
    ) / step_token_normalizer

    assert torch.isclose(final_loss, torch.tensor(10.0 / fixed_scale_factor))


def test_per_token_finalizer_requires_step_global_not_microbatch_normalizer(cp_utils_module):
    import torch

    fixed_scale_factor = 8
    first_microbatch = torch.tensor([2.0, 3.0])
    second_microbatch = torch.tensor([5.0])
    correct_scale = cp_utils_module.get_per_token_loss_scale(2, 2, 1, step_token_normalizer=3)
    incorrect_first_scale = cp_utils_module.get_per_token_loss_scale(2, 2, 1, step_token_normalizer=2)
    incorrect_second_scale = cp_utils_module.get_per_token_loss_scale(2, 2, 1, step_token_normalizer=1)

    correct_loss = (
        first_microbatch.sum() / fixed_scale_factor * correct_scale / 2
        + second_microbatch.sum() / fixed_scale_factor * correct_scale / 2
    ) / 3
    microbatch_weighted_loss = (
        first_microbatch.sum() / fixed_scale_factor * incorrect_first_scale / 2
        + second_microbatch.sum() / fixed_scale_factor * incorrect_second_scale / 2
    ) / 3

    assert torch.isclose(correct_loss, torch.tensor(10.0 / 16.0))
    assert not torch.isclose(microbatch_weighted_loss, correct_loss)


def test_step_loss_normalizer_is_shared_by_each_microbatch_in_a_step(cp_utils_module):
    import torch

    first_step, second_step = torch.tensor(7.0), torch.tensor(11.0)

    expanded = cp_utils_module.expand_step_loss_normalizers([first_step, second_step], [2, 3])

    assert expanded == [first_step, first_step, second_step, second_step, second_step]


def test_static_cp_dr_grpo_matches_cp_one_fixed_scale_gradient(tmp_path, cp_utils_module):
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is required for the static CP process-group test.")

    cp_one_values = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    cp_one_reducer = cp_utils_module.get_sequence_loss_aggregator(
        "seq-mean-token-sum-norm",
        total_lengths=[6],
        response_lengths=[4],
        loss_masks=[torch.ones(4)],
        scale_factor=8,
    )
    cp_one_loss = cp_one_reducer(cp_one_values)
    cp_one_loss.backward()

    init_file = tmp_path / "gloo-init"
    result_path = tmp_path / "result.pt"
    torch.multiprocessing.spawn(
        _run_static_cp_dr_grpo_worker,
        args=(f"file://{init_file}", str(result_path)),
        nprocs=2,
        join=True,
    )

    total_loss, step_token_normalizer, fixed_scale_gradient = torch.load(result_path, weights_only=True)

    assert torch.isclose(total_loss, cp_one_loss.detach())
    assert torch.isclose(step_token_normalizer, torch.tensor(4.0))
    assert torch.isclose(fixed_scale_gradient, cp_one_values.grad.mean())
