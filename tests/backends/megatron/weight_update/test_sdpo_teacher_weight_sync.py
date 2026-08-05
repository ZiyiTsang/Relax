# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace

import pytest


class _FakeRemoteMethod:
    def __init__(self, event, name):
        self._event = event
        self._name = name

    def remote(self, *args, **kwargs):
        self._event.append(self._name)
        return self._name


class _FakeEngine:
    def __init__(self, event):
        self.pause_generation = _FakeRemoteMethod(event, "pause")
        self.flush_cache = _FakeRemoteMethod(event, "flush")
        self.continue_generation = _FakeRemoteMethod(event, "continue")


class _FakeIterator:
    def get_hf_weight_chunks(self, weights):
        yield [("weight", weights["weight"])]


def _nonzero_rank_failure_worker(rank, init_file):
    import torch
    import torch.distributed as dist

    from relax.backends.megatron.weight_update import update_weight_from_tensor as module
    from relax.utils.distributed_utils import init_gloo_group

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    init_gloo_group()

    class _RemoteMethod:
        def remote(self, *args, **kwargs):
            return None

    class _Engine:
        pause_generation = _RemoteMethod()
        flush_cache = _RemoteMethod()
        continue_generation = _RemoteMethod()

    updater = module.UpdateWeightFromTensor.__new__(module.UpdateWeightFromTensor)
    updater.args = SimpleNamespace()
    updater.model = []
    updater.weights_getter = lambda: {"weight": torch.tensor([1.0])}
    updater.model_name = "test"
    updater.quantization_config = None
    updater.weight_version = 0
    updater.lora_enabled = False
    updater.lora_adapter_mode = False
    updater._hf_weight_iterator = _FakeIterator()
    updater.rollout_engines = [_Engine()]
    updater.distributed_rollout_engines = []

    def send_hf_params(_named_tensors, *, weight_version=None):
        if rank == 1:
            raise RuntimeError("nonzero sender failed")
        return [], None

    updater._send_hf_params = send_hf_params
    module.ray.get = lambda refs: None
    module.device_utils.maybe_backend_barrier_on_weight_chunk = lambda **kwargs: None

    try:
        updater.update_weights()
    except RuntimeError as exc:
        assert "Weight update failed before commit" in str(exc)
        assert updater.weight_version == 0
    else:
        raise AssertionError("expected synchronized weight update failure")
    finally:
        dist.destroy_process_group()


def _make_updater(monkeypatch, event, *, send_error=None, versions=None):
    import torch

    from relax.backends.megatron.weight_update import update_weight_from_tensor as module

    updater = module.UpdateWeightFromTensor.__new__(module.UpdateWeightFromTensor)
    updater.args = SimpleNamespace()
    updater.model = []
    updater.weights_getter = lambda: {"weight": torch.tensor([1.0])}
    updater.model_name = "test"
    updater.quantization_config = None
    updater.weight_version = 0
    updater.lora_enabled = False
    updater.lora_adapter_mode = False
    updater._hf_weight_iterator = _FakeIterator()
    updater.rollout_engines = [_FakeEngine(event)]
    updater.distributed_rollout_engines = []

    def send_hf_params(_named_tensors, *, weight_version=None):
        event.append("send")
        if versions is not None:
            versions.append(weight_version)
        if send_error is not None:
            raise send_error
        return [], None

    updater._send_hf_params = send_hf_params
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: event.append("barrier"))
    monkeypatch.setattr(module.dist, "all_reduce", lambda tensor, **kwargs: event.append("all_reduce"))
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(module.device_utils, "maybe_backend_barrier_on_weight_chunk", lambda **kwargs: None)
    monkeypatch.setattr(module.ray, "get", lambda refs: None)
    return updater


def test_sdpo_teacher_publish_pauses_flushes_transfers_and_resumes(monkeypatch):
    event = []
    versions = []
    updater = _make_updater(monkeypatch, event, versions=versions)

    updater.update_weights()

    assert event[:3] == ["pause", "flush", "barrier"]
    assert event.count("all_reduce") == 8
    assert event.count("send") == 1
    assert event[-2:] == ["barrier", "all_reduce"]
    assert updater.weight_version == 1
    assert versions == [1]


def test_sdpo_teacher_publish_failure_does_not_resume_serving(monkeypatch):
    event = []
    versions = []
    updater = _make_updater(monkeypatch, event, send_error=RuntimeError("chunk failed"), versions=versions)

    with pytest.raises(RuntimeError, match="chunk failed"):
        updater.update_weights()

    assert event[:3] == ["pause", "flush", "barrier"]
    assert event.count("all_reduce") == 5
    assert "continue" not in event
    assert updater.weight_version == 0
    assert versions == [1]


def test_sdpo_teacher_publish_retry_starts_from_uncommitted_version(monkeypatch):
    event = []
    failed_updater = _make_updater(monkeypatch, event, send_error=RuntimeError("chunk failed"))

    with pytest.raises(RuntimeError, match="chunk failed"):
        failed_updater.update_weights()
    assert failed_updater.weight_version == 0

    retry_event = []
    retry_versions = []
    retry_updater = _make_updater(monkeypatch, retry_event, versions=retry_versions)
    retry_updater.update_weights()

    assert retry_updater.weight_version == 1
    assert retry_versions == [1]


def test_weight_update_coordinates_nonzero_sender_failure(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(
        _nonzero_rank_failure_worker,
        args=(str(tmp_path / "weight-update-failure"),),
        nprocs=2,
        join=True,
    )


def test_torch_memory_saver_uses_cpu_weight_serialization(monkeypatch):
    import torch

    from relax.backends.megatron.weight_update import update_weight_from_tensor as module

    captured = {}

    class _RemoteMethod:
        def remote(self, **kwargs):
            captured["request"] = kwargs
            return "ref"

    class _Engine:
        update_weights_from_tensor = _RemoteMethod()

    def fake_gather_object(obj, object_gather_list, **_kwargs):
        object_gather_list[0] = obj
        captured["serialized"] = obj

    monkeypatch.setenv("TMS_INIT_ENABLE", "1")
    monkeypatch.setattr(module, "make_current_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda _group=None: 1)
    monkeypatch.setattr(module.dist, "gather_object", fake_gather_object)

    from torch.multiprocessing import get_sharing_strategy

    previous_strategy = get_sharing_strategy()
    refs, long_lived_tensors = module._send_to_colocated_engine(
        [("weight", torch.ones(4))],
        ipc_engine=_Engine(),
        ipc_gather_src=0,
        ipc_gather_group=object(),
        weight_version=1,
    )

    assert refs == ["ref"]
    assert get_sharing_strategy() == previous_strategy
    assert long_lived_tensors[0]["flattened_tensor"].device.type == "cpu"
    decoded = module.MultiprocessingSerializer.deserialize(captured["serialized"][0])
    assert decoded["flattened_tensor"].device.type == "cpu"
