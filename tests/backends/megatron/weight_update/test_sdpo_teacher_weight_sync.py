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


def _make_updater(monkeypatch, event, *, send_error=None):
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

    def send_hf_params(_named_tensors):
        event.append("send")
        if send_error is not None:
            raise send_error
        return [], None

    updater._send_hf_params = send_hf_params
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: event.append("barrier"))
    monkeypatch.setattr(module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(module.device_utils, "maybe_backend_barrier_on_weight_chunk", lambda **kwargs: None)
    monkeypatch.setattr(module.ray, "get", lambda refs: None)
    return updater


def test_sdpo_teacher_publish_pauses_flushes_transfers_and_resumes(monkeypatch):
    event = []
    updater = _make_updater(monkeypatch, event)

    updater.update_weights()

    assert event == ["pause", "flush", "barrier", "send", "barrier", "continue", "barrier"]
    assert updater.weight_version == 1


def test_sdpo_teacher_publish_failure_does_not_resume_serving(monkeypatch):
    event = []
    updater = _make_updater(monkeypatch, event, send_error=RuntimeError("chunk failed"))

    with pytest.raises(RuntimeError, match="chunk failed"):
        updater.update_weights()

    assert event == ["pause", "flush", "barrier", "send"]
    assert "continue" not in event
