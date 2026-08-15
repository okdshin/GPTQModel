# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor


class _FakeDeviceTensor(torch.Tensor):
    """Wraps a real CPU tensor but reports a caller-chosen `.device`.

    Simulates a stale thread-local mask left on another device by a pooled
    worker thread, without needing actual multi-GPU hardware.
    """

    @staticmethod
    def wrap(tensor, fake_device):
        obj = tensor.as_subclass(_FakeDeviceTensor)
        obj._fake_device = torch.device(fake_device)
        obj.to_calls = []
        return obj

    @property
    def device(self):
        return self._fake_device

    def to(self, *args, **kwargs):
        self.to_calls.append(kwargs)
        return torch.Tensor.to(self.as_subclass(torch.Tensor), *args, **kwargs)


class _DummyTask:
    def add_batch(self, inp, out, batch_index=None):
        pass


class _DummyProcessor:
    def __init__(self, keep_mask):
        self.tasks = {"layer": _DummyTask()}
        self._mask_tls = type("Mask", (), {"value": keep_mask})()

    def current_batch_index(self):
        return 0


def _run_hook(fake_mask_device):
    keep_mask = _FakeDeviceTensor.wrap(torch.tensor([[True, False, True]]), fake_mask_device)
    inp_tensor = torch.randn(1, 3, 4)

    processor = _DummyProcessor(keep_mask)
    hook = GPTQProcessor.pre_process_fwd_hook(processor, "layer")
    hook(module=None, inp=(inp_tensor,), out=inp_tensor)

    return keep_mask, inp_tensor


def test_pre_process_fwd_hook_moves_stale_keep_mask_onto_input_device():
    keep_mask, inp_tensor = _run_hook(fake_mask_device="meta")

    assert keep_mask.to_calls == [{"device": inp_tensor.device}]


def test_pre_process_fwd_hook_skips_move_when_keep_mask_already_matches_device():
    keep_mask, _ = _run_hook(fake_mask_device="cpu")

    assert keep_mask.to_calls == []
