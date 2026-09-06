# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ForwardExecutor retries a module's forward once after a recoverable OOM,
re-invoking every hook that already fired during the failed attempt. Each
processor's per-batch statistics capture must therefore dedupe by batch
index so a retry doesn't double-count -- not just GPTQ's Hessian calibration.
"""

import threading
import types

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.eora_processor import EoraProcessor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.quantization import qqq as qqq_mod
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


def test_gptq_hook_retry_with_same_batch_index_accumulates_once():
    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    gptq = GPTQ(layer)

    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.tasks = {"layer": gptq}
    processor.current_batch_index = lambda: 0
    processor._mask_tls = type("Mask", (), {"value": None})()

    hook = GPTQProcessor.pre_process_fwd_hook(processor, "layer")

    inp = torch.randn(1, 3, 4)
    hook(module=None, inp=(inp,), out=inp)
    # Forward retried after a recoverable OOM re-invokes the same hook.
    hook(module=None, inp=(inp,), out=inp)

    assert gptq.nsamples == 3


def test_eora_accumulate_contribution_skips_duplicate_batch_index():
    processor = EoraProcessor.__new__(EoraProcessor)
    processor.lock = threading.Lock()
    processor._segment_accumulators = {"mod": {}}
    processor._seen_batch_indices = {"mod": set()}

    contribution = torch.ones(2, 2)
    processor._accumulate_eora_contribution(
        name="mod", batch_index=0, batch=1, contribution=contribution, scale=1.0
    )
    # Forward retried after a recoverable OOM re-invokes the same hook.
    processor._accumulate_eora_contribution(
        name="mod", batch_index=0, batch=1, contribution=contribution, scale=1.0
    )

    record = processor._segment_accumulators["mod"][torch.device("cpu")]
    assert record["count"] == 1
    assert torch.equal(record["total"], contribution)


def test_awq_pre_process_fwd_hook_retry_with_same_batch_index_captures_once():
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {}
    processor.current_batch_index = lambda: 0

    hook = AWQProcessor.pre_process_fwd_hook(processor, "mod")

    inp = torch.randn(1, 3, 4)
    hook(module=None, inp=(inp,), out=inp)
    # Forward retried after a recoverable OOM re-invokes the same hook.
    hook(module=None, inp=(inp,), out=inp)

    assert len(processor.tasks["mod"]["inputs"]) == 1


def test_paroquant_pre_process_fwd_hook_retry_with_same_batch_index_captures_once():
    processor = ParoQuantProcessor.__new__(ParoQuantProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {}
    processor.current_batch_index = lambda: 0

    hook = ParoQuantProcessor.pre_process_fwd_hook(processor, "mod")

    inp = torch.randn(1, 3, 4)
    hook(module=None, inp=(inp,), out=inp)
    # Forward retried after a recoverable OOM re-invokes the same hook.
    hook(module=None, inp=(inp,), out=inp)

    assert len(processor.tasks["mod"]["inputs"]) == 1


def test_paroquant_record_input_feature_falls_back_to_current_batch_index_when_omitted():
    """Callers that don't pass batch_index (pre-existing direct callers,
    tests, or external extensions) must keep getting current_batch_index()
    recorded into input_batch_indices, not None.
    """
    processor = ParoQuantProcessor.__new__(ParoQuantProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {}
    processor.current_batch_index = lambda: 7

    processor._record_input_feature("proj", torch.randn(1, 3, 4))

    assert processor.tasks["proj"]["input_batch_indices"] == [7]


def test_gptq_add_batch_rolls_back_reservation_on_failure(monkeypatch):
    """A failure while actually accumulating (after batch_index is reserved)
    must release the reservation -- otherwise a legitimate retry of the same
    batch_index is silently skipped forever instead of ever accumulating.
    """
    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    gptq = GPTQ(layer)

    original_process_batch = GPTQ.process_batch
    calls = {"count": 0}

    def _patched_process_batch(self, inp):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated failure")
        return original_process_batch(self, inp)

    monkeypatch.setattr(GPTQ, "process_batch", _patched_process_batch)

    inp = torch.randn(1, 3, 4)

    with pytest.raises(RuntimeError, match="simulated failure"):
        gptq.add_batch(inp, None, batch_index=0)

    assert gptq.nsamples == 0
    assert 0 not in gptq._seen_batch_indices

    # A genuine retry with the same batch_index must actually accumulate,
    # not be discarded as a duplicate of the failed attempt.
    gptq.add_batch(inp, None, batch_index=0)
    assert gptq.nsamples == 3


def test_paroquant_record_input_feature_rolls_back_reservation_on_failure(monkeypatch):
    processor = ParoQuantProcessor.__new__(ParoQuantProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {}

    original_detach = torch.Tensor.detach
    calls = {"count": 0}

    def _patched_detach(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated failure")
        return original_detach(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "detach", _patched_detach)

    feature = torch.randn(1, 3, 4)

    with pytest.raises(RuntimeError, match="simulated failure"):
        processor._record_input_feature("mod", feature, batch_index=0)

    assert 0 not in processor.tasks["mod"].get("seen_batch_indices", set())
    assert processor.tasks["mod"]["inputs"] == []

    # A genuine retry with the same batch_index must actually capture the
    # feature, not be discarded as a duplicate of the failed attempt.
    processor._record_input_feature("mod", feature, batch_index=0)
    assert len(processor.tasks["mod"]["inputs"]) == 1


def test_eora_accumulate_contribution_rolls_back_on_merge_failure(monkeypatch):
    """A failure partway through merging (after batch_index is reserved) must
    neither corrupt the shared accumulator (scaled without the matching add)
    nor leave batch_index permanently marked seen.
    """
    processor = EoraProcessor.__new__(EoraProcessor)
    processor.lock = threading.Lock()
    processor._segment_accumulators = {"mod": {}}
    processor._seen_batch_indices = {"mod": set()}

    first = torch.ones(2, 2)
    processor._accumulate_eora_contribution(
        name="mod", batch_index=0, batch=1, contribution=first, scale=1.0
    )
    record = processor._segment_accumulators["mod"][torch.device("cpu")]
    original_total = record["total"].clone()

    def _failing_add_(self, *args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(torch.Tensor, "add_", _failing_add_)

    second = torch.full((2, 2), 5.0)
    with pytest.raises(RuntimeError, match="simulated failure"):
        processor._accumulate_eora_contribution(
            name="mod", batch_index=1, batch=1, contribution=second, scale=2.0
        )

    assert torch.equal(record["total"], original_total)
    assert record["count"] == 1
    assert 1 not in processor._seen_batch_indices["mod"]

    monkeypatch.undo()

    # A genuine retry with the same batch_index must actually merge, not be
    # discarded as a duplicate of the failed attempt.
    processor._accumulate_eora_contribution(
        name="mod", batch_index=1, batch=1, contribution=second, scale=2.0
    )
    assert record["count"] == 2
    assert torch.equal(record["total"], torch.full((2, 2), 7.0))


def test_qqq_add_batch_retry_with_same_batch_index_accumulates_once():
    """QQQ is a standalone Hessian accumulator (not a GPTQ subclass) that
    shares ForwardExecutor's retry, so it needs the same batch_index dedupe.
    """
    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    qqq = QQQ(layer)

    inp = torch.randn(1, 3, 4)
    out = torch.empty(0)

    qqq.add_batch(inp, out, batch_index=0)
    nsamples_after_first = qqq.nsamples

    # Forward retried after a recoverable OOM re-invokes the same hook.
    qqq.add_batch(inp, out, batch_index=0)
    assert qqq.nsamples == nsamples_after_first

    qqq.add_batch(inp, out, batch_index=1)
    assert qqq.nsamples == nsamples_after_first * 2


def test_qqq_add_batch_rolls_back_reservation_on_failure(monkeypatch):
    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    qqq = QQQ(layer)

    inp = torch.randn(1, 3, 4)
    out = torch.empty(0)

    original_addmm_ = torch.Tensor.addmm_
    calls = {"count": 0}

    def _patched_addmm_(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated failure")
        return original_addmm_(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "addmm_", _patched_addmm_)

    with pytest.raises(RuntimeError, match="simulated failure"):
        qqq.add_batch(inp, out, batch_index=0)

    assert qqq.nsamples == 0
    assert 0 not in qqq._seen_batch_indices

    # A genuine retry with the same batch_index must actually accumulate,
    # not be discarded as a duplicate of the failed attempt.
    qqq.add_batch(inp, out, batch_index=0)
    assert qqq.nsamples == 3


def test_native_processor_pre_process_fwd_hook_retry_with_same_batch_index_captures_once():
    processor = NativeProcessor.__new__(NativeProcessor)
    processor.lock = threading.Lock()
    processor.native_inp_caches = {"mod": []}
    processor._seen_batch_indices = {"mod": set()}
    processor.current_batch_index = lambda: 0
    processor.qcfg = types.SimpleNamespace(gptaq=types.SimpleNamespace(device="cpu"), foem=None)

    hook = NativeProcessor.pre_process_fwd_hook(processor, "mod")

    inp = torch.randn(1, 3, 4)
    hook(module=None, inp=(inp,), out=inp)
    # Forward retried after a recoverable OOM re-invokes the same hook.
    hook(module=None, inp=(inp,), out=inp)

    assert len(processor.native_inp_caches["mod"]) == 1


def test_native_processor_finalize_releases_seen_batch_indices():
    processor = NativeProcessor.__new__(NativeProcessor)
    processor.native_inp_caches = {"mod": []}
    processor._seen_batch_indices = {"mod": {0, 1}}

    processor.finalize(model=None)

    assert not hasattr(processor, "native_inp_caches")
    assert not hasattr(processor, "_seen_batch_indices")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CPU fallback regression coverage")
def test_qqq_materialize_hessian_falls_back_to_cpu_on_oom(monkeypatch):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()
    qqq = QQQ(layer)
    inp = torch.randn(1, 3, 4, device=device)
    out = torch.empty(0, device=device)
    qqq.add_batch(inp, out, batch_index=0)

    original_impl = QQQ._materialize_hessian_on_device
    calls = []

    def _patched(self, target_device):
        calls.append(target_device)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. simulated for regression test")
        return original_impl(self, target_device)

    monkeypatch.setattr(QQQ, "_materialize_hessian_on_device", _patched)

    log_messages = []
    monkeypatch.setattr(
        qqq_mod.log,
        "warn",
        lambda message, *args, **kwargs: log_messages.append(message % args if args else message),
    )

    H = qqq.materialize_hessian()

    assert len(calls) == 2
    assert calls[1] == torch.device("cpu")
    assert H.device == torch.device("cpu")
    joined_logs = "\n".join(log_messages)
    assert "falling back to CPU" in joined_logs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CPU fallback regression coverage")
def test_qqq_materialize_hessian_restores_partials_on_merge_failure(monkeypatch):
    """A merge failure partway through must not silently drop whatever
    partial was already popped-and-merged before it -- the OOM-fallback
    retry rebuilds the Hessian from scratch, so every partial (not just the
    one that failed) has to still be present in _device_hessian_partials.
    """
    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    qqq = QQQ(layer)

    device = torch.device("cuda", 0)
    cpu = torch.device("cpu")
    partial_cuda = torch.eye(4, dtype=torch.float32, device=device)
    partial_cpu = torch.eye(4, dtype=torch.float32, device=cpu) * 2.0
    # Insertion order matters: dict.popitem() is LIFO, so `device` (inserted
    # last) is popped -- and merged successfully, no .to() needed -- before
    # `cpu` is popped and fails.
    qqq._device_hessian_partials = {cpu: partial_cpu, device: partial_cuda}
    qqq._device_sample_counts = {device: 3, cpu: 5}

    original_to = torch.Tensor.to
    state = {"failed": False}

    def _patched_to(self, *args, **kwargs):
        if not state["failed"] and kwargs.get("device") == device and kwargs.get("dtype") == torch.float32:
            state["failed"] = True
            raise RuntimeError("CUDA out of memory. simulated for regression test")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _patched_to)

    with pytest.raises(RuntimeError, match="out of memory"):
        qqq._materialize_hessian_on_device(device)

    assert state["failed"]
    assert set(qqq._device_hessian_partials.keys()) == {device, cpu}
    assert qqq._device_sample_counts == {device: 3, cpu: 5}

    monkeypatch.undo()

    H = qqq.materialize_hessian(target_device=device)
    expected = (partial_cuda + partial_cpu.to(device)) * (2.0 / 8.0)
    assert torch.allclose(H, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CPU fallback regression coverage")
def test_qqq_quantize_completes_after_hessian_cpu_fallback(monkeypatch):
    """materialize_hessian() falling back to CPU must not leave self.H on CPU
    while the rest of quantize() (W clone, dead-column masking, damping,
    Cholesky, the block loop) runs on the module's original GPU device --
    that mismatch previously crashed quantize() right after the fallback
    had "succeeded".
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    layer = nn.Linear(8, 8, bias=False, dtype=torch.float32).to(device).eval()
    qqq = QQQ(layer)
    qqq.quantizer.configure(bits=4, perchannel=True, sym=True, groupsize=-1)
    inp = torch.randn(1, 4, 8, device=device)
    out = torch.empty(0, device=device)
    qqq.add_batch(inp, out, batch_index=0)

    original_impl = QQQ._materialize_hessian_on_device
    calls = []

    def _patched(self, target_device):
        calls.append(target_device)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. simulated for regression test")
        return original_impl(self, target_device)

    monkeypatch.setattr(QQQ, "_materialize_hessian_on_device", _patched)

    Q, *_ = qqq.quantize(blocksize=4)

    assert calls[1] == torch.device("cpu")
    assert Q.device == device
    assert torch.isfinite(Q).all()
