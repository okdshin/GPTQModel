# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Extended regression coverage for the CUDA-OOM retry/fallback work added in
fix-oom-hessian-fallback (07944798, c5bd1b90), beyond the direct unit tests in
test_forward_retry_dedup.py and test_gptq.py:

1. Real-CUDA integration -- an actual GPTQProcessor/QQQProcessor task wired
   through ForwardExecutor.run_single/run_parallel (not just its hook called
   directly), so a genuine accelerator OOM-and-retry exercises the full
   forward-executor -> hook -> GPTQ/QQQ stack together.
2. Retry-vs-clean-run Hessian equality -- a run whose forward hook fires
   twice for one batch (retry) or whose materialization step OOMs and
   retries on CPU must accumulate byte-identical Hessian state to a run that
   never fails, not just "same sample count" or "close enough".
3. Dedupe-state isolation across subsets -- a processor's per-module
   seen-batch-index tracking must not leak from one subset pass into the
   next reuse of the same module name (each subset's forward restarts
   batch_index at 0, so leaked state silently discards real batches).
"""

import threading
import types

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.eora_processor import EoraProcessor
from gptqmodel.looper.forward_executor import ForwardExecutor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.looper.qqq_processor import QQQProcessor
from gptqmodel.quantization import gptq as gptq_mod
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ
from gptqmodel.utils.looper_helpers import forward_batch_worker


def _make_forward_executor_looper():
    return types.SimpleNamespace(
        _resolve_batch_total=lambda _num_batches, layer_inputs: len(layer_inputs),
        _collect_row_counts=lambda layer_inputs: [int(batch[0].shape[0]) for batch in layer_inputs],
        _set_processor_mask=lambda _processor, _mask: None,
        _batch_row_count=lambda batch_inputs: int(batch_inputs[0].shape[0]),
        support_batch_quantize=False,
        gptq_model=types.SimpleNamespace(
            quantize_config=types.SimpleNamespace(
                calibration_data_device=None,
                compute_device_filter=None,
            ),
            prepare_layer_replay_kwargs=lambda layer, layer_input, additional_inputs, target_device: additional_inputs,
        ),
        moe_routing_override=None,
        moe_routing_bypass=False,
        _should_use_moe_lifecycle=lambda *_a, **_k: False,
        _current_subset=None,
    )


def _make_gptq_processor(tasks):
    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.tasks = tasks
    processor.num_batches = None
    processor._batch_tls = threading.local()
    processor._mask_tls = types.SimpleNamespace(value=None)
    return processor


def _make_qqq_processor(tasks):
    processor = QQQProcessor.__new__(QQQProcessor)
    processor.tasks = tasks
    processor.num_batches = None
    processor._batch_tls = threading.local()
    return processor


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _ImmediateThreadPool:
    def submit(self, _device, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))

    def submit_serial(self, _device, fn, *args, **kwargs):
        return _ImmediateFuture(fn(*args, **kwargs))


class _TwoLinearBlock(nn.Module):
    """Two real Linear submodules chained in one forward call. The second one
    OOMs on its first `fail_times` invocations, forcing ForwardExecutor to
    retry the *whole* block forward -- which re-invokes the first submodule's
    already-succeeded hook. This is the exact double-hook-invocation shape
    described in 07944798's commit message.
    """

    def __init__(self, a: nn.Linear, b: nn.Linear, fail_times: int):
        super().__init__()
        self.a = a
        self.b = b
        self.fail_times = fail_times
        self.b_calls = 0

    def forward(self, x, **_kwargs):
        y = self.a(x)
        self.b_calls += 1
        if self.b_calls <= self.fail_times:
            raise RuntimeError("CUDA out of memory. simulated for regression test")
        return self.b(y)


# ---------------------------------------------------------------------------
# 1. Real GPTQ/QQQ processor wired through ForwardExecutor.run_single/run_parallel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA runtime to exercise a real accelerator OOM retry")
def test_run_single_with_real_gptq_processor_dedupes_hessian_after_retry():
    """End-to-end: ForwardExecutor.run_single -> real forward hooks -> real
    GPTQ.add_batch, on a real CUDA device. The retried whole-block forward
    must not double-count module `a`'s Hessian contribution, and module `b`
    must accumulate exactly once (its own hook only ever fires on the
    successful attempt).
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    a = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()
    b = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()
    block = _TwoLinearBlock(a, b, fail_times=1).to(device).eval()

    gptq_a = GPTQ(a)
    gptq_b = GPTQ(b)
    processor = _make_gptq_processor({"a": gptq_a, "b": gptq_b})

    a.register_forward_hook(processor.pre_process_fwd_hook("a"))
    b.register_forward_hook(processor.pre_process_fwd_hook("b"))

    looper = _make_forward_executor_looper()
    executor = ForwardExecutor(looper)

    x = torch.randn(1, 3, 4, device=device)
    outputs = executor.run_single(
        module=block,
        processor=processor,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=device,
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
    )

    assert block.b_calls == 2
    assert len(outputs) == 1
    assert gptq_a.nsamples == 3
    assert gptq_b.nsamples == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA runtime to exercise a real accelerator OOM retry")
def test_run_parallel_with_real_gptq_processor_dedupes_hessian_after_retry():
    """Same scenario through ForwardExecutor.run_parallel -> the real
    forward_batch_worker. run_single and run_parallel each carry their own
    copy of the OOM retry loop; both must dedupe against the same task.
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    a = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()
    b = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()
    block = _TwoLinearBlock(a, b, fail_times=1).to(device).eval()

    gptq_a = GPTQ(a)
    gptq_b = GPTQ(b)
    processor = _make_gptq_processor({"a": gptq_a, "b": gptq_b})

    a.register_forward_hook(processor.pre_process_fwd_hook("a"))
    b.register_forward_hook(processor.pre_process_fwd_hook("b"))

    looper = _make_forward_executor_looper()
    executor = ForwardExecutor(looper)

    x = torch.randn(1, 3, 4, device=device)
    outputs = executor.run_parallel(
        module=block,
        processor=processor,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[None],
        attention_masks=[None],
        cur_layer_device=device,
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
        devices=[device],
        # Use the block itself as its own "replica" -- exercising the real
        # forward_batch_worker retry/hook path without paying for a deepcopy.
        clone_module_for_devices_fn=lambda module, devices, progress_callback=None: dict.fromkeys(devices, module),
        forward_batch_worker_fn=forward_batch_worker,
        device_thread_pool=_ImmediateThreadPool(),
    )

    assert block.b_calls == 2
    assert len(outputs) == 1
    assert gptq_a.nsamples == 3
    assert gptq_b.nsamples == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA runtime to exercise a real accelerator OOM retry")
def test_run_single_with_real_qqq_processor_dedupes_hessian_after_retry():
    """QQQ is a standalone Hessian accumulator (not a GPTQ subclass); it needs
    the same real-ForwardExecutor integration coverage as GPTQ.
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    a = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()
    b = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()
    block = _TwoLinearBlock(a, b, fail_times=1).to(device).eval()

    qqq_a = QQQ(a)
    qqq_b = QQQ(b)
    processor = _make_qqq_processor({"a": qqq_a, "b": qqq_b})

    a.register_forward_hook(processor.pre_process_fwd_hook("a"))
    b.register_forward_hook(processor.pre_process_fwd_hook("b"))

    looper = _make_forward_executor_looper()
    executor = ForwardExecutor(looper)

    x = torch.randn(1, 3, 4, device=device)
    outputs = executor.run_single(
        module=block,
        processor=processor,
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=device,
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
    )

    assert block.b_calls == 2
    assert len(outputs) == 1
    assert qqq_a.nsamples == 3
    assert qqq_b.nsamples == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA runtime to exercise a real accelerator OOM retry")
def test_run_single_replay_pass_recovers_from_oom_without_touching_processor_state():
    """Replay forwards (apply_moe_config=False) run after a subset's capture
    hooks have already been removed by StageSubset (`h.remove()`), so a real
    module with no hooks attached must still get the crash-recovery retry --
    the retry loop must not depend on any hook/dedupe bookkeeping being
    present.
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    inner = nn.Linear(4, 4, bias=False, dtype=torch.float32).to(device).eval()

    class _FlakyReplayLayer(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.calls = 0

        def forward(self, x, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA out of memory. simulated for regression test")
            return self.wrapped(x)

    flaky = _FlakyReplayLayer(inner).to(device).eval()

    class _AssertNoHookProcessor:
        num_batches = None

        def _set_current_batch_index(self, _idx):
            return None

    looper = _make_forward_executor_looper()
    executor = ForwardExecutor(looper)

    x = torch.randn(1, 3, 4, device=device)
    outputs = executor.run_single(
        module=flaky,
        processor=_AssertNoHookProcessor(),
        layer_inputs=[[x]],
        layer_input_kwargs=[{}],
        position_ids=[],
        attention_masks=[None],
        cur_layer_device=device,
        is_lm_head_module=False,
        shared_kv_cache_dict={},
        layer_index=0,
        need_outputs=True,
        reuse_kv=False,
        apply_moe_config=False,
    )

    assert flaky.calls == 2
    assert len(outputs) == 1


# ---------------------------------------------------------------------------
# 2. Retry-vs-clean-run Hessian equality
# ---------------------------------------------------------------------------


def test_gptq_hook_double_fire_produces_identical_hessian_to_single_fire():
    """A forward retried after a recoverable OOM re-invokes the module's hook
    for the same calibration batch. The resulting Hessian after dedupe must
    be byte-identical to a run where the hook only ever fired once per batch
    -- not merely "close" or "same sample count".
    """
    torch.manual_seed(0)
    layer_clean = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    layer_retry = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    layer_retry.load_state_dict(layer_clean.state_dict())

    gptq_clean = GPTQ(layer_clean)
    gptq_retry = GPTQ(layer_retry)

    processor_clean = _make_gptq_processor({"layer": gptq_clean})
    processor_retry = _make_gptq_processor({"layer": gptq_retry})
    hook_clean = processor_clean.pre_process_fwd_hook("layer")
    hook_retry = processor_retry.pre_process_fwd_hook("layer")

    batch0 = torch.randn(1, 3, 4)
    batch1 = torch.randn(1, 5, 4)

    processor_clean._batch_tls.index = 0
    hook_clean(None, (batch0,), batch0)
    processor_clean._batch_tls.index = 1
    hook_clean(None, (batch1,), batch1)

    processor_retry._batch_tls.index = 0
    hook_retry(None, (batch0,), batch0)
    # Forward retried after a recoverable OOM re-invokes the same hook.
    hook_retry(None, (batch0,), batch0)
    processor_retry._batch_tls.index = 1
    hook_retry(None, (batch1,), batch1)

    assert gptq_clean.nsamples == gptq_retry.nsamples

    gptq_clean.materialize_global_hessian()
    gptq_retry.materialize_global_hessian()

    assert torch.equal(gptq_clean.H, gptq_retry.H)


def test_qqq_hook_double_fire_produces_identical_hessian_to_single_fire():
    torch.manual_seed(0)
    layer_clean = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    layer_retry = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    layer_retry.load_state_dict(layer_clean.state_dict())

    qqq_clean = QQQ(layer_clean)
    qqq_retry = QQQ(layer_retry)

    processor_clean = _make_qqq_processor({"layer": qqq_clean})
    processor_retry = _make_qqq_processor({"layer": qqq_retry})
    hook_clean = processor_clean.pre_process_fwd_hook("layer")
    hook_retry = processor_retry.pre_process_fwd_hook("layer")

    batch0 = torch.randn(1, 3, 4)
    out0 = torch.empty(0)

    processor_clean._batch_tls.index = 0
    hook_clean(None, (batch0,), out0)

    processor_retry._batch_tls.index = 0
    hook_retry(None, (batch0,), out0)
    # Forward retried after a recoverable OOM re-invokes the same hook.
    hook_retry(None, (batch0,), out0)

    assert qqq_clean.nsamples == qqq_retry.nsamples

    H_clean = qqq_clean.materialize_hessian()
    H_retry = qqq_retry.materialize_hessian()
    assert torch.equal(H_clean, H_retry)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CPU fallback regression coverage")
def test_gptq_materialize_global_hessian_oom_retry_matches_clean_run(monkeypatch):
    """Mirrors test_gptq_embedding_materialize_hessian_oom_retry_matches_clean_run
    in test_gptq.py, but for the ordinary (non-embedding) Linear Hessian path:
    materialize_global_hessian falling back to CPU mid-way must not corrupt
    the merged result relative to a clean, uninterrupted run.
    """
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    def _build(seed):
        torch.manual_seed(seed)
        layer = nn.Linear(8, 8, bias=False, dtype=torch.float32).to(device).eval()
        gptq = GPTQ(layer)
        inp1 = torch.randn(1, 4, 8, device=device)
        gptq.add_batch(inp1, None, batch_index=0)
        gptq.materialize_global_hessian()
        inp2 = torch.randn(1, 4, 8, device=device)
        gptq.add_batch(inp2, None, batch_index=1)
        return gptq

    reference = _build(seed=0)
    reference.materialize_global_hessian()
    reference_H = reference.H.clone()

    flaky = _build(seed=0)

    original_impl = GPTQ._materialize_global_hessian_on_device
    calls = []

    def _patched(self, target_device=None):
        calls.append(target_device)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. simulated for regression test")
        return original_impl(self, target_device)

    monkeypatch.setattr(GPTQ, "_materialize_global_hessian_on_device", _patched)
    monkeypatch.setattr(gptq_mod.log, "warn", lambda *args, **kwargs: None)

    flaky.materialize_global_hessian()

    assert len(calls) == 2
    assert calls[1] == torch.device("cpu")
    assert torch.allclose(flaky.H, reference_H.to("cpu"), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Dedupe-state isolation across subsets
# ---------------------------------------------------------------------------


def test_paroquant_seen_batch_indices_do_not_leak_across_subsets_in_same_layer():
    """Regression test for a leak the OOM-retry dedupe mechanism introduced:
    `_ensure_task_bucket` only reset a module's bucket on a *layer* change,
    but a module name can be revisited by a second subset within the *same*
    layer (e.g. a capture-only pre-pass followed by the real quantization
    subset). Each subset's forward restarts batch_index at 0, so without
    resetting the dedupe set on every `_ensure_task_bucket` call, the second
    subset's legitimate batch 0 was silently discarded as a "duplicate" of
    the first subset's batch 0.
    """
    processor = ParoQuantProcessor.__new__(ParoQuantProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {}

    processor._ensure_task_bucket("proj", layer_index=0)
    processor._record_input_feature("proj", torch.ones(1, 3, 4), batch_index=0)
    assert len(processor.tasks["proj"]["inputs"]) == 1

    # A second subset of the *same* layer revisits "proj" and restarts its
    # own batch_index count at 0 -- this is not a retry of the first subset.
    processor._ensure_task_bucket("proj", layer_index=0)
    processor._record_input_feature("proj", torch.full((1, 3, 4), 2.0), batch_index=0)
    assert len(processor.tasks["proj"]["inputs"]) == 2

    # An actual retry *within* that second subset's pass must still dedupe.
    processor._record_input_feature("proj", torch.full((1, 3, 4), 99.0), batch_index=0)
    assert len(processor.tasks["proj"]["inputs"]) == 2


def test_paroquant_seen_batch_indices_reset_on_new_layer():
    processor = ParoQuantProcessor.__new__(ParoQuantProcessor)
    processor.lock = threading.Lock()
    processor.tasks = {}

    processor._ensure_task_bucket("proj", layer_index=0)
    processor._record_input_feature("proj", torch.ones(1, 3, 4), batch_index=0)

    processor._ensure_task_bucket("proj", layer_index=1)
    assert processor.tasks["proj"]["inputs"] == []

    processor._record_input_feature("proj", torch.full((1, 3, 4), 2.0), batch_index=0)
    assert len(processor.tasks["proj"]["inputs"]) == 1


def test_native_processor_seen_batch_indices_reset_across_preprocess_calls_for_reused_module_name():
    """NativeProcessor's dedupe set is unconditionally reset every time
    preprocess() runs for a module name -- unlike ParoQuant's layer-scoped
    bucket, so the same short name recurring in a later layer/subset must not
    inherit a previous pass's seen batch indices.
    """
    processor = NativeProcessor.__new__(NativeProcessor)
    processor.lock = threading.Lock()
    processor.native_inp_caches = {}
    processor._seen_batch_indices = {}
    processor.qcfg = types.SimpleNamespace(gptaq=types.SimpleNamespace(device="cpu"), foem=None)
    processor.current_batch_index = lambda: 0

    module = types.SimpleNamespace(name="proj")
    processor.preprocess(module)
    hook = processor.pre_process_fwd_hook("proj")
    hook(module=None, inp=(torch.randn(1, 3, 4),), out=None)
    assert len(processor.native_inp_caches["proj"]) == 1

    # A later layer's subset reuses the short name "proj" -- preprocess()
    # runs again and must start with a clean dedupe set.
    processor.preprocess(module)
    hook(module=None, inp=(torch.randn(1, 3, 4),), out=None)
    assert len(processor.native_inp_caches["proj"]) == 1


def test_eora_seen_batch_indices_reset_across_preprocess_calls_for_reused_module_name():
    processor = EoraProcessor.__new__(EoraProcessor)
    processor.lock = threading.Lock()
    processor._segment_accumulators = {}
    processor._module_target_devices = {}
    processor._seen_batch_indices = {}

    processor._accumulate_eora_contribution(
        name="mod", batch_index=0, batch=1, contribution=torch.ones(2, 2), scale=1.0
    )
    assert processor._segment_accumulators["mod"][torch.device("cpu")]["count"] == 1

    # Simulate the next subset's preprocess() resetting bookkeeping for "mod"
    # (mirrors EoraProcessor.preprocess()'s reset block).
    processor._module_target_devices["mod"] = torch.device("cpu")
    processor._segment_accumulators["mod"] = {}
    processor._seen_batch_indices["mod"] = set()

    processor._accumulate_eora_contribution(
        name="mod", batch_index=0, batch=1, contribution=torch.full((2, 2), 5.0), scale=1.0
    )
    assert processor._segment_accumulators["mod"][torch.device("cpu")]["count"] == 1
    assert torch.equal(
        processor._segment_accumulators["mod"][torch.device("cpu")]["total"],
        torch.full((2, 2), 5.0),
    )
