# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.quantization.gptq import GPTQ


def test_pre_process_fwd_hook_masked_multi_sample_batch_accumulates_every_kept_sample():
    """A masked batch_size>1 forward calls add_batch once per kept sample with
    the same forward's batch index. GPTQ's retry-dedupe must not treat those
    per-sample calls as duplicates of each other and drop all but the first.
    """

    layer = nn.Linear(4, 4, bias=False, dtype=torch.float32).eval()
    gptq = GPTQ(layer)

    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.tasks = {"layer": gptq}
    processor.current_batch_index = lambda: 0
    processor._mask_tls = type("Mask", (), {"value": None})()

    hook = GPTQProcessor.pre_process_fwd_hook(processor, "layer")

    inp = torch.randn(2, 3, 4)
    processor._mask_tls.value = torch.tensor([[True, True, True], [True, True, True]])

    hook(module=None, inp=(inp,), out=inp)

    # 2 samples x 3 kept tokens each -- both samples must be counted, not just
    # the first one that add_batch's batch_index dedupe sees.
    assert gptq.nsamples == 6
