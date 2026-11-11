# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The modules of the MiniMax-H3 scenario ``minimax_h3_nvfp4__sm_107__u1_u4`` hold the process's one
configuration object (SM 107 only: the modules build their kernels at import)."""

import importlib

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
    config as scenario_config,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or tuple(torch.cuda.get_device_capability()) != (10, 7),
    reason="the scenario's kernels are SM 107 only",
)

_PACKAGE = scenario_config.__name__.rsplit(".", 1)[0]
_CONSUMERS = (
    "kernels.fused_fp8_qk",
    "kernels.fused_out_a2a",
    "kernels.fused_out_quant",
    "kernels.fused_quant",
    "kernels.vae_decoder_fp8",
    "modeling",
    "pipeline",
)


@pytest.mark.parametrize("name", _CONSUMERS)
def test_consumer_holds_the_selected_config(name):
    module = importlib.import_module(f"{_PACKAGE}.{name}")
    assert module.SCENARIO is scenario_config.SCENARIO


def test_differing_selection_is_refused_once_consumers_are_loaded():
    for name in _CONSUMERS:
        importlib.import_module(f"{_PACKAGE}.{name}")
    in_use = scenario_config.SCENARIO
    with pytest.raises(RuntimeError, match="one scenario configuration per process"):
        scenario_config.apply({"out_quant": not in_use.out_quant})
    assert scenario_config.SCENARIO is in_use
