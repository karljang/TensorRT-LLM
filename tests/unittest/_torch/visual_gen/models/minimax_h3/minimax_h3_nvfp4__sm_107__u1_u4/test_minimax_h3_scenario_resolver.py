# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing and settings of the MiniMax-H3 scenario ``minimax_h3_nvfp4__sm_107__u1_u4`` (CPU only)."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import scenarios
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import config as scenario_config
from tensorrt_llm.quantization.mode import QuantAlgo


class _Family:
    """Stands in for MiniMaxH3Pipeline: the resolver only hands the class back."""


def _config(*, quant_algo=QuantAlgo.NVFP4, dynamic=False, backend="CUDNN", fp8=True, ulysses=1, tp=1,
            async_ulysses=False, mode="auto", overrides=None):
    qac = SimpleNamespace(qk_dtype="fp8", v_dtype="fp8") if fp8 else None
    extra = {"visual_gen_scenario": mode}
    if overrides is not None:
        extra["minimax_h3_scenario"] = overrides
    return SimpleNamespace(
        model_configs={"transformer": SimpleNamespace(quant_config=SimpleNamespace(quant_algo=quant_algo))},
        quant_config=SimpleNamespace(quant_algo=quant_algo),
        dynamic_weight_quant=dynamic,
        attention=SimpleNamespace(backend=backend, quant_attention_config=qac),
        visual_gen_mapping=SimpleNamespace(ulysses_size=ulysses),
        mapping=SimpleNamespace(tp_size=tp),
        parallel=SimpleNamespace(async_ulysses=async_ulysses),
        extra_attrs=extra,
    )


def test_off_keeps_the_family_and_imports_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(scenarios, "ROUTES", (lambda cfg: called.append(cfg),))
    assert scenarios.resolve_scenario(_Family, _config(mode="off")) is _Family
    assert called == []


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_route_declines_without_sm_107(monkeypatch, mode):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 3))
    if mode == "auto":
        assert scenarios.resolve_scenario(_Family, _config(mode="auto")) is _Family
    else:
        with pytest.raises(RuntimeError, match="SM 107"):
            scenarios.resolve_scenario(_Family, _config(mode="require"))


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (dict(quant_algo=QuantAlgo.FP8), "NVFP4"),
        (dict(dynamic=True), "dynamic_weight_quant"),
        (dict(backend="FA4"), "CUDNN"),
        (dict(fp8=False), "fp8"),
        (dict(ulysses=2), "Ulysses 1 or 4"),
        (dict(tp=2), "tp_size"),
        (dict(async_ulysses=True), "async_ulysses"),
    ],
)
def test_route_names_the_failed_requirement(monkeypatch, kwargs, reason):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    with pytest.raises(scenarios.NoScenario, match=reason):
        scenarios.route_nvfp4_sm_107_u1_u4(_config(**kwargs))


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="visual_gen_scenario"):
        scenarios.resolve_scenario(_Family, _config(mode="sometimes"))


def test_config_defaults_are_the_recipe():
    cfg = scenario_config.ScenarioConfig()
    cfg.validate()
    assert cfg.attention_fmha == "trtllm-gen-fp16sm"
    assert cfg.fp8_scale_fusion == "qkstatic"
    assert cfg.skip_corr_threshold == 1.0
    assert cfg.grouped_cast and cfg.vamax_fold and cfg.fused_quant and cfg.out_quant
    assert cfg.resolved_vae_fp8_qtile(1) == (32, 8, 8)
    assert cfg.resolved_vae_fp8_qtile(4) == (64, 4, 8)


def test_apply_overrides_and_validates():
    before = scenario_config.SCENARIO
    try:
        cfg = scenario_config.apply({"skip_corr_threshold": 0.0, "vae_fp8_qtile": [16, 4, 4]})
        assert cfg.skip_corr_threshold == 0.0 and cfg.vae_fp8_qtile == (16, 4, 4)
        assert scenario_config.SCENARIO is cfg
        with pytest.raises(ValueError, match="unknown settings"):
            scenario_config.apply({"no_such_switch": 1})
        with pytest.raises(ValueError, match="qkstatic"):
            scenario_config.apply({"fp8_scale_fusion": "dynamic"})
    finally:
        scenario_config.SCENARIO = before
