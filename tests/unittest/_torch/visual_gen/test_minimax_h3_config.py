# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MiniMax-H3 switches in ``VisualGenArgs.pipeline_config``: defaults, validation, registry and
plumbing into the internal configs."""

import pytest
import yaml
from pydantic import ValidationError

import tensorrt_llm._torch.visual_gen.models  # noqa: F401  registers the pipelines
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY
from tensorrt_llm.visual_gen.args import MiniMaxH3Config, VisualGenArgs

pytestmark = pytest.mark.cpu_only


def test_defaults_are_the_recipe():
    cfg = MiniMaxH3Config()
    assert cfg.fused_quant and cfg.out_quant
    assert cfg.fused_quant_warps == 4
    assert cfg.fp8_scale_fusion == "dynamic" and cfg.fp8_fusion_split
    assert (cfg.fp8_fusion_block_h, cfg.fp8_fusion_warps) == (4, 2)
    assert cfg.fp8_static_scales is None and cfg.fp8_static_margin == 1.0
    assert cfg.cudnn_prefix_inplace and cfg.schedule and cfg.vae_fp8
    assert cfg.vae_tile_geometry == (288, 272, 32)
    assert not cfg.ulysses_fp8_payload


def test_group_from_pipeline_config_yaml():
    doc = yaml.safe_load(
        """
        attention_config:
          backend: CUDNN
          quant_attention_config: {qk_dtype: fp8, v_dtype: fp8}
        pipeline_config:
          fused_quant: false
          out_quant: false
          vae_tile: 256x256
        """
    )
    args = VisualGenArgs(model="x", **doc)
    entry = PIPELINE_REGISTRY["MiniMaxH3ModularPipeline"]
    resolved = {**entry.defaults, **args.pipeline_config}  # what PipelineLoader hands over
    cfg = MiniMaxH3Config.from_pipeline_config(resolved)
    assert not cfg.fused_quant and not cfg.out_quant
    assert cfg.fp8_scale_fusion == "dynamic"
    assert cfg.vae_tile_geometry == (256, 256, 32)
    assert resolved["workflow"] == "fl2va"  # other keys of the entry are untouched


def test_registry_defaults_carry_every_field():
    """The registry entry is the legal key set of pipeline_config; it must list every field
    with the recipe default, so an unknown key fails at load and a bare YAML runs the recipe."""
    defaults = PIPELINE_REGISTRY["MiniMaxH3ModularPipeline"].defaults
    assert set(MiniMaxH3Config.model_fields) <= set(defaults)
    assert MiniMaxH3Config.from_pipeline_config(defaults) == MiniMaxH3Config()


def test_other_pipelines_get_the_defaults():
    assert MiniMaxH3Config.from_pipeline_config({"text_encoder_path": ""}) == MiniMaxH3Config()


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        MiniMaxH3Config(fused_quant_warp=4)
    with pytest.raises(ValidationError):
        MiniMaxH3Config.from_pipeline_config({"fused_quant": "sometimes"})


def test_static_needs_scales():
    with pytest.raises(ValidationError, match="fp8_static_scales"):
        MiniMaxH3Config(fp8_scale_fusion="static")
    cfg = MiniMaxH3Config(fp8_scale_fusion="static", fp8_static_scales="/tmp/scales.json")
    assert cfg.fp8_scale_fusion == "static"


@pytest.mark.parametrize("tile", ["288", "288x", "ax272", "288x272x32x4", "0x272"])
def test_vae_tile_format(tile):
    with pytest.raises(ValidationError, match="vae_tile"):
        MiniMaxH3Config(vae_tile=tile)


def test_vae_tile_none_keeps_decoder_tiling():
    assert MiniMaxH3Config(vae_tile=None).vae_tile_geometry is None


def test_group_copied_into_model_configs():
    pipeline_config = DiffusionPipelineConfig(minimax_h3=MiniMaxH3Config(fused_quant=False))
    model_config = pipeline_config._make_model_config("transformer", None)
    assert model_config.minimax_h3 == pipeline_config.minimax_h3
    assert not model_config.minimax_h3.fused_quant
