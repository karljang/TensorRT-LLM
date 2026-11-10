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
"""Settings of the ``minimax_h3_nvfp4__sm_107__u1_u4`` scenario.

The defaults are the measured recipe (board 692, iterations 1 to 4 of round 2 and the F.1 rider): every
value here replaced an environment switch of the study branch. ``SCENARIO`` is read by the scenario's
kernels at import time and by its modeling and pipeline classes, so :func:`apply` must run before the
scenario package is imported, which ``models/minimax_h3/scenarios.py`` guarantees. A caller overrides
settings for an A/B through ``pipeline_config.extra_attrs["minimax_h3_scenario"]`` (a mapping of field
name to value); the family pipeline and the shared code never read these.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping, Optional

Tile3 = tuple[int, int, int]


@dataclass
class ScenarioConfig:
    # --- attention (the trtllm-gen dense FP8 FMHA on pre-quantized Q / K / V) --------------------
    # Kernel for the pre-quantized FP8 problem (cuDNN backend option ``fp8_dense_fmha``): the
    # Fp16Softmax SM107a cubin with the softmax scale folded into the Q / K scales.
    attention_fmha: str = "trtllm-gen-fp16sm"
    # Deferred output rescale threshold t (the kernel skips the O rescale while the running row max
    # has grown by at most 2^t; E4M3 BMM2 then uses a P scale of 1.75). Bit-identical to 0 at t = 1
    # on the stack's reference clip; 0 = rescale every step.
    skip_corr_threshold: float = 1.0
    # Ulysses > 1: the op reads Q / K / V in place from the [S, 3, H / U, D] receive buffer.
    strided_receive: bool = True
    # One packed FP8 all-to-all for Q / K / V at Ulysses > 1, and the attention output exchanged as
    # NVFP4 codes + scales for the static-NVFP4 ``to_out``.
    packed_a2a: bool = True
    out_a2a_fp4: bool = True

    # --- the fused QK-norm + RoPE + FP8 cast producer ---------------------------------------------
    # "qkstatic": static Q / K scales that fold sm_scale * log2 e (the FP16-softmax kernel runs with a
    # unit scale), V with its dynamic per-tensor amax; "static" / "dynamic" are the recipe's other modes.
    fp8_scale_fusion: str = "qkstatic"
    fp8_static_scales: Optional[str] = None  # a descale table for fp8_scale_fusion="static"
    fp8_static_margin: float = 1.0
    fp8_fusion_split: bool = True  # Q / K and V in separate launches (the recipe's split)
    fp8_fusion_block_h: int = 4
    fp8_fusion_warps: int = 2
    grouped_cast: bool = True  # one grouped Q / K cast launch (board 692 N.3)
    vamax_fold: bool = True  # V's amax taken inside the grouped cast (board 692 F.1 (a))
    # The per-tensor amax reduce over the Ulysses group: "symm" (symmetric memory, board 692 X.1) | "nccl".
    amax_allreduce: str = "symm"

    # --- the NVFP4 producers (norm + modulation, SwiGLU, attention output) -----------------------
    fused_quant: bool = True  # the norm / SwiGLU producers write the Linears' NVFP4 inputs directly
    fused_quant_warps: int = 4
    swiglu_tile: Optional[Tile3] = (8, 512, 4)
    out_quant: bool = True  # lever 17: the attention output quantized by the tiled kernel
    out_quant_tile: Tile3 = (16, 512, 4)

    # --- outside the denoise loop ----------------------------------------------------------------
    vae_fp8: bool = True  # the VAE ViT decoder's Linears as FP8 per-tensor GEMMs
    vae_fp8_tiled: bool = True  # tiled SwiGLU / attention-output quantize in that decoder
    vae_fused_norm: bool = True  # shared q / k / v quantize, one-kernel q / k norm + RoPE (board 692 V.1 (a))
    vae_fp8_tile: Tile3 = (16, 1024, 8)
    vae_fp8_qtile: Optional[Tile3] = None  # None: (32, 8, 8) at Ulysses 1, (64, 4, 8) at Ulysses 4
    vae_tile: Tile3 = (288, 272, 32)  # decoder spatial tiles (height x width x min overlap, pixels)
    vae_graph_max_shapes: int = 2  # decoder call replayed from a CUDA graph per chunk shape (0 = off)
    text_encoder_graph: bool = True  # text-only prefill replayed from a CUDA graph (board 692 V.1 (b1))
    text_encoder_graph_max: int = 8

    def resolved_vae_fp8_qtile(self, ulysses_size: int) -> Tile3:
        if self.vae_fp8_qtile is not None:
            return self.vae_fp8_qtile
        return (64, 4, 8) if ulysses_size > 1 else (32, 8, 8)

    def validate(self) -> None:
        if self.attention_fmha not in ("cudnn", "trtllm-gen", "trtllm-gen-fp16sm"):
            raise ValueError(f"attention_fmha={self.attention_fmha!r}")
        if self.attention_fmha == "trtllm-gen-fp16sm" and self.fp8_scale_fusion != "qkstatic":
            raise ValueError("the FP16-softmax kernel needs fp8_scale_fusion='qkstatic' (the fold)")
        if not 0.0 <= self.skip_corr_threshold <= 8.0:
            raise ValueError("skip_corr_threshold must be in [0, 8]")
        if self.fp8_scale_fusion not in ("qkstatic", "static", "dynamic"):
            raise ValueError(f"fp8_scale_fusion={self.fp8_scale_fusion!r}")
        if self.amax_allreduce not in ("symm", "nccl"):
            raise ValueError(f"amax_allreduce={self.amax_allreduce!r}")
        if self.vae_fused_norm and not self.vae_fp8_tiled:
            raise ValueError("vae_fused_norm needs vae_fp8_tiled")


SCENARIO = ScenarioConfig()


def apply(overrides: Optional[Mapping[str, Any]]) -> ScenarioConfig:
    """Replace ``SCENARIO``'s fields from *overrides* (field name -> value) and validate.

    Must run before the scenario package's modules are imported; ``scenarios.resolve_scenario`` does.
    Tuple fields accept lists; unknown names raise.
    """
    global SCENARIO
    cfg = ScenarioConfig()
    if overrides:
        names = {f.name for f in fields(cfg)}
        unknown = sorted(set(overrides) - names)
        if unknown:
            raise ValueError(f"minimax_h3_scenario: unknown settings {unknown}; known: {sorted(names)}")
        values = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items()}
        cfg = replace(cfg, **values)
    cfg.validate()
    SCENARIO = cfg
    return cfg
