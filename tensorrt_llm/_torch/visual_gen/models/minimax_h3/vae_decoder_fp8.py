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

"""FP8 per-tensor GEMMs for the MiniMax-H3 VAE ViT decoder (opt-in, ``H3_L14_VAE_FP8=1``).

The decoder's 36 transformer blocks run once per clip in fp16 autocast and are GEMM-bound
(fp16 ``nvjet`` GEMM kernels dominate the decode). Every ``nn.Linear`` under the
decoder blocks is replaced by :class:`Fp8Linear`: the weight is quantized once to e4m3 with a
static per-tensor scale, the activation is quantized per call with a dynamic per-tensor scale,
and ``torch._scaled_mm`` produces the fp16 output with the original bias. This changes the
decoded pixels (a format change, gated by the quality worker), not the latents.

``H3_L14_VAE_FP8_TILED=1`` additionally routes the two largest activation quantizes of each
block (the SwiGLU output feeding the down projection, and the attention output feeding
``to_out``) through tiled Triton kernels exposed as custom ops instead of the
``torch.compile``-generated split reductions. The kernels reproduce Inductor's arithmetic
(SiLU as ``g / (1 + exp(-g))`` in fp32, fp32 product, the amax stored as fp16, the cast's scale
recomputed from the fp16 amax, the GEMM's scale from the fp32 amax), so the e4m3 codes, the
GEMM inputs and therefore the decoded pixels are bit-identical to the non-tiled path.
"""

import os

import torch
from torch import nn

from tensorrt_llm.logger import logger

E4M3_MAX = 448.0
INV_E4M3_MAX = 0.002232142857142857  # Inductor's constant for "/ 448.0"
_TILED = os.environ.get("H3_L14_VAE_FP8_TILED") == "1"


class Fp8Linear(nn.Module):
    """``nn.Linear`` replacement: e4m3 weight (static per-tensor scale), e4m3 activation (dynamic
    per-tensor scale), fp32 accumulation, fp16 output via ``torch._scaled_mm``."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach().to(torch.float32)
        amax = weight.abs().amax().clamp(min=1e-12)
        w_scale = (amax / E4M3_MAX).to(torch.float32).reshape(())
        w_fp8 = (weight / w_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("weight_fp8", w_fp8.contiguous())  # (out, in) row-major
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().to(torch.float16).contiguous())
        else:
            self.bias = None
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        x_scale = (x.abs().amax().to(torch.float32).clamp(min=1e-12) / E4M3_MAX).reshape(())
        x_fp8 = (x.to(torch.float32) / x_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # (in, out) column-major, as cuBLASLt FP8 requires
            scale_a=x_scale,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.float16,
        )
        return out.reshape(*shape[:-1], self.out_features)


# ---------------------------------------------------------------------------------------------
# Tiled quantize path (H3_L14_VAE_FP8_TILED=1)
# ---------------------------------------------------------------------------------------------
if _TILED:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HEAD_DIM = 64
    _KN = dict(mlp_bt=8, mlp_bc=512, mlp_warps=4, q_bt=16, q_bh=4, q_warps=4)

    @triton.jit
    def _silu_mul(x, g):
        # Inductor: g / (libdevice.exp(-g) + 1.0), then x * that, all fp32
        return x * (g / (libdevice.exp(-g) + 1.0))

    @triton.jit
    def _scale_from_amax16(amax16_ptr):
        # Inductor's cast kernel: stored fp16 amax -> fp32, max(., 1e-12), * (1/448)
        a = tl.load(amax16_ptr).to(tl.float32)
        return tl.maximum(a, 1e-12) * 0.002232142857142857

    @triton.jit
    def _silu_mul_amax_kernel(h_ptr, out_ptr, T, HALF, stride_t, BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < T) & (cols[None, :] < HALF)
        off = rows[:, None] * stride_t + cols[None, :]
        x = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(h_ptr + off + HALF, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(tl.abs(_silu_mul(x, g)))
        tl.atomic_max(out_ptr, m)

    @triton.jit
    def _silu_mul_cast_kernel(h_ptr, amax16_ptr, out_ptr, T, HALF, stride_t, BLOCK_T: tl.constexpr,
                              BLOCK_C: tl.constexpr):
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (rows[:, None] < T) & (cols[None, :] < HALF)
        off = rows[:, None] * stride_t + cols[None, :]
        x = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(h_ptr + off + HALF, mask=mask, other=0.0).to(tl.float32)
        q = _silu_mul(x, g) / _scale_from_amax16(amax16_ptr)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(out_ptr + rows[:, None] * HALF + cols[None, :], q.to(tl.float8e4nv), mask=mask)

    @triton.jit
    def _amax3_kernel(x_ptr, out_ptr, N, T, H, stride_b, stride_t, stride_h, D: tl.constexpr,
                      BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr):
        # rows run over the flattened [B, T] leading dims: row = b * T + t
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ds = tl.arange(0, D)
        mask = (rows[:, None, None] < N) & (hs[None, :, None] < H) & (ds[None, None, :] < D)
        row_off = (rows // T) * stride_b + (rows % T) * stride_t
        off = row_off[:, None, None] + hs[None, :, None] * stride_h + ds[None, None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_max(out_ptr, tl.max(tl.abs(x)))

    @triton.jit
    def _cast3_kernel(x_ptr, amax16_ptr, out_ptr, N, T, H, stride_b, stride_t, stride_h, D: tl.constexpr,
                      BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)
        rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        hs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        ds = tl.arange(0, D)
        mask = (rows[:, None, None] < N) & (hs[None, :, None] < H) & (ds[None, None, :] < D)
        row_off = (rows // T) * stride_b + (rows % T) * stride_t
        off = row_off[:, None, None] + hs[None, :, None] * stride_h + ds[None, None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
        q = x / _scale_from_amax16(amax16_ptr)
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        out_off = rows[:, None, None] * (H * D) + hs[None, :, None] * D + ds[None, None, :]
        tl.store(out_ptr + out_off, q.to(tl.float8e4nv), mask=mask)

    @torch.library.custom_op("h3vae::silu_mul_amax", mutates_args=())
    def _silu_mul_amax(h: torch.Tensor) -> torch.Tensor:
        h2 = h.reshape(-1, h.shape[-1])
        T, C = h2.shape
        half = C // 2
        out = torch.zeros((), dtype=torch.float32, device=h.device)
        grid = (triton.cdiv(T, _KN["mlp_bt"]), triton.cdiv(half, _KN["mlp_bc"]))
        _silu_mul_amax_kernel[grid](h2, out, T, half, h2.stride(0), BLOCK_T=_KN["mlp_bt"], BLOCK_C=_KN["mlp_bc"],
                                    num_warps=_KN["mlp_warps"])
        return out

    @_silu_mul_amax.register_fake
    def _(h):
        return torch.empty((), dtype=torch.float32, device=h.device)

    @torch.library.custom_op("h3vae::silu_mul_cast", mutates_args=())
    def _silu_mul_cast(h: torch.Tensor, amax16: torch.Tensor) -> torch.Tensor:
        h2 = h.reshape(-1, h.shape[-1])
        T, C = h2.shape
        half = C // 2
        out = torch.empty((T, half), dtype=torch.float8_e4m3fn, device=h.device)
        grid = (triton.cdiv(T, _KN["mlp_bt"]), triton.cdiv(half, _KN["mlp_bc"]))
        _silu_mul_cast_kernel[grid](h2, amax16, out, T, half, h2.stride(0), BLOCK_T=_KN["mlp_bt"],
                                    BLOCK_C=_KN["mlp_bc"], num_warps=_KN["mlp_warps"])
        return out

    @_silu_mul_cast.register_fake
    def _(h, amax16):
        return torch.empty((h.numel() // h.shape[-1], h.shape[-1] // 2), dtype=torch.float8_e4m3fn, device=h.device)

    def _as_bthd(x: torch.Tensor):
        """A [B, T, H, D] attention output (any B / T / H strides, D contiguous; the decoder
        batches up to 16 tiles so B is 1..16) or a contiguous [..., C] activation as
        (view, B, T, H, D, stride_b, stride_t, stride_h)."""
        if x.dim() == 4:
            B, T, H, D = x.shape
            if x.stride(3) != 1:
                raise ValueError("tiled quantize expects a [B, T, H, D] view with contiguous D")
            return x, B, T, H, D, x.stride(0), x.stride(1), x.stride(2)
        x2 = x.reshape(-1, x.shape[-1])
        T, C = x2.shape
        D = _HEAD_DIM
        return x2, 1, T, C // D, D, 0, x2.stride(0), D

    @torch.library.custom_op("h3vae::amax", mutates_args=())
    def _amax(x: torch.Tensor) -> torch.Tensor:
        v, B, T, H, D, sb, st, sh = _as_bthd(x)
        out = torch.zeros((), dtype=torch.float32, device=x.device)
        grid = (triton.cdiv(B * T, _KN["q_bt"]), triton.cdiv(H, _KN["q_bh"]))
        _amax3_kernel[grid](v, out, B * T, T, H, sb, st, sh, D=D, BLOCK_T=_KN["q_bt"], BLOCK_H=_KN["q_bh"],
                            num_warps=_KN["q_warps"])
        return out

    @_amax.register_fake
    def _(x):
        return torch.empty((), dtype=torch.float32, device=x.device)

    @torch.library.custom_op("h3vae::cast", mutates_args=())
    def _cast(x: torch.Tensor, amax16: torch.Tensor) -> torch.Tensor:
        v, B, T, H, D, sb, st, sh = _as_bthd(x)
        out = torch.empty((B * T, H * D), dtype=torch.float8_e4m3fn, device=x.device)
        grid = (triton.cdiv(B * T, _KN["q_bt"]), triton.cdiv(H, _KN["q_bh"]))
        _cast3_kernel[grid](v, amax16, out, B * T, T, H, sb, st, sh, D=D, BLOCK_T=_KN["q_bt"],
                            BLOCK_H=_KN["q_bh"], num_warps=_KN["q_warps"])
        return out

    @_cast.register_fake
    def _(x, amax16):
        if x.dim() == 4:
            B, T, H, D = x.shape
            return torch.empty((B * T, H * D), dtype=torch.float8_e4m3fn, device=x.device)
        return torch.empty((x.numel() // x.shape[-1], x.shape[-1]), dtype=torch.float8_e4m3fn, device=x.device)

    def _gemm_scale(amax32: torch.Tensor) -> torch.Tensor:
        # Inductor computes the GEMM's scale_a from the fp32 amax held in registers
        return (torch.clamp(amax32, min=1e-12) * INV_E4M3_MAX).reshape(())

    class Fp8LinearTiled(nn.Module):
        """``Fp8Linear`` whose activation amax and cast run as the tiled kernels; accepts the
        [B, T, H, D] attention output directly (B = tiles per decoder call)."""

        def __init__(self, lin: Fp8Linear) -> None:
            super().__init__()
            self.lin = lin
            self.out_features = lin.out_features

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lead = x.shape[:2] if x.dim() == 4 else x.shape[:-1]  # [B, T] rows of the [B*T, out] GEMM
            amax32 = torch.ops.h3vae.amax(x)
            codes = torch.ops.h3vae.cast(x, amax32.to(torch.float16))
            out = torch._scaled_mm(codes, self.lin.weight_fp8.t(), scale_a=_gemm_scale(amax32), scale_b=self.lin.w_scale,
                                   bias=self.lin.bias, out_dtype=torch.float16)
            return out.reshape(*lead, self.out_features)

    class Fp8SwiGLUDown(nn.Module):
        """Replaces ``FeedForward.net`` = [SwiGLU(proj), Dropout, Fp8Linear(down)]: the proj GEMM, the
        fused silu*mul amax and cast, the down GEMM."""

        def __init__(self, proj: Fp8Linear, down: Fp8Linear) -> None:
            super().__init__()
            self.proj = proj
            self.down = down

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.proj(x)
            amax32 = torch.ops.h3vae.silu_mul_amax(h)
            codes = torch.ops.h3vae.silu_mul_cast(h, amax32.to(torch.float16))
            out = torch._scaled_mm(codes, self.down.weight_fp8.t(), scale_a=_gemm_scale(amax32), scale_b=self.down.w_scale,
                                   bias=self.down.bias, out_dtype=torch.float16)
            return out.reshape(*x.shape[:-1], self.down.out_features)

    def _make_tiled_processor():
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import MiniMaxH3VideoAttnProcessor

        class TiledAttnProcessor(MiniMaxH3VideoAttnProcessor):
            """The stock processor, except that the attention output reaches ``to_out`` as the strided
            ``[B, T, H, D]`` view (no flatten copy in front of the opaque quantize op)."""

            def __call__(self, attn, hidden_states, rotary_emb=None):
                query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
                key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
                value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
                query = attn.norm_q(query.float()).to(query.dtype)
                key = attn.norm_k(key.float()).to(key.dtype)
                if rotary_emb is not None:
                    cos, sin = rotary_emb
                    cos = cos.to(query.dtype)
                    sin = sin.to(query.dtype)
                    rotary_dim = cos.shape[-1]
                    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
                    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
                    query_first, query_second = query_rotary.chunk(2, dim=-1)
                    key_first, key_second = key_rotary.chunk(2, dim=-1)
                    query_rotated = torch.cat([-query_second, query_first], dim=-1)
                    key_rotated = torch.cat([-key_second, key_first], dim=-1)
                    query = torch.cat([query_rotary * cos + query_rotated * sin, query_pass], dim=-1)
                    key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)
                hidden_states = dispatch_attention_fn(query, key, value, attn_mask=None, backend=self._attention_backend,
                                                      parallel_config=self._parallel_config)
                return attn.to_out[0](hidden_states)

        return TiledAttnProcessor()

    def _tile_block(block: nn.Module) -> None:
        swiglu, down = block.ff.net[0], block.ff.net[2]
        block.ff.net = nn.ModuleList([Fp8SwiGLUDown(swiglu.proj, down)])
        block.attn.to_out[0] = Fp8LinearTiled(block.attn.to_out[0])
        block.attn.set_processor(_make_tiled_processor())


def convert_decoder_linears_fp8(decoder: nn.Module) -> int:
    """Replace every ``nn.Linear`` under ``decoder.transformer_blocks`` with :class:`Fp8Linear`.

    ``proj_in`` (K = 24) and ``proj_out`` run once per decoder call and stay fp16. Returns the
    number of replaced modules. Call after ``_prepare_vae_decoder`` and before ``torch_compile``.
    With ``H3_L14_VAE_FP8_TILED=1`` the SwiGLU-output and attention-output quantizes of every
    block then move to the tiled kernels (bit-identical codes).
    """
    replaced = 0
    for block in decoder.transformer_blocks:
        for parent in list(block.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Fp8Linear(child))
                    replaced += 1
        if _TILED:
            _tile_block(block)
    logger.info(f"MiniMax-H3 VAE decoder: {replaced} Linears converted to FP8 per-tensor"
                + (", tiled SwiGLU / attention-output quantize" if _TILED else ""))
    return replaced
