# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``kernels/fused_fp8_qk.py``: per-head RMSNorm + partial split-half RoPE + FP8 E4M3 per-tensor cast.

The module docstring states two contracts, and each is checked against a reference that does not
share the kernel's code:

* the cast: ``scale = bf16(max(amax / 448, 1 / (448 * 512)))``, codes ``rn.satfinite(x * (1 / scale))``,
  descale ``float(scale)``, the scale arithmetic of TensorRT-LLM's per-tensor quantizer. The reference
  is ``torch.ops.tensorrt_llm.quantize_e4m3_per_tensor`` on the BF16 normalised / rotated tensor, and
  the same arithmetic written in torch where the C++ op has no equivalent (padded rows, the static Q /
  K descale);
* the norm and RoPE: the BF16 value equals the compiled eager path of the family (RMSNorm in fp32,
  one BF16 rounding, weight and rotation in fp32, one BF16 rounding), i.e. ``torch.compile`` of
  ``apply_minimax_h3_rotary_emb`` after the norm.

The kernel variants the scenario can select (split vs grouped Q / K cast, V-amax fold, scale mode,
plain vs packed Ulysses send buffer) are pinned per case with ``monkeypatch`` on the module globals;
the process scenario configuration is never re-selected.
"""

import pytest
import torch
from utils.util import skip_pre_ada

import tensorrt_llm  # noqa: F401  -- loads libth_common.so (registers the tensorrt_llm ops)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
    fused_fp8_qk as fq,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
    apply_minimax_h3_rotary_emb,
)

pytestmark = skip_pre_ada

H, D, ROT, EPS = 56, 128, 96, 1e-6
U = 4
E4M3 = torch.float8_e4m3fn
MIN_SCALE = 1.0 / (448.0 * 512.0)
ROWS = (1017, 9455)
# (``_V2`` grouped cast, ``_VAMAX_FOLD``): the split kernels, the grouped Q / K cast, and the
# grouped cast with V's amax folded into the Q launch (inert unless the scale mode is qkstatic).
KERNELS = {"split": (False, False), "grouped": (True, False), "grouped_fold": (True, True)}


def _inputs(rows: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn((1, rows, 3 * H * D), generator=gen, device="cuda") * 2.0
    # V outliers in the first rows: the dynamic V scale is set there, and the padded case keeps them.
    qkv[0, : max(rows // 97, 1), 2 * H * D :: 977] *= 40.0
    qkv = qkv.to(torch.bfloat16)
    wq = (1.0 + 0.1 * torch.randn(D, generator=gen, device="cuda")).to(torch.bfloat16)
    wk = (1.0 + 0.1 * torch.randn(D, generator=gen, device="cuda")).to(torch.bfloat16)
    pos = torch.arange(rows, dtype=torch.float32, device="cuda")[:, None]
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROT, 2, dtype=torch.float32, device="cuda") / ROT))
    ang = torch.cat([pos * inv, pos * inv], dim=-1)
    return qkv, wq, wk, ang.cos(), ang.sin()


def _planes(qkv: torch.Tensor):
    """Q, K, V of the packed projection output as contiguous ``[1, S, H, D]`` BF16 tensors."""
    s = qkv.shape[1]
    return [
        qkv[..., i * H * D : (i + 1) * H * D].reshape(1, s, H, D).contiguous() for i in range(3)
    ]


def _norm_rope_bf16(x, w, cos, sin):
    out = fq.norm_rope_tensor(x, w, cos, sin, EPS)
    assert out.dtype == torch.bfloat16 and out.shape == x.shape
    return out


def _descale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """fp32 ``float(bf16(max(amax / 448, 1 / (448 * 512))))``, the documented scale arithmetic."""
    return (
        torch.maximum(amax.float() / 448.0, torch.tensor(MIN_SCALE, device=amax.device))
        .to(torch.bfloat16)
        .float()
    )


def _cast(y: torch.Tensor, descale: torch.Tensor) -> torch.Tensor:
    """``rn.satfinite.e4m3(y * (1 / descale))`` in fp32; the clamp makes torch's cast saturating."""
    return (y.float() * (1.0 / descale.float())).clamp(-448.0, 448.0).to(E4M3)


def _reference(qkv, wq, wk, cos, sin, mode, valid_len=None):
    """Codes ``[3][1, S, H * D]`` and the ``[3]`` fp32 descale table for one scale mode."""
    s = qkv.shape[1]
    q, k, v = _planes(qkv)
    tensors = (_norm_rope_bf16(q, wq, cos, sin), _norm_rope_bf16(k, wk, cos, sin), v)
    codes, descales = [], []
    for i, y in enumerate(tensors):
        if mode == "qkstatic" and i < 2:
            d = torch.tensor(fq.qk_static_descale_value(D), dtype=torch.float32, device=y.device)
        else:
            rows = y if valid_len is None else y[:, : int(valid_len.item())]
            d = _descale_from_amax(rows.float().abs().amax())
            if valid_len is None:
                # The C++ per-tensor quantizer must agree with the documented arithmetic.
                ref_codes, ref_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(y)
                assert torch.equal(ref_scale.float().reshape(()), d), (i, ref_scale, d)
                assert torch.equal(ref_codes.view(torch.uint8), _cast(y, d).view(torch.uint8)), i
        codes.append(_cast(y, d).reshape(1, s, H * D))
        descales.append(d.reshape(()))
    return codes, torch.stack(descales)


def _run(qkv, wq, wk, cos, sin, mode, ulysses, valid_len=None):
    static = fq.qk_static_descale(D, qkv.device) if mode == "qkstatic" else None
    outs, descale = fq._qk_norm_rope_fp8_impl(
        qkv, wq, wk, cos, sin, static, EPS, H, D, 0, valid_len, ulysses
    )
    if ulysses > 1:
        # Send buffer [U, S, 3, H / U, D] -> the three plain [1, S, H * D] tensors.
        send = outs[0]
        s = qkv.shape[1]
        plain = [send[:, :, i].permute(1, 0, 2, 3).reshape(1, s, H * D) for i in range(3)]
        return plain, descale
    return list(outs), descale


def _pin(monkeypatch, mode: str, kernel: str) -> None:
    v2, fold = KERNELS[kernel]
    monkeypatch.setattr(fq, "_MODE", mode)
    monkeypatch.setattr(fq, "_V2", v2)
    monkeypatch.setattr(fq, "_VAMAX_FOLD", fold)
    monkeypatch.setattr(fq, "_split_launch", lambda: True)
    monkeypatch.setattr(fq, "_REDUCE_GROUP", None)


def _assert_codes(actual, expected, what):
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.dtype == E4M3 and a.shape == e.shape, (what, i, a.dtype, a.shape, e.shape)
        assert torch.equal(a.view(torch.uint8), e.view(torch.uint8)), (what, "QKV"[i])


@pytest.mark.parametrize("ulysses", (0, U), ids=("plain", "packed_u4"))
@pytest.mark.parametrize("kernel", tuple(KERNELS))
@pytest.mark.parametrize("mode", ("dynamic", "qkstatic"))
@pytest.mark.parametrize("rows", ROWS)
def test_cast_matches_reference(monkeypatch, rows, mode, kernel, ulysses):
    _pin(monkeypatch, mode, kernel)
    qkv, wq, wk, cos, sin = _inputs(rows, seed=rows)
    ref_codes, ref_descale = _reference(qkv, wq, wk, cos, sin, mode)
    codes, descale = _run(qkv, wq, wk, cos, sin, mode, ulysses)
    _assert_codes(codes, ref_codes, f"{mode}/{kernel}/{ulysses}")
    assert torch.equal(descale.view(torch.int32), ref_descale.view(torch.int32)), (
        descale,
        ref_descale,
    )


@pytest.mark.parametrize(
    "mode,kernel,ulysses",
    (("dynamic", "split", 0), ("qkstatic", "grouped_fold", U)),
    ids=("dynamic-split-plain", "qkstatic-fold-packed"),
)
def test_padded_rows_leave_the_scale(monkeypatch, mode, kernel, ulysses):
    """Rows past ``valid_len`` are cast but do not enter the amax."""
    _pin(monkeypatch, mode, kernel)
    rows = 1017
    qkv, wq, wk, cos, sin = _inputs(rows, seed=7)
    valid = rows - rows // 10
    qkv[0, valid:] *= 1000.0  # padded rows far above the valid ones: they would set the scale
    valid_len = torch.tensor([valid], dtype=torch.int32, device="cuda")
    ref_codes, ref_descale = _reference(qkv, wq, wk, cos, sin, mode, valid_len)
    codes, descale = _run(qkv, wq, wk, cos, sin, mode, ulysses, valid_len)
    _assert_codes(codes, ref_codes, f"padded {mode}/{kernel}")
    assert torch.equal(descale.view(torch.int32), ref_descale.view(torch.int32)), (
        descale,
        ref_descale,
    )


def test_zero_tensor_takes_the_scale_floor(monkeypatch):
    _pin(monkeypatch, "dynamic", "grouped")
    qkv, wq, wk, cos, sin = _inputs(1017, seed=11)
    qkv[..., H * D : 2 * H * D] = 0.0  # K all zero: descale = bf16(1 / (448 * 512)), codes 0
    codes, descale = _run(qkv, wq, wk, cos, sin, "dynamic", 0)
    floor = torch.tensor(MIN_SCALE, dtype=torch.float32).to(torch.bfloat16).float().item()
    assert descale[1].item() == floor
    assert not codes[1].view(torch.uint8).any()


def _compiled_eager_norm_rope():
    def eager(x, w, cos, sin):
        # The family's unfused path: RMSNorm in fp32 with one BF16 rounding, the weight in the
        # activation dtype, then the partial split-half RoPE (Inductor keeps the last two in fp32).
        x32 = x.float()
        y = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype)
        return apply_minimax_h3_rotary_emb(w * y, cos, sin)

    return torch.compile(eager, fullgraph=True, dynamic=False)


@pytest.mark.parametrize("rows", ROWS)
def test_norm_rope_bf16_matches_compiled_eager(rows):
    """The module's claim that its BF16 value is the compiled eager path's, element for element."""
    qkv, wq, wk, cos, sin = _inputs(rows, seed=rows + 1)
    q, k, _ = _planes(qkv)
    ref = _compiled_eager_norm_rope()
    for x, w, name in ((q, wq, "Q"), (k, wk, "K")):
        expected = ref(x, w, cos, sin)
        actual = _norm_rope_bf16(x, w, cos, sin)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), name


@pytest.mark.parametrize("mode", ("dynamic", "qkstatic"))
def test_tensor_major_send_is_the_row_major_send_permuted(monkeypatch, mode):
    """The tensor-major producer layout ``[3, U, S, H / U, D]`` holds the row-major codes."""
    _pin(monkeypatch, mode, "grouped")
    rows, hl = 1017, H // U
    qkv, wq, wk, cos, sin = _inputs(rows, seed=13)
    static = fq.qk_static_descale(D, qkv.device) if mode == "qkstatic" else None
    block_h, num_warps = fq._launch_config()
    send_rm = torch.empty((U, rows, 3, hl, D), dtype=E4M3, device="cuda")
    send_tm = torch.empty((3, U, rows, hl, D), dtype=E4M3, device="cuda")
    for tensor_major, outs in (
        (False, (send_rm,) * 3),
        (True, (send_tm[0], send_tm[1], send_tm[2])),
    ):
        fq._qk_norm_rope_fp8_split(
            qkv,
            wq,
            wk,
            cos,
            sin,
            static,
            EPS,
            H,
            D,
            rows,
            rows,
            outs,
            block_h,
            num_warps,
            None,
            packed_hl=hl,
            qk_static=static is not None,
            tensor_major=tensor_major,
        )
    expected = send_rm.permute(2, 0, 1, 3, 4).contiguous()
    assert torch.equal(send_tm.view(torch.uint8), expected.view(torch.uint8))
