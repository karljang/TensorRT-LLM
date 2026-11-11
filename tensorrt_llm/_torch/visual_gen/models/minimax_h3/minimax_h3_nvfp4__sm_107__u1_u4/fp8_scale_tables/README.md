# MiniMax-H3 static FP8 descale table

`minimax_h3_fp8_qkv_descales_544x960x124.json` stores a `descale_max` mapping from `str(layer_idx)` to `[descale_q, descale_k, descale_v]`, with `calls` and `set` metadata.
The loader accepts this wrapper or a bare layer mapping for `fp8_scale_fusion="static"`.
Set `fp8_static_scales` to this file and select `attention_fmha="cudnn"` or `"trtllm-gen"`; FP16-softmax requires `"qkstatic"` instead.

A descale is `max(amax / 448, 1 / (448 * 512))` rounded to BF16, matching dynamic mode's per-call scale.
`fp8_static_margin` (default 1.0) multiplies each table entry before its BF16 rounding.

The artifact identifies a held-out calibration set at 544 × 960 × 124, seed 42, and records 328 calls per layer.
The prompt set is not included, so this file alone does not reproduce calibration.
It is not a scale guarantee for other inputs or shapes; the default `qkstatic` mode uses fixed Q/K scales and dynamic V scales, not this table.
