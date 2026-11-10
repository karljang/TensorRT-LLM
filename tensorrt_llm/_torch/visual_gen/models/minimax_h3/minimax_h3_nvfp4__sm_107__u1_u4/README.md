# MiniMax-H3 scenario `minimax_h3_nvfp4__sm_107__u1_u4`

The MiniMax-H3 pipeline as measured on R200 (SM 107) in board thread 692: pre-quantized NVFP4 Linears,
FP8 attention through trtllm-gen's FP16-softmax dense FMHA, Ulysses 1 or 4 on one node. This folder is
one *scenario* in the shape of Modeling V2's targets: a complete variant of the H3 transformer and
pipeline with its own kernels, selected by the family's `scenarios.py`, leaving the shared H3 code and
the shared attention backends on their default paths.

## Selecting it

```yaml
# pipeline config
attention:
  backend: CUDNN
  quant_attention_config: {qk_dtype: fp8, v_dtype: fp8}
extra_attrs:
  visual_gen_scenario: require      # off (default) | auto | require
  # minimax_h3_scenario: {skip_corr_threshold: 0.0}   # optional overrides of config.py's fields
```

`scenarios.route_nvfp4_sm_107_u1_u4` admits a pre-quantized NVFP4 transformer checkpoint
(`dynamic_weight_quant=False`), an SM 107 device, the cuDNN backend with the per-tensor FP8 recipe,
Ulysses size 1 or 4, `tp_size=1`, and not `async_ulysses`; anything else falls back to the family
pipeline under `auto` or raises under `require`. `off` imports none of this folder.

## What runs (the defaults of `config.py`)

| part | kernel / path | board 692 |
|---|---|---|
| QK norm + RoPE + FP8 cast | one grouped Q / K cast with V's amax folded in; static Q / K scales that carry `sm_scale · log2 e`; V with its dynamic per-tensor amax | lever 8, N.3, F.1 (a), A.3 |
| attention | trtllm-gen `fmha_dense_fp8`, Fp16Softmax SM107a cubin (unit softmax scale), deferred output rescale threshold 1 | A.3, A.4, A.5 (a) |
| Ulysses 4 | one packed FP8 all-to-all into `[S, 3, H / U, D]`, read in place by the op (strided views); amax reduce through symmetric memory; the attention output exchanged as NVFP4 codes + scales | X.1, U.3, lever 13b |
| attention output | the tiled NVFP4 quantize for the static-NVFP4 `to_out` | lever 17 |
| norm + modulation, SwiGLU | producers write the Linears' NVFP4 inputs directly | lever 9 |
| VAE decoder | FP8 per-tensor Linears, tiled SwiGLU / attention-output quantize, shared q / k / v quantize and one-kernel q / k norm + RoPE, spatial tiles 288 × 272 × 32, decoder call replayed from CUDA graphs | lever 14, V.1 (a), §689 rider |
| text encoder | the text-only prefill replayed from a CUDA graph per prompt length | V.1 (b1) |
| denoise loop host side | (family) no per-step host sync, one plan transfer, overlapped audio decode | lever 14, §689 rider |

Every value is a field of `config.ScenarioConfig`; overrides go through
`extra_attrs["minimax_h3_scenario"]` and are read once, before this package's modules are imported.

## Layout

```
minimax_h3_nvfp4__sm_107__u1_u4/
├── config.py        # ScenarioConfig, SCENARIO, apply()
├── modeling.py      # ScenarioAttention / ScenarioBlock / ScenarioTransformer (subclasses of the family's)
├── pipeline.py      # MiniMaxH3Nvfp4Sm107U1U4Pipeline (subclass of MiniMaxH3Pipeline)
├── kernels/         # Triton / PTX: fused_fp8_qk, fused_out_quant, fused_out_a2a, fused_quant, vae_decoder_fp8
└── fp8_scale_tables/  # static descale tables for fp8_scale_fusion="static"
```

The shared code this scenario relies on is in the main tree: the trtllm-gen FMHA runner and op
(unit-scale cubins, Q / K token strides, `skip_corr_threshold`), the cuDNN backend options
(`fp8_dense_fmha`, `fp8_skip_corr_threshold`, `fp8_strided_views`), `UlyssesAttention.forward_packed_fp8`,
and the family hooks (`attention_cls` / `block_cls`, `_attention_backend_kwargs`, `resolve_variant`,
`_text_prefill_fast`, `decoder_graph_max_shapes`).

Timings stay on the board (internal); nothing here states a number.
