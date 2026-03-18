# Qwen3-TTS TensorRT-LLM Integration Worklog

## Task
Add Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) support to TensorRT-LLM.

## User Instructions
- It is NOT an LLM, but a TTS pipeline with multiple models
- Use %11 for GPU access
- Final artifact: a script demonstrating TTS producing audio from a given prompt
- **MUST use TRT-LLM modules in demo — without executor integration, not done**
- If PyExecutor not doable, use custom pipeline with TRT-LLM modules
- **Pipeline must match HF pipeline exactly** — same generation dynamics
- Always use `pip install --no-deps` to avoid breaking NVIDIA custom torch
- Log progress frequently

## Model Architecture

**Qwen3-TTS** is a unified end-to-end TTS system with 3 components:

### 1. Talker Model (28-layer Transformer, ~1.7B params)
- Text tokens → Codebook-0 (semantic) speech tokens (autoregressive)
- 2048 hidden size, 16 attention heads, 8 KV heads (GQA)
- M-RoPE positional encoding (sections [24, 20, 20], interleaved=true)
- QK-norm (RMSNorm on head_dim)
- Two embeddings: codec_embedding (vocab=3072) + text_embedding (vocab=151936)
- Text vocab projected via text_projection MLP (text_hidden_size → hidden_size)

### 2. CodePredictor (5-layer Transformer)
- Codebook-0 → Codebooks 1-15 (autoregressive multi-token prediction)
- 1024 hidden size, 16 attention heads, 8 KV heads (GQA)
- QK-norm (RMSNorm on head_dim), standard RoPE
- 16 code groups, 2048 vocab per group
- 15 separate lm_head Linear layers + 15 separate embedding layers
- small_to_mtp_projection: talker_hidden_size → code_predictor_hidden_size

### 3. Speech Tokenizer Decoder (Causal ConvNet vocoder)
- 16 codebook codes per frame → 24kHz waveform
- Frame rate: 12.5 Hz (80ms per frame)
- Lightweight, non-diffusion, NOT a transformer

## TRT-LLM Integration Design

### Pipeline (matches HF exactly)

```
┌──────────────────────────────────────────────────────────┐
│ Embedding Preparation (Python, reuse HF weights)         │
│   Text → tokenize → text_embed + codec_embed + speaker   │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Prefill: Talker.forward(input_embeds)                    │
│   ★ TRT-LLM modules (28L GQA + M-RoPE + QK-norm)       │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Decode Loop (custom Python, step by step):               │
│                                                          │
│   1. Sample codebook-0 token from Talker logits          │
│                                                          │
│   2. CodePredictor.generate(talker_hidden, codebook-0)   │
│      → generates codebooks 1-15                          │
│      ★ TRT-LLM modules (5L GQA + RoPE + QK-norm)       │
│                                                          │
│   3. Sum all 16 codebook embeddings + text embedding     │
│      (Python, same as HF)                                │
│                                                          │
│   4. Talker.forward(next_embed, kv_cache)                │
│      ★ TRT-LLM modules (single decode step)             │
│                                                          │
│   5. logits = codec_head(hidden_state)                   │
│      repeat until codec_eos_token                        │
└──────────────────────┬───────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Decode Audio: 16 codebooks → 24kHz waveform              │
│   HF Speech Tokenizer (ConvNet vocoder, not transformer) │
└──────────────────────────────────────────────────────────┘
```

### What uses TRT-LLM vs HF

| Component | TRT-LLM? | Notes |
|-----------|----------|-------|
| **Talker** (28L transformer) | ✅ Yes | QKNormRoPEAttention + M-RoPE, GatedMLP, RMSNorm |
| **CodePredictor** (5L transformer) | ✅ Yes | QKNormRoPEAttention + RoPE, GatedMLP, RMSNorm |
| Embedding lookup + text_projection | ❌ No | Light ops, vanilla PyTorch |
| Generation loop orchestration | ❌ No | Custom Python (matches HF logic exactly) |
| Speech Tokenizer Decoder | ❌ No | ConvNet vocoder, not a transformer |

### Key Implementation Details

**KV Cache Management**: Both Talker and CodePredictor need KV cache for
autoregressive generation. We manage this via `AttentionMetadata` with
`KVCacheManager`, same pattern as TRT-LLM's test infrastructure.

**M-RoPE for Talker**: Uses `PositionEmbeddingType.mrope` with sections
[24, 20, 20] and interleaved=true. Position IDs are 3D: (3, batch, seq).
For TTS, all 3 dimensions use the same sequential position (no spatial dims).

**CodePredictor per-step**: At each Talker decode step, CodePredictor runs
a short autoregressive loop (15 steps for codebooks 1-15). Each CodePredictor
run is independent (no cross-step KV cache reuse).

**Weight mapping**: HF checkpoint has prefix `talker.model.*` for Talker and
`talker.code_predictor.*` for CodePredictor. Remap to `model.*` for TRT-LLM.

## Checklist

### Phase 0: Setup & Research
- [x] Research model architecture from HuggingFace
- [x] Explore TRT-LLM codebase for existing TTS/audio patterns
- [x] Read HF modeling source code
- [x] Set up GPU environment on %11

### Phase 1: HF Baseline (reference)
- [x] HF baseline working, audio verified
- [x] Profiled: RTF ~1.1-1.6x (baseline)
- [x] torch.compile: RTF ~0.55 (2x speedup reference)

### Phase 2: TRT-LLM Model Definitions
- [x] `tensorrt_llm/_torch/models/modeling_qwen3_tts.py`
  - Talker: Qwen3TTSTalkerAttention, DecoderLayer, Model, ForCausalLM
  - CodePredictor: Qwen3TTSCodePredictorAttention, DecoderLayer, Model
- [x] Registered Talker in `__init__.py`
- [x] Talker-only checkpoint prepared (`talker_checkpoint/`)

### Phase 3: TRT-LLM Pipeline Integration ← CURRENT
- [ ] Fix venv (rebuild with bld.sh to get all C++ deps)
- [x] Fix venv (rebuilt with bld.sh)
- [x] Verify TRT-LLM model instantiation + weight loading (ALL TESTS PASS)
- [x] Fix M-RoPE config: rope_scaling.type="default" → "mrope" for TRT-LLM
- [ ] Implement KV cache management (currently full recomputation)
- [x] Implement custom generation loop (Talker TRT-LLM + CodePredictor HF)
- [x] Match HF pipeline logic exactly
- [x] End-to-end audio generation ✅

### Phase 4: Demo & Verification
- [x] `demo_trtllm_pipeline.py` — full TTS demo using TRT-LLM Talker
- [x] Audio generated successfully (15.68s, RTF=1.526)
- [ ] Port CodePredictor to TRT-LLM modules (currently HF)
- [ ] Add KV cache for faster decode
- [ ] Performance comparison

## Performance Results

### HF Baseline (bf16, H100/H200)
| Test | Avg Time | Audio Duration | RTF |
|------|----------|----------------|-----|
| Short English | 6.01s | ~5s | ~1.2 |
| Medium English | 11.78s | ~7s | ~1.7 |
| Chinese | 6.81s | ~5.9s | ~1.15 |

### torch.compile (mode="default", bf16, H100/H200)
| Test | Avg Time | Audio Duration | RTF |
|------|----------|----------------|-----|
| Short English | ~2.5s | ~4.5s | ~0.55 |
| Medium English | ~4.4s | ~7.9s | ~0.56 |
| Chinese | 3.37s | ~5.9s | ~0.57 |

### TRT-LLM Pipeline (Talker=TRT-LLM, CodePredictor=HF, no KV cache)
| Test | Time | Audio | RTF | Notes |
|------|------|-------|-----|-------|
| Team intro (long) | 23.92s | 15.68s | 1.526 | Full recomputation every step |

**Note**: RTF > 1.0 because no KV cache — every decode step recomputes all previous tokens.
With KV cache, only the new token would be processed, giving ~10-20x speedup on decode.

## Files

| File | Description |
|------|-------------|
| `tensorrt_llm/_torch/models/modeling_qwen3_tts.py` | TRT-LLM model definitions (Talker + CodePredictor) |
| `tensorrt_llm/_torch/models/__init__.py` | Registration (Qwen3TTSTalkerForCausalLM) |
| `local/qwen3-tts/worklog.md` | This worklog |
| `local/qwen3-tts/talker_checkpoint/` | Extracted Talker weights + config |
| `local/qwen3-tts/prepare_talker_checkpoint.py` | Script to extract Talker checkpoint |
| `local/qwen3-tts/test_llm_api.py` | TRT-LLM LLM() API test (pending) |
| `local/qwen3-tts/test_trtllm_talker.py` | Model instantiation + weight loading test |
| `local/qwen3-tts/demo_hf_baseline.py` | HF baseline (reference) |
| `local/qwen3-tts/demo_profiled.py` | Profiled benchmark |
| `local/qwen3-tts/demo_compiled.py` | torch.compile benchmark |
| `/home/scratch.kanghwanj_coreai_1/n/bin/bld.sh` | Updated with QWEN3TTS=1 option |

## Environment Notes
- Docker: trtllm-dev on 4u8g-gen-0029
- Venv: `.venv-3.12` with `--system-site-packages`
- Patched `qwen_tts/core/__init__.py` and `inference/qwen3_tts_tokenizer.py`:
  Made 25hz tokenizer import optional (torchaudio incompatible with NVIDIA torch)
- flashinfer required for full TRT-LLM import chain (system-site-packages provides it)

## Work Log

### 2026-03-18: Research & Setup
- Analyzed Qwen3-TTS architecture from HuggingFace model card and config.json
- Explored TRT-LLM codebase: found visual_gen pipeline pattern and existing Qwen models
- Key finding: Talker is similar to Qwen3 (GQA + QK-norm) with M-RoPE from Qwen3-VL
- Set up GPU environment on %11 (Docker trtllm-dev, .venv-3.12)
- Fixed torchaudio incompatibility by patching qwen_tts imports

### 2026-03-18: HF Baseline & torch.compile
- HF baseline working, profiled RTF ~1.1-1.6x
- torch.compile achieves RTF ~0.55 (2x speedup, faster than real-time)
- Studied generate() flow: nested loop where CodePredictor runs inside each Talker step

### 2026-03-18: TRT-LLM Model Definitions
- Created `modeling_qwen3_tts.py` with Talker and CodePredictor classes
- Talker: QKNormRoPEAttention with M-RoPE (like Qwen3-VL)
- CodePredictor: QKNormRoPEAttention with standard RoPE
- Registered Talker as `Qwen3TTSTalkerForCausalLM` in `__init__.py`
- Committed on branch `hack/qwen3-tts` (d528dcb66d)

### 2026-03-18: Pipeline Design
- Designed decoupled pipeline (Talker first, CodePredictor after) — rejected
  by user because it doesn't match HF pipeline dynamics
- Redesigned: exact HF pipeline with TRT-LLM modules inside both transformers
- Both Talker (28L) and CodePredictor (5L) use TRT-LLM optimized modules
- Only ConvNet vocoder and embedding lookups stay vanilla PyTorch
- Created Talker-only checkpoint (talker_checkpoint/)
- Hit venv issues: missing flashinfer, CUTLASS DSL extensions
- User rebuilding venv via bld.sh
- Added QWEN3TTS=1 option to bld.sh for TTS-specific deps

### 2026-03-18: TRT-LLM Pipeline Working ✅
- Rebuilt venv with bld.sh (all deps resolved)
- TRT-LLM model tests ALL PASS: instantiation, weight loading, forward pass
- Fixed M-RoPE config: Qwen3-TTS has rope_scaling.type="default" with mrope_section,
  needed to override to "mrope" for TRT-LLM's RotaryScalingType parser
- Implemented full generation pipeline in `demo_trtllm_pipeline.py`:
  - Talker forward passes use TRT-LLM (28L QKNormRoPEAttention + GatedMLP + RMSNorm)
  - CodePredictor uses HF generate() (to be ported to TRT-LLM later)
  - Embedding prep and vocoder use HF (non-transformer ops)
  - Generation loop matches HF pipeline exactly
- First successful audio: 15.68s, RTF=1.526 (slow due to no KV cache)
- Design doc saved to /home/scratch.kanghwanj_coreai_1/n/docs/qwen3-tts-trtllm-design.md
