# Qwen3-TTS TensorRT-LLM Integration Worklog

## Task
Add Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) support to TensorRT-LLM.

## User Instructions
- It is NOT an LLM, but a TTS pipeline with multiple models
- Use %11 for GPU access
- Final artifact: a script demonstrating TTS producing audio from a given prompt
- Write this worklog.md with all instructions and work done
- Be autonomous: plan tasks, make checklist, complete independently
- Always use `pip install --no-deps` to avoid breaking NVIDIA custom torch
- Log progress frequently

## Model Architecture
**Qwen3-TTS** is a unified end-to-end TTS system with 3 components:

1. **Talker Model** (28-layer Transformer, ~1.7B params)
   - Text tokens → Codebook-0 (semantic) speech tokens (autoregressive)
   - 2048 hidden size, 16 attention heads, 8 KV heads (GQA)
   - M-RoPE positional encoding (sections [24, 20, 20], interleaved=true)
   - QK-norm (RMSNorm on head_dim)
   - Two embeddings: codec_embedding (vocab=3072) + text_embedding (vocab=151936)
   - Text vocab projected via text_projection MLP (text_hidden_size → hidden_size)
   - Text vocab: 151,936 | Speech vocab: 3,072

2. **Code Predictor** (5-layer Transformer)
   - Codebook-0 → Codebooks 1-15 (autoregressive multi-token prediction)
   - 1024 hidden size, 16 attention heads, 8 KV heads
   - QK-norm (RMSNorm on head_dim)
   - 16 code groups, 2048 vocab per group
   - 15 separate lm_head Linear layers + 15 separate embedding layers
   - small_to_mtp_projection: talker_hidden_size → code_predictor_hidden_size

3. **Speech Tokenizer Decoder** (Causal ConvNet vocoder)
   - 16 codebook codes per frame → 24kHz waveform
   - Frame rate: 12.5 Hz (80ms per frame)
   - Lightweight, non-diffusion

### Key Pipeline Flow
```
Text Input → Tokenizer → [Talker autoregressively generates codebook-0 tokens]
  At each Talker decode step:
    1. Previous codebook-0 token → CodePredictor generates codebooks 1-15
    2. Sum all 16 codebook embeddings → next Talker input
    3. Add streaming text embedding (trailing_text_hidden)
  → Talker outputs all 16 codebook codes per frame
  → Speech Tokenizer Decoder → 24kHz audio
```

## Checklist

### Phase 0: Setup & Research
- [x] Research model architecture from HuggingFace
- [x] Explore TRT-LLM codebase for existing TTS/audio patterns
- [x] Read HF modeling source code (`/home/scratch.kanghwanj_coreai_1/n/pub/Qwen3-TTS/`)
- [x] Set up GPU environment on %11 (Docker with .venv-3.12)
- [x] Model weights at `/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice/`

### Phase 1: Baseline HF Inference
- [x] Install qwen_tts package (patched 25hz tokenizer import for torchaudio compatibility)
- [x] Get HF transformers Qwen3-TTS running on GPU
- [x] Profile baseline performance (RTF ~1.1-1.6x, slower than real-time)
- [x] Generate audio samples (English + Chinese)

### Phase 2: torch.compile Optimization
- [x] Apply torch.compile to Talker and CodePredictor transformer layers
- [x] Benchmark compiled version (RTF ~0.55, faster than real-time!)
- [x] Validate audio output quality

### Phase 3: TRT-LLM Model Registration
- [x] Create `tensorrt_llm/_torch/models/modeling_qwen3_tts.py`
  - Qwen3TTSTalkerAttention (QKNormRoPEAttention with M-RoPE)
  - Qwen3TTSTalkerDecoderLayer, Qwen3TTSTalkerModel, Qwen3TTSTalkerForCausalLM
  - Qwen3TTSCodePredictorAttention, Qwen3TTSCodePredictorDecoderLayer, Qwen3TTSCodePredictorModel
- [x] Register Talker in `tensorrt_llm/_torch/models/__init__.py`
- [ ] Weight loading verification
- [ ] Full TRT-LLM executor integration (future work - requires custom generation loop)

### Phase 4: Demo Script
- [x] `demo_hf_baseline.py` - HF baseline (RTF ~1.3)
- [x] `demo_profiled.py` - Profiled baseline with multiple runs
- [x] `demo_compiled.py` - torch.compile optimized (RTF ~0.55)
- [ ] Final polished demo script

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

**Speedup: ~2-2.5x over HF baseline**, achieving **faster-than-real-time** (RTF < 1.0).

## Files Created

| File | Description |
|------|-------------|
| `local/qwen3-tts/worklog.md` | This worklog |
| `local/qwen3-tts/demo_hf_baseline.py` | HF baseline demo |
| `local/qwen3-tts/demo_profiled.py` | Profiled baseline with multiple runs |
| `local/qwen3-tts/demo_compiled.py` | torch.compile optimized demo |
| `tensorrt_llm/_torch/models/modeling_qwen3_tts.py` | TRT-LLM model definitions |

## Environment Patches
- Patched `/home/scratch.kanghwanj_coreai_1/n/pub/Qwen3-TTS/qwen_tts/core/__init__.py`:
  Made 25hz tokenizer import optional (requires torchaudio, incompatible with NVIDIA torch)
- Patched `/home/scratch.kanghwanj_coreai_1/n/pub/Qwen3-TTS/qwen_tts/inference/qwen3_tts_tokenizer.py`:
  Made V1 tokenizer import and registration optional

## Work Log

### 2026-03-18: Research & Setup
- Analyzed Qwen3-TTS architecture from HuggingFace model card and config.json
- Explored TRT-LLM codebase: found visual_gen pipeline pattern and existing Qwen models
- No existing TTS support in TRT-LLM; visual_gen provides pipeline template
- Key finding: Talker is similar to Qwen3 (GQA + QK-norm) with M-RoPE from Qwen3-VL
- Set up GPU environment on %11 (Docker trtllm-dev, .venv-3.12)
- Installed qwen_tts package with --no-deps
- Fixed torchaudio incompatibility by making 25hz tokenizer import optional

### 2026-03-18: Baseline & Optimization
- Got HF baseline working, generating audio successfully
- Profiled performance: RTF ~1.1-1.6x (slower than real-time)
- Applied torch.compile: achieved RTF ~0.55 (2-2.5x speedup, faster than real-time!)
- Studied full generate() flow: nested loop where CodePredictor runs inside each Talker step
- This nested pattern means standard TRT-LLM executor can't manage the generation directly

### 2026-03-18: TRT-LLM Model Implementation
- Created `modeling_qwen3_tts.py` with both Talker and CodePredictor models
- Talker uses QKNormRoPEAttention with M-RoPE (like Qwen3-VL)
- CodePredictor uses QKNormRoPEAttention with standard RoPE
- Registered Talker in `__init__.py`
- Note: Full executor integration requires custom generation loop (future work)

## Architecture Notes

### Why Full TRT-LLM Executor Integration is Non-Trivial
The Qwen3-TTS Talker has a custom forward() that runs the CodePredictor at each decode step:
1. Talker generates codebook-0 token via its main transformer
2. CodePredictor generates codebooks 1-15 from Talker's hidden state (another autoregressive loop)
3. All 16 codebook embeddings are summed for the next step's input
4. Streaming text embedding is added at each step

This "model-in-a-model" loop doesn't fit TRT-LLM's standard executor which manages a single generation loop. Options for deeper integration:
- Custom executor that orchestrates both models
- Fuse CodePredictor into Talker's forward as a "sub-execution"
- Use TRT-LLM's PyExecutor with custom hooks

### M-RoPE in TRT-LLM
TRT-LLM has full M-RoPE support via:
- `PositionEmbeddingType.mrope`
- `MRotaryEmbedding` class in `modules/rotary_embedding.py`
- Position IDs shape: (3, batch, seq) for temporal/height/width
- `mrope_section` controls head dimension splitting
- `mrope_interleaved` controls interleaved vs concatenated application
