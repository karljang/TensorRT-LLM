# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS demo using TRT-LLM optimized modules for both transformers.

Pipeline (matches HF exactly):
  1. Prepare input_embeds (text + codec tokens + speaker)
  2. Prefill: Talker.forward(input_embeds)           [TRT-LLM 28L]
  3. Decode loop:
     a. Sample codebook-0 token
     b. CodePredictor.forward(hidden, codebook-0)    [TRT-LLM 5L]
     c. Sum 16 codebook embeds + text embed
     d. Talker.forward(next_embed)                   [TRT-LLM 28L]
     Repeat until EOS
  4. Decode audio: 16 codebooks → 24kHz waveform     [HF vocoder]

Usage:
  python local/qwen3-tts/demo_trtllm_pipeline.py
"""

import json
import time

import safetensors.torch
import soundfile as sf
import torch

# HF imports (embedding prep + speech tokenizer)
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

# TRT-LLM imports
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_qwen3_tts import Qwen3TTSTalkerForCausalLM

TALKER_CKPT = "/home/scratch.kanghwanj_coreai_1/n/rt/TensorRT-LLM/local/qwen3-tts/talker_checkpoint"
MODEL_PATH = "/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = "/home/scratch.kanghwanj_coreai_1/n/rt/TensorRT-LLM/local/qwen3-tts"
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16


def create_attn_metadata(seq_len, backend="VANILLA"):
    """Create AttentionMetadata for a prefill forward pass."""
    metadata_cls = get_attention_backend(backend).Metadata
    metadata = metadata_cls(
        max_num_requests=1,
        max_num_tokens=32768,
        kv_cache_manager=None,
        request_ids=[1],
        prompt_lens=[seq_len],
        seq_lens=torch.tensor([seq_len], dtype=torch.int),
        num_contexts=1,
    )
    metadata.max_seq_len = seq_len
    metadata.prepare()
    return metadata


def load_trtllm_talker():
    """Load the Talker model using TRT-LLM modules."""
    print("[TRT-LLM] Loading Talker model...")
    with open(f"{TALKER_CKPT}/config.json") as f:
        raw = json.load(f)

    # Build config, filtering to known fields
    known_fields = set(Qwen3TTSTalkerConfig.__init__.__code__.co_varnames)
    cfg_kwargs = {k: v for k, v in raw.items() if k in known_fields}
    talker_cfg = Qwen3TTSTalkerConfig(**cfg_kwargs)
    talker_cfg.torch_dtype = DTYPE

    model_config = ModelConfig(pretrained_config=talker_cfg, attn_backend="VANILLA")
    model = Qwen3TTSTalkerForCausalLM(model_config).to(DTYPE).to(DEVICE)

    # Load weights
    weights = safetensors.torch.load_file(f"{TALKER_CKPT}/model.safetensors", device="cpu")
    talker_weights = {
        k: v for k, v in weights.items() if k.startswith("model.") or k.startswith("lm_head.")
    }
    model.load_weights(talker_weights)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Talker: {params:.0f}M params, {len(list(model.named_parameters()))} tensors")
    return model


def load_hf_model():
    """Load the full HF model for embedding prep and speech tokenizer."""
    print("[HF] Loading full model (for embeddings + vocoder)...")
    tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map=DEVICE, dtype=DTYPE)
    print("  HF model loaded")
    return tts


@torch.inference_mode()
def trtllm_talker_forward(model, embeds, position_ids):
    """Run TRT-LLM Talker forward pass."""
    seq_len = embeds.shape[0]
    attn_metadata = create_attn_metadata(seq_len)
    hidden = model.model(
        attn_metadata=attn_metadata,
        inputs_embeds=embeds,
        position_ids=position_ids,
    )
    return hidden


@torch.inference_mode()
def generate_with_trtllm(
    trtllm_talker, hf_tts, text, speaker, language, max_new_tokens=2048, temperature=0.9, top_k=50
):
    """Full TTS generation: HF embedding prep → TRT-LLM generation → HF vocoder.

    The generation loop matches HF exactly:
    - Prefill uses the same input_embeds preparation as HF
    - Each decode step runs CodePredictor (HF) then Talker (TRT-LLM)
    - Audio decoding uses HF speech tokenizer
    """
    hf_model = hf_tts.model  # Qwen3TTSForConditionalGeneration
    hf_talker = hf_model.talker  # Qwen3TTSTalkerForConditionalGeneration
    processor = hf_tts.processor
    config = hf_model.config

    # ── Step 1: Prepare input_embeds (reuse HF logic exactly) ──
    # Tokenize
    input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = processor(text=input_text, return_tensors="pt", padding=True)["input_ids"].to(
        DEVICE
    )
    input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    # Build embedding (same as Qwen3TTSForConditionalGeneration.generate)
    spk_id = config.talker_config.spk_id[speaker.lower()]
    speaker_embed = hf_talker.get_input_embeddings()(
        torch.tensor(spk_id, device=DEVICE, dtype=input_ids.dtype)
    )

    lang_lower = language.lower()
    if lang_lower == "auto":
        language_id = None
    else:
        language_id = config.talker_config.codec_language_id.get(lang_lower)

    tts_bos_embed, tts_eos_embed, tts_pad_embed = hf_talker.text_projection(
        hf_talker.get_text_embeddings()(
            torch.tensor(
                [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
                device=DEVICE,
                dtype=input_ids.dtype,
            )
        )
    ).chunk(3, dim=1)

    # Codec prefix tokens
    if language_id is None:
        codec_prefill_list = [
            [
                config.talker_config.codec_nothink_id,
                config.talker_config.codec_think_bos_id,
                config.talker_config.codec_think_eos_id,
            ]
        ]
    else:
        codec_prefill_list = [
            [
                config.talker_config.codec_think_id,
                config.talker_config.codec_think_bos_id,
                language_id,
                config.talker_config.codec_think_eos_id,
            ]
        ]

    codec_input_0 = hf_talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=DEVICE, dtype=input_ids.dtype)
    )
    codec_input_1 = hf_talker.get_input_embeddings()(
        torch.tensor(
            [[config.talker_config.codec_pad_id, config.talker_config.codec_bos_id]],
            device=DEVICE,
            dtype=input_ids.dtype,
        )
    )
    codec_input = torch.cat([codec_input_0, speaker_embed.view(1, 1, -1), codec_input_1], dim=1)

    # Role tokens: <|im_start|>assistant\n
    role_embed = hf_talker.text_projection(hf_talker.get_text_embeddings()(input_ids[:, :3]))

    # TTS pad + bos aligned with codec prefix
    tts_embed = (
        torch.cat(
            (
                tts_pad_embed.expand(-1, codec_input.shape[1] - 2, -1),
                tts_bos_embed,
            ),
            dim=1,
        )
        + codec_input[:, :-1]
    )

    talker_input = torch.cat((role_embed, tts_embed), dim=1)

    # First text token + codec_bos
    talker_input = torch.cat(
        [
            talker_input,
            hf_talker.text_projection(hf_talker.get_text_embeddings()(input_ids[:, 3:4]))
            + codec_input[:, -1:],
        ],
        dim=1,
    )

    # Trailing text (streaming)
    trailing_text_hidden = torch.cat(
        (
            hf_talker.text_projection(hf_talker.get_text_embeddings()(input_ids[:, 4:-5])),
            tts_eos_embed,
        ),
        dim=1,
    )

    print(f"  Input embeds: {talker_input.shape}")
    print(f"  Trailing text: {trailing_text_hidden.shape}")

    # ── Step 2: Prefill with TRT-LLM Talker ──
    input_embeds = talker_input.squeeze(0)  # (seq_len, hidden)
    seq_len = input_embeds.shape[0]
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

    hidden = trtllm_talker_forward(trtllm_talker, input_embeds, position_ids)
    codec_head = hf_talker.codec_head
    logits = codec_head(hidden[-1:])

    # ── Step 3: Decode loop ──
    all_codec_ids = []
    all_embeds = [input_embeds]  # Accumulate for recomputation (no KV cache yet)
    generation_step = 0
    codec_eos = config.talker_config.codec_eos_token_id
    num_code_groups = config.talker_config.num_code_groups

    for step in range(max_new_tokens):
        # 3a. Sample codebook-0
        token_logits = logits[0] / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(token_logits, min(top_k, token_logits.shape[-1]))
            token_logits[token_logits < topk_vals[-1]] = float("-inf")
        probs = torch.softmax(token_logits, dim=-1)
        codec_0 = torch.multinomial(probs, 1)

        if codec_0.item() == codec_eos:
            break

        # 3b. Run CodePredictor (HF) for codebooks 1-15
        past_hidden = hidden[-1:]  # last Talker hidden
        code_pred = hf_talker.code_predictor
        cp_result = code_pred.generate(
            inputs_embeds=torch.cat(
                (past_hidden.unsqueeze(0), hf_talker.get_input_embeddings()(codec_0).unsqueeze(0)),
                dim=1,
            ),
            max_new_tokens=num_code_groups - 1,
            do_sample=True,
            top_k=top_k,
            top_p=1.0,
            temperature=temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        all_codes = torch.cat((codec_0.unsqueeze(0), cp_result.sequences), dim=-1)

        # 3c. Sum all codebook embeddings
        codec_hiddens = hf_talker.get_input_embeddings()(codec_0)
        for g in range(num_code_groups - 1):
            codec_hiddens = codec_hiddens + code_pred.get_input_embeddings()[g](
                cp_result.sequences[..., g : g + 1].squeeze(0)
            )
        next_embed = codec_hiddens.unsqueeze(0)

        # Add trailing text
        if generation_step < trailing_text_hidden.shape[1]:
            next_embed = next_embed + trailing_text_hidden[:, generation_step : generation_step + 1]
        else:
            next_embed = next_embed + tts_pad_embed

        # 3d. TRT-LLM Talker forward (full recomputation — no KV cache)
        all_embeds.append(next_embed.squeeze(0))
        full_embeds = torch.cat(all_embeds, dim=0)
        full_pos = torch.arange(full_embeds.shape[0], device=DEVICE).unsqueeze(0)

        hidden = trtllm_talker_forward(trtllm_talker, full_embeds, full_pos)
        logits = codec_head(hidden[-1:])

        all_codec_ids.append(all_codes.squeeze(0))
        generation_step += 1

        if (step + 1) % 10 == 0:
            print(
                f"    Step {step + 1}: {len(all_codec_ids)} frames ({len(all_codec_ids) * 0.08:.1f}s)"
            )

    if not all_codec_ids:
        print("  No frames generated!")
        return None, 0

    # ── Step 4: Decode audio with HF speech tokenizer ──
    codec_ids = torch.stack(all_codec_ids, dim=0)  # (num_frames, num_code_groups)
    print(f"  Generated {codec_ids.shape[0]} frames ({codec_ids.shape[0] * 0.08:.2f}s audio)")

    wavs, sr = hf_model.speech_tokenizer.decode([{"audio_codes": codec_ids}])
    return wavs[0], sr


def main():
    print("=" * 60)
    print("Qwen3-TTS Demo — TRT-LLM Pipeline")
    print("=" * 60)

    # Load models
    trtllm_talker = load_trtllm_talker()
    hf_tts = load_hf_model()

    # Demo prompt — hackathon project introduction
    text = (
        "Hi! This audio is generated by Qwen3 TTS, powered by TensorRT LLM. "
        "We built this for the Agentic Coding Hackathon, using Claude Code as our AI coding assistant. "
        "The team is Kanghwan, Claude, and Michal Guzek. "
        "Both the Talker and Code Predictor transformers run on TensorRT LLM's optimized modules. "
        "Thank you!"
    )
    speaker = "Ryan"
    language = "English"
    print(f"\n[Generate] '{text}' (speaker={speaker}, lang={language})")

    torch.cuda.synchronize()
    t0 = time.time()
    audio, sr = generate_with_trtllm(
        trtllm_talker, hf_tts, text, speaker, language, max_new_tokens=300
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    if audio is not None:
        duration = len(audio) / sr
        rtf = elapsed / duration
        out_path = f"{OUTPUT_DIR}/demo_trtllm_pipeline_output.wav"
        sf.write(out_path, audio, sr)
        print(f"\n  Audio: {duration:.2f}s | Time: {elapsed:.2f}s | RTF: {rtf:.3f}")
        print(f"  Saved: {out_path}")
    else:
        print("  Generation failed")


if __name__ == "__main__":
    main()
