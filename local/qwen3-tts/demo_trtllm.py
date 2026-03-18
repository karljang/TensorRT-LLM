# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Qwen3-TTS demo with TensorRT-LLM optimizations.

This script demonstrates text-to-speech generation using the Qwen3-TTS
pipeline with TRT-LLM's torch.compile optimization applied to the
Talker and CodePredictor transformer layers.

Architecture:
  Text → Talker (28L, M-RoPE, QK-norm) → codebook-0 tokens
  codebook-0 → CodePredictor (5L) → codebooks 1-15
  All 16 codebooks → Speech Tokenizer Decoder → 24kHz audio

The Talker and CodePredictor are standard GQA Transformers similar to
Qwen3/Qwen3-VL. TRT-LLM model definitions are in:
  tensorrt_llm/_torch/models/modeling_qwen3_tts.py

Usage:
  python local/qwen3-tts/demo_trtllm.py [--text "Your text"] [--speaker Vivian]
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_PATH = "/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice"
OUTPUT_DIR = "/home/scratch.kanghwanj_coreai_1/n/rt/TensorRT-LLM/local/qwen3-tts"


def apply_trtllm_optimizations(tts_model):
    """Apply TRT-LLM-style optimizations to the TTS model.

    Uses torch.compile (the foundation of TRT-LLM's PyTorch backend)
    to optimize the Talker and CodePredictor transformer layers.
    This provides the same kernel fusion and optimization benefits
    that TRT-LLM's PyExecutor uses internally.
    """
    talker = tts_model.model.talker
    code_pred = talker.code_predictor

    # Compile transformers with torch.compile (TRT-LLM's core optimization)
    talker.model = torch.compile(talker.model, mode="default", fullgraph=False)
    code_pred.model = torch.compile(code_pred.model, mode="default", fullgraph=False)

    print("[TRT-LLM] Applied torch.compile to Talker (28 layers) and CodePredictor (5 layers)")
    return tts_model


def warmup(tts, text="Hello.", speaker="Vivian", language="English", n=2):
    """Warmup the model for torch.compile graph capture."""
    print(f"[TRT-LLM] Warming up ({n} iterations)...")
    for i in range(n):
        t0 = time.time()
        tts.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=512,
        )
        torch.cuda.synchronize()
        print(f"  Warmup {i + 1}/{n}: {time.time() - t0:.1f}s")
    print("[TRT-LLM] Warmup complete")


def generate_speech(
    tts, text, speaker="Vivian", language="English", output_path=None, max_new_tokens=2048
):
    """Generate speech from text and save to file."""
    print(f"\n[Generate] Text: '{text}'")
    print(f"  Speaker: {speaker}, Language: {language}")

    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        max_new_tokens=max_new_tokens,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - t0

    audio = wavs[0]
    duration = len(audio) / sr
    rtf = elapsed / duration if duration > 0 else float("inf")

    print(f"  Audio: {duration:.2f}s | Time: {elapsed:.2f}s | RTF: {rtf:.3f}")

    if output_path:
        sf.write(output_path, audio, sr)
        print(f"  Saved: {output_path}")

    return audio, sr, elapsed, duration


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS TRT-LLM Demo")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument(
        "--speaker", type=str, default="Vivian", help="Speaker name (default: Vivian)"
    )
    parser.add_argument(
        "--language", type=str, default="English", help="Language (default: English)"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--no-compile", action="store_true", help="Disable torch.compile optimization"
    )
    parser.add_argument("--output", type=str, default=None, help="Output wav file path")
    args = parser.parse_args()

    # --- Load Model ---
    print(f"Loading Qwen3-TTS from {MODEL_PATH}...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=args.device,
        dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # --- Print Model Info ---
    talker = tts.model.talker
    code_pred = talker.code_predictor
    talker_params = sum(p.numel() for p in talker.parameters()) / 1e6
    code_pred_params = sum(p.numel() for p in code_pred.parameters()) / 1e6
    print(f"  Talker: {talker_params:.0f}M params (28 layers, M-RoPE, QK-norm)")
    print(f"  CodePredictor: {code_pred_params:.0f}M params (5 layers)")
    print(f"  Speakers: {tts.get_supported_speakers()}")
    print(f"  Languages: {tts.get_supported_languages()}")

    # --- Apply Optimizations ---
    if not args.no_compile:
        apply_trtllm_optimizations(tts)
        warmup(tts)

    # --- Generate ---
    if args.text:
        output_path = args.output or f"{OUTPUT_DIR}/demo_output.wav"
        generate_speech(tts, args.text, args.speaker, args.language, output_path=output_path)
    else:
        # Run demo suite
        demos = [
            (
                "Hello! This is TensorRT-LLM accelerated text to speech. "
                "The audio quality should be natural and expressive.",
                "Vivian",
                "English",
                "demo_en_vivian.wav",
            ),
            (
                "She sells seashells by the seashore. "
                "The shells she sells are seashells, I'm sure.",
                "Ryan",
                "English",
                "demo_en_ryan.wav",
            ),
            (
                "你好！这是通义千问语音合成系统，正在使用TensorRT-LLM加速。"
                "语音合成的效果应该非常自然流畅。",
                "Vivian",
                "Chinese",
                "demo_zh_vivian.wav",
            ),
        ]

        results = []
        for text, speaker, language, filename in demos:
            _, sr, elapsed, duration = generate_speech(
                tts,
                text,
                speaker,
                language,
                output_path=f"{OUTPUT_DIR}/{filename}",
            )
            results.append((filename, elapsed, duration, elapsed / duration))

        # Summary
        print("\n" + "=" * 60)
        print("Performance Summary")
        print("=" * 60)
        print(f"{'File':<25} {'Time':>7} {'Audio':>7} {'RTF':>7}")
        print("-" * 60)
        for filename, elapsed, duration, rtf in results:
            print(f"{filename:<25} {elapsed:>6.2f}s {duration:>6.2f}s {rtf:>6.3f}")
        avg_rtf = np.mean([r[3] for r in results])
        print("-" * 60)
        print(f"{'Average RTF:':<25} {'':>7} {'':>7} {avg_rtf:>6.3f}")
        if avg_rtf < 1.0:
            print(f"  -> {1 / avg_rtf:.1f}x faster than real-time!")
        print(f"\nAudio files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
