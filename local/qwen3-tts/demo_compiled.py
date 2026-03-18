"""Qwen3-TTS demo with torch.compile optimization.

Applies torch.compile to the Talker and CodePredictor transformer layers
for faster autoregressive generation.
"""

import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_PATH = "/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE = "cuda:0"
OUTPUT_DIR = "/home/scratch.kanghwanj_coreai_1/n/rt/TensorRT-LLM/local/qwen3-tts"


def compile_model(tts_model):
    """Apply torch.compile to the Talker and CodePredictor transformer layers."""
    talker = tts_model.model.talker
    code_pred = talker.code_predictor

    # Compile the Talker's transformer model (28 layers)
    # Use "default" mode since "reduce-overhead" (CUDA graphs) conflicts
    # with dynamic shapes in the autoregressive generation loop
    talker.model = torch.compile(talker.model, mode="default", fullgraph=False)

    # Compile the CodePredictor's transformer model (5 layers)
    code_pred.model = torch.compile(code_pred.model, mode="default", fullgraph=False)

    print("Applied torch.compile to Talker and CodePredictor")
    return tts_model


def profile_generate(tts, text, speaker, language, warmup=1, runs=3):
    """Profile TTS generation with warmup and multiple runs."""
    for i in range(warmup):
        print(f"  Warmup {i + 1}/{warmup}...")
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=2048,
        )

    times = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=2048,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        duration = len(wavs[0]) / sr
        rtf = elapsed / duration if duration > 0 else float("inf")
        times.append(elapsed)
        print(f"  Run {i + 1}: {elapsed:.2f}s, audio={duration:.2f}s, RTF={rtf:.3f}")

    avg = np.mean(times)
    print(f"  Average: {avg:.2f}s")
    return wavs, sr


def main():
    print(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=DEVICE,
        dtype=torch.bfloat16,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Apply torch.compile
    compile_model(tts)

    # --- Short English ---
    print("\n--- Short English (compiled) ---")
    wavs, sr = profile_generate(
        tts,
        text="Hello! This is a test of Qwen3 TTS with torch compile optimization.",
        speaker="Vivian",
        language="English",
        warmup=2,
        runs=3,
    )
    sf.write(f"{OUTPUT_DIR}/compiled_short_en.wav", wavs[0], sr)

    # --- Medium English ---
    print("\n--- Medium English (compiled) ---")
    wavs, sr = profile_generate(
        tts,
        text=(
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the English alphabet."
        ),
        speaker="Ryan",
        language="English",
        warmup=0,
        runs=3,
    )
    sf.write(f"{OUTPUT_DIR}/compiled_medium_en.wav", wavs[0], sr)

    # --- Chinese ---
    print("\n--- Chinese (compiled) ---")
    wavs, sr = profile_generate(
        tts,
        text="你好，欢迎使用通义千问语音合成系统。今天天气真不错。",
        speaker="Vivian",
        language="Chinese",
        warmup=0,
        runs=3,
    )
    sf.write(f"{OUTPUT_DIR}/compiled_chinese.wav", wavs[0], sr)

    print("\nDone! Audio files saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
