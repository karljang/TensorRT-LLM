"""Qwen3-TTS profiled baseline - measures time per component."""

import time

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_PATH = "/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE = "cuda:0"
OUTPUT_DIR = "/home/scratch.kanghwanj_coreai_1/n/rt/TensorRT-LLM/local/qwen3-tts"


def profile_generate(tts, text, speaker, language, warmup=1, runs=3):
    """Profile TTS generation with warmup and multiple runs."""
    # Warmup
    for _ in range(warmup):
        wavs, sr = tts.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            max_new_tokens=2048,
        )

    # Timed runs
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
        t1 = time.time()
        duration = len(wavs[0]) / sr
        elapsed = t1 - t0
        rtf = elapsed / duration
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

    # Print model sizes
    talker = tts.model.talker
    code_pred = talker.code_predictor
    talker_params = sum(p.numel() for p in talker.parameters()) / 1e6
    code_pred_params = sum(p.numel() for p in code_pred.parameters()) / 1e6
    print(f"\nTalker params: {talker_params:.1f}M")
    print(f"CodePredictor params: {code_pred_params:.1f}M")

    # Short English
    print("\n--- Short English ---")
    wavs, sr = profile_generate(
        tts,
        text="Hello! This is a test of Qwen3 TTS.",
        speaker="Vivian",
        language="English",
        warmup=1,
        runs=3,
    )
    sf.write(f"{OUTPUT_DIR}/profiled_short_en.wav", wavs[0], sr)

    # Medium English
    print("\n--- Medium English ---")
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
    sf.write(f"{OUTPUT_DIR}/profiled_medium_en.wav", wavs[0], sr)

    # Chinese
    print("\n--- Chinese ---")
    wavs, sr = profile_generate(
        tts,
        text="你好，欢迎使用通义千问语音合成系统。今天天气真不错。",
        speaker="Vivian",
        language="Chinese",
        warmup=0,
        runs=3,
    )
    sf.write(f"{OUTPUT_DIR}/profiled_chinese.wav", wavs[0], sr)

    print("\nDone! Audio files saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
