"""Qwen3-TTS HuggingFace baseline demo.

Generates audio from text using the CustomVoice model with preset speakers.
"""

import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

MODEL_PATH = "/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEVICE = "cuda:0"
OUTPUT_DIR = "/home/scratch.kanghwanj_coreai_1/n/rt/TensorRT-LLM/local/qwen3-tts"


def main():
    print(f"Loading model from {MODEL_PATH}...")
    t0 = time.time()
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=DEVICE,
        dtype=torch.bfloat16,
    )
    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.1f}s")

    # --- Single generation ---
    text = "Hello! This is a test of the Qwen3 text to speech system running on TensorRT LLM."
    speaker = "Vivian"
    language = "English"

    print(f"\nGenerating: '{text}'")
    print(f"Speaker: {speaker}, Language: {language}")

    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text=text,
        language=language,
        speaker=speaker,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t_gen = time.time() - t0

    out_path = f"{OUTPUT_DIR}/demo_baseline_output.wav"
    sf.write(out_path, wavs[0], sr)

    duration = len(wavs[0]) / sr
    rtf = t_gen / duration if duration > 0 else float("inf")
    print(f"Generated {duration:.2f}s audio in {t_gen:.2f}s (RTF={rtf:.3f})")
    print(f"Saved to: {out_path}")

    # --- List supported speakers ---
    speakers = tts.get_supported_speakers()
    print(f"\nSupported speakers: {speakers}")


if __name__ == "__main__":
    main()
