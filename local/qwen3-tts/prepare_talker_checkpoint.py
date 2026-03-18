"""Extract Talker weights from Qwen3-TTS and create a standalone checkpoint.

Creates a directory that looks like a standard CausalLM checkpoint so that
TRT-LLM's `LLM(model=talker_dir)` can load it directly.

Output structure:
  talker_checkpoint/
    config.json            # CausalLM-compatible config
    model.safetensors      # Talker-only weights (remapped keys)
"""

import json
import os

import safetensors.torch

SRC_MODEL = "/home/scratch.trt_llm_data_ci/llm-models/Qwen3/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DST_DIR = "/home/scratch.kanghwanj_coreai_1/n/rt/TensorRT-LLM/local/qwen3-tts/talker_checkpoint"


def create_talker_config():
    """Create a CausalLM-compatible config.json for the Talker."""
    with open(f"{SRC_MODEL}/config.json") as f:
        full_cfg = json.load(f)

    talker = full_cfg["talker_config"]

    # Build a config that TRT-LLM's AutoModelForCausalLM can recognize.
    # The architecture name must match @register_auto_model in modeling_qwen3_tts.py
    config = {
        "architectures": ["Qwen3TTSTalkerForCausalLM"],
        "model_type": "qwen3_tts_talker",
        "vocab_size": talker["vocab_size"],  # 3072 (codec vocab)
        "hidden_size": talker["hidden_size"],  # 2048
        "intermediate_size": talker["intermediate_size"],  # 6144
        "num_hidden_layers": talker["num_hidden_layers"],  # 28
        "num_attention_heads": talker["num_attention_heads"],  # 16
        "num_key_value_heads": talker["num_key_value_heads"],  # 8
        "head_dim": talker.get("head_dim", 128),
        "hidden_act": talker.get("hidden_act", "silu"),
        "max_position_embeddings": talker.get("max_position_embeddings", 32768),
        "rms_norm_eps": talker.get("rms_norm_eps", 1e-6),
        "rope_theta": talker.get("rope_theta", 1000000),
        "rope_scaling": talker.get("rope_scaling"),
        "attention_bias": talker.get("attention_bias", False),
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        # Extra fields for the Talker
        "text_vocab_size": talker.get("text_vocab_size", 151936),
        "text_hidden_size": talker.get("text_hidden_size", 2048),
        # Code predictor sub-config (needed for full pipeline)
        "code_predictor_config": talker.get("code_predictor_config", {}),
        "num_code_groups": talker.get("num_code_groups", 16),
        # Codec special tokens
        "codec_eos_token_id": talker.get("codec_eos_token_id"),
        "codec_bos_id": talker.get("codec_bos_id"),
        "codec_pad_id": talker.get("codec_pad_id"),
        "codec_think_id": talker.get("codec_think_id"),
        "codec_nothink_id": talker.get("codec_nothink_id"),
        "codec_think_bos_id": talker.get("codec_think_bos_id"),
        "codec_think_eos_id": talker.get("codec_think_eos_id"),
        "codec_language_id": talker.get("codec_language_id"),
        "spk_id": talker.get("spk_id"),
    }

    return config


def extract_talker_weights():
    """Load full checkpoint and extract + remap Talker weights."""
    print(f"Loading weights from {SRC_MODEL}/model.safetensors ...")
    all_weights = safetensors.torch.load_file(f"{SRC_MODEL}/model.safetensors", device="cpu")
    print(f"  Total tensors: {len(all_weights)}")

    talker_weights = {}
    skipped = []

    for key, tensor in all_weights.items():
        if not key.startswith("talker."):
            skipped.append(key)
            continue

        # Remap: talker.model.X → model.X
        if key.startswith("talker.model."):
            new_key = "model." + key[len("talker.model.") :]
        # Remap: talker.codec_head.X → lm_head.X
        elif key.startswith("talker.codec_head."):
            new_key = "lm_head." + key[len("talker.codec_head.") :]
        # Keep text_projection and code_predictor as-is (under talker.)
        elif key.startswith("talker.text_projection."):
            new_key = key[len("talker.") :]  # text_projection.X
        elif key.startswith("talker.code_predictor."):
            new_key = key[len("talker.") :]  # code_predictor.X
        else:
            new_key = key[len("talker.") :]

        talker_weights[new_key] = tensor

    print(f"  Talker tensors: {len(talker_weights)}")
    print(f"  Skipped (non-talker): {len(skipped)}")
    if skipped:
        print(f"    e.g.: {skipped[:3]}")

    return talker_weights


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    # 1. Create config
    config = create_talker_config()
    config_path = f"{DST_DIR}/config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    print(f"  architecture: {config['architectures'][0]}")
    print(f"  layers: {config['num_hidden_layers']}, hidden: {config['hidden_size']}")
    print(f"  vocab: {config['vocab_size']} (codec), text_vocab: {config['text_vocab_size']}")

    # 2. Extract and save weights
    weights = extract_talker_weights()
    weights_path = f"{DST_DIR}/model.safetensors"
    safetensors.torch.save_file(weights, weights_path)
    total_bytes = os.path.getsize(weights_path)
    print(f"Saved weights to {weights_path} ({total_bytes / 1e9:.2f} GB)")

    # 3. Print weight key samples
    print("\nSample weight keys:")
    for i, key in enumerate(sorted(weights.keys())):
        if i < 10 or "lm_head" in key or "text_projection" in key:
            print(f"  {key}: {weights[key].shape}")
        if i == 15:
            print(f"  ... ({len(weights)} total)")
            break

    print(f"\nTalker checkpoint ready at: {DST_DIR}")


if __name__ == "__main__":
    main()
