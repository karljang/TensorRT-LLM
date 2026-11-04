"""Many H3 generations from one loaded engine: the §640 lever-7 reference-free ship check.

Same variant spec, provenance and output layout as gen_quality.py, but the engine is built once per
arm and every prompt of a prompt-set file is generated in turn:

    python gen_quality_batch.py --model <dir> --prompts requests/ship-prompts-544x960x124.json \
        --variant CUDNN:q=fp8:v=fp8 --root <quality-root> --arm nvfp4-cudnn-fp8

Writes <root>/<prompt_id>/<arm>/output.pt (video uint8 [1, F, H, W, C] or as returned, audio) and
result.json, plus <root>/<arm>.provenance.json once; <root>/<prompt_id>/ then matches the per-seed
layout export_clips.py and make_clips.py consume. Prompts already generated are skipped (resume).
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h3_ungate  # noqa: E402  study root, after the sys.path insert
from gen_quality import _installed, parse_variant  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True, help="prompt-set JSON (size, steps, seed, prompts{id: text})")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--only", nargs="*", default=None, help="prompt ids to generate (default: all)")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="off")
    parser.add_argument("--ulysses", type=int, default=1, help="Ulysses size = parallel VAE size (the engine's worker pool shards; 1 = single GPU)")
    parser.add_argument("--async-ulysses", action="store_true", help="parallel_config.async_ulysses (needs the #20333 wiring in the runtime)")
    args = parser.parse_args()
    backend, quant, want_2cta, sparse = parse_variant(args.variant)
    if want_2cta:
        os.environ["FA_DISABLE_2CTA"] = "0"
    spec = json.loads(args.prompts.read_text())
    prompts = {k: v for k, v in spec["prompts"].items() if not args.only or k in args.only}

    import torch

    import tensorrt_llm
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

    cudnn_ungate = h3_ungate.apply()
    attention_config = {"backend": backend}
    if quant is not None:
        attention_config["quant_attention_config"] = quant
    if sparse is not None:
        attention_config["sparse_attention_config"] = sparse
    provenance = {
        "arm": args.arm, "variant": args.variant, "backend": backend, "quant_attention_config": quant, "sparse_attention_config": sparse,
        "fa_disable_2cta": os.environ.get("FA_DISABLE_2CTA"), "cudnn_ungate": cudnn_ungate, "model": args.model,
        "study_overrides": {k: os.environ.get(k) for k in ("H3_UNGATE_CUDNN", "H3_UNGATE_TRTLLM", "H3_UNGATE_SAGE", "TLLM_FMHA_LIBS", "CUTE_DSL_ARCH", "H3_SOL_STATS")},
        "prompt_set": str(args.prompts), "prompt_ids": sorted(prompts), "seed": spec["seed"],
        "steps": spec["num_inference_steps"], "compile": args.compile, "cuda_graph": args.cuda_graph,
        "ulysses": args.ulysses, "async_ulysses": args.async_ulysses, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "size": [spec["height"], spec["width"], spec["num_frames"]],
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "host": platform.node(),
        "arch": platform.machine(), "tensorrt_llm": tensorrt_llm.__version__, "tensorrt_llm_file": tensorrt_llm.__file__,
        "h3": h3.__file__, "h3_sha256": hashlib.sha256(Path(h3.__file__).read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ("nvidia-cutlass-dsl", "diffusers", "triton", "flashinfer-python", "nvidia-cudnn-frontend")
                     if _installed(p)},
        "sys_path": sys.path,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / f"{args.arm}.provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: v for k, v in provenance.items() if k not in ("sys_path",)}), flush=True)

    vg_kwargs = {}
    if args.ulysses > 1:
        parallel = {"ulysses_size": args.ulysses, "parallel_vae_size": args.ulysses}
        if args.async_ulysses:
            parallel["async_ulysses"] = True
        vg_kwargs["parallel_config"] = parallel
    vg_args = VisualGenArgs(
        model=args.model,
        attention_config=attention_config,
        torch_compile_config={"enable": args.compile == "on"},
        cuda_graph_config={"enable": args.cuda_graph == "on"},
        compilation_config={"resolutions": [[spec["height"], spec["width"]]], "num_frames": [spec["num_frames"]]},
        **vg_kwargs,
    )
    t0 = time.time()
    engine = VisualGen(model=vg_args.model, args=vg_args)
    print(f"engine loaded in {time.time() - t0:.0f} s", flush=True)
    done = 0
    try:
        for prompt_id, prompt in prompts.items():
            out_dir = args.root / prompt_id / args.arm
            if (out_dir / "output.pt").exists():
                print(f"[{prompt_id}] exists, skipped", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            output = engine.generate(
                inputs=prompt,
                params=VisualGenParams(height=spec["height"], width=spec["width"], num_frames=spec["num_frames"],
                                       num_inference_steps=spec["num_inference_steps"], seed=spec["seed"]),
            )
            elapsed = time.time() - t0
            video = torch.as_tensor(output.video).cpu()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            audio = None if getattr(output, "audio", None) is None else torch.as_tensor(output.audio).cpu()
            torch.save({"video": video, "audio": audio}, out_dir / "output.pt")
            (out_dir / "result.json").write_text(json.dumps({"status": "ok", "generate_seconds": elapsed, "prompt": prompt,
                                                             "video_shape": list(video.shape), "video_dtype": str(video.dtype)}))
            done += 1
            print(f"[{prompt_id}] {args.arm} generated in {elapsed:.1f} s", flush=True)
    finally:
        engine.shutdown()
    print("H3_QUALITY_BATCH_DONE", done, flush=True)


if __name__ == "__main__":
    main()
