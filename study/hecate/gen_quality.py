"""One H3 generation for the §628 quality gate: a named attention variant, same seed and request.

Records provenance, the per-step scheduler inputs (model velocity, sample) through
probe/sitecustomize.py (H3_PROBE_DIR on PYTHONPATH), the final video as output.pt and the
wall time. Compare runs with quality_compare.py.

Variant spec: BACKEND[:q=<qk_dtype>:v=<v_dtype>[:vblk=N]][:2cta=1]; '2cta=1' re-enables FA4's
2-CTA kernel by exporting FA_DISABLE_2CTA=0 before the backend module is imported.
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


def parse_variant(spec):
    """BACKEND[:q=..:v=..[:vblk=N]][:2cta=1][:sol=TAU[:dut=CUTOFF][:thresh=diag|exact][:dense=all|i,j]].

    `sol=` adds attention_config.sparse_attention_config = {algorithm: sol_attn, tau, disabled_until_timestep
    (`dut`, 'none' or absent = no dense prefix), thresh_type, dense_layers}; TRTLLM serves it with the predictor
    op + PrimsTS block-sparse FMHA, CUTEDSL with the fused kernel (lever 21, §657).
    """
    backend, *opts = spec.split(":")
    options = dict(opt.split("=", 1) for opt in opts)
    quant = None
    if "q" in options or "v" in options:
        quant = {"qk_dtype": options.get("q", "bf16"), "v_dtype": options.get("v", "fp8"),
                 "v_block_size": int(options.get("vblk", "0"))}
        if "blk" in options:  # lever 22: SageAttention block sizes q,k,v (TRTLLM backend), e.g. blk=1,16,1
            qb, kb, vb = (int(x) for x in options["blk"].split(","))
            quant.update({"q_block_size": qb, "k_block_size": kb, "v_block_size": vb})
    sparse = None
    if "sol" in options:
        sparse = {"algorithm": "sol_attn", "tau": float(options["sol"]), "thresh_type": options.get("thresh", "diag")}
        if options.get("dut", "none").lower() != "none":
            sparse["disabled_until_timestep"] = float(options["dut"])
        if "dense" in options:
            sparse["dense_layers"] = list(range(256)) if options["dense"] == "all" else [int(x) for x in options["dense"].split(",")]
    return backend.upper(), quant, options.get("2cta") == "1", sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", type=Path, required=True, help="request JSON (height, width, num_frames, prompt, ...)")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="off")
    parser.add_argument("--record-steps", choices=["on", "off"], default="on")
    parser.add_argument("--pipeline-warmup", choices=["on", "off"], default="off",
                        help="run the pipeline's own warmup before generating (AutoTuner tactics for quantized GEMMs, the shipping path); default off = skip_warmup")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    backend, quant, want_2cta, sparse = parse_variant(args.variant)
    if want_2cta:
        os.environ["FA_DISABLE_2CTA"] = "0"
    if args.record_steps == "on":
        os.environ["H3_PROBE_DIR"] = str(args.output / "steps")

    import torch

    import tensorrt_llm
    from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import transformer_minimax_h3 as h3

    cudnn_ungate = h3_ungate.apply()
    request = json.loads(args.request.read_text())
    attention_config = {"backend": backend}
    if quant is not None:
        attention_config["quant_attention_config"] = quant
    if sparse is not None:
        attention_config["sparse_attention_config"] = sparse
    provenance = {
        "variant": args.variant, "backend": backend, "quant_attention_config": quant, "sparse_attention_config": sparse,
        "fa_disable_2cta": os.environ.get("FA_DISABLE_2CTA"),
        "cudnn_ungate": cudnn_ungate, "h3_ungate_trtllm": os.environ.get("H3_UNGATE_TRTLLM"), "h3_ungate_sage": os.environ.get("H3_UNGATE_SAGE"), "tllm_fmha_libs": os.environ.get("TLLM_FMHA_LIBS"),
        "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
        "seed": args.seed, "steps": args.steps, "compile": args.compile, "cuda_graph": args.cuda_graph, "pipeline_warmup": args.pipeline_warmup, "request": request,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "host": platform.node(),
        "arch": platform.machine(), "tensorrt_llm": tensorrt_llm.__version__, "tensorrt_llm_file": tensorrt_llm.__file__,
        "h3": h3.__file__, "h3_sha256": hashlib.sha256(Path(h3.__file__).read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ("nvidia-cutlass-dsl", "diffusers", "triton", "flashinfer-python", "nvidia-cudnn-frontend")
                     if _installed(p)},
        "probe_dir": os.environ.get("H3_PROBE_DIR"), "sys_path": sys.path,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: v for k, v in provenance.items() if k not in ("sys_path", "request")}), flush=True)

    vg_args = VisualGenArgs(
        model=args.model,
        attention_config=attention_config,
        torch_compile_config={"enable": args.compile == "on"},
        cuda_graph_config={"enable": args.cuda_graph == "on"},
        compilation_config={"skip_warmup": args.pipeline_warmup == "off"},
    )
    engine = VisualGen(model=vg_args.model, args=vg_args)
    # The pipeline warmup runs its own scheduler steps; the probe records them too. Move those files out
    # so steps/ keeps the reference layout (one file per scheduler per sampler step of the request).
    warmup_steps = 0
    steps_dir = args.output / "steps"
    if args.pipeline_warmup == "on" and steps_dir.is_dir():
        warm = args.output / "steps-warmup"
        warm.mkdir(exist_ok=True)
        for f in sorted(steps_dir.glob("*.pt")):
            f.rename(warm / f.name)
            warmup_steps += 1
    try:
        t0 = time.time()
        output = engine.generate(
            inputs=request["prompt"],
            params=VisualGenParams(height=request["height"], width=request["width"], num_frames=request["num_frames"],
                                   num_inference_steps=args.steps, seed=args.seed),
        )
        elapsed = time.time() - t0
        video = torch.as_tensor(output.video).cpu()
        if video.ndim == 4:
            video = video.unsqueeze(0)
        audio = None if getattr(output, "audio", None) is None else torch.as_tensor(output.audio).cpu()
        torch.save({"video": video, "audio": audio}, args.output / "output.pt")
        (args.output / "result.json").write_text(json.dumps({"status": "ok", "generate_seconds": elapsed,
                                                             "video_shape": list(video.shape), "video_dtype": str(video.dtype),
                                                             "pipeline_warmup": args.pipeline_warmup, "warmup_steps_moved": warmup_steps}))
        print("H3_QUALITY_GEN_DONE", elapsed, flush=True)
    finally:
        engine.shutdown()


def _installed(name):
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
