"""End-to-end MiniMax-H3 timing harness for TRTLLM-17216 (board §641 track A).

One process = one configuration (checkpoint, attention backend, compile, CUDA graphs) on one
GPU. It loads the shipping pipeline through ``PipelineLoader``, runs ``--warmup`` generations
(excluded from every number), then ``--timed`` generations of the request and records, per
generation:

* whole-generation wall time and the pipeline's own CUDA-event phases (pre-denoise, denoise,
  post-denoise, ``CudaPhaseTimer``);
* a component split from CUDA events placed around the pipeline methods: text encoder
  (``_encode_prompt``), keyframe VAE encode (``_encode_keyframes``, FL2VA only), static
  context (``prepare_static_context``: text refiner + RoPE), every transformer call
  (``transformer.forward``, one per denoising step, graph replay included when enabled),
  video VAE decode and audio VAE decode;
* per-step transformer time (ms) as a list, so the steady state and the first-step outlier
  are both visible;
* NVML SM clock / power / throttle samples at ~10 Hz while the generation runs;
* provenance: TRT-LLM commit and file, container tag, torch / CUDA / cuDNN / flash_attn_4 /
  nvidia-cutlass-dsl versions, the attention artifact that loaded (``flash_attn.cute.interface``
  path), the effective FA4 2-CTA switch, the resolved attention backend class, the quant recipe
  of the loaded transformer, ``sys.path`` and the GPU state.

With ``--profile`` the first timed generation is bracketed by ``torch.cuda.profiler.start()``
and ``stop()`` so ``nsys --capture-range=cudaProfilerApi`` records only that generation; NVTX
ranges name the components (``h3/prompt_encode``, ``h3/transformer``, ``h3/block{i}``,
``h3/video_decode``, ``h3/audio_decode``).

Request JSON: ``{"prompt", "height", "width", "num_frames", "num_inference_steps", "seed"}``;
FL2VA adds ``"keyframes": [{"path": "...", "anchor": "first"|"last"}, ...]`` with paths
relative to the request file. Outputs go to ``--output``: ``results.json`` (everything above),
``steps.csv`` (per-step ms per generation) and ``output.pt`` (uint8 video + audio of the last
timed generation, for a later LPIPS comparison with ``quality_compare.py``).
"""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch


class NvmlSampler:
    """Polls nvidia-smi for SM clock, power and throttle reasons while enabled (~10 Hz)."""

    def __init__(self, device_index, period=0.1):
        self.index = device_index
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        t0 = time.perf_counter()
        query = "clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "-i", str(self.index), f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            try:
                sm, pw, temp, thr = [x.strip() for x in out.split(",")]
                self.samples.append(
                    {"t": time.perf_counter() - t0, "sm_mhz": float(sm), "power_w": float(pw), "temp_c": float(temp), "throttle": thr}
                )
            except ValueError:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        return self.samples


def summarize_samples(samples):
    if not samples:
        return None
    sm = sorted(s["sm_mhz"] for s in samples)
    pw = sorted(s["power_w"] for s in samples)
    return {
        "n": len(samples),
        "sm_mhz_median": sm[len(sm) // 2],
        "sm_mhz_min": sm[0],
        "sm_mhz_max": sm[-1],
        "power_w_median": pw[len(pw) // 2],
        "power_w_max": pw[-1],
        "throttle_reasons": sorted({s["throttle"] for s in samples}),
    }


class EventTimer:
    """CUDA-event bracket around named calls; ``collect`` returns per-call ms after a sync."""

    def __init__(self):
        self.records = []  # (name, start_event, end_event, host_seconds)

    def wrap(self, target, method, name, nvtx=None):
        original = getattr(target, method)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if nvtx:
                torch.cuda.nvtx.range_push(nvtx)
            start.record()
            t = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                end.record()
                if nvtx:
                    torch.cuda.nvtx.range_pop()
                self.records.append((name, start, end, time.perf_counter() - t))

        setattr(target, method, wrapped)

    def collect(self):
        torch.cuda.synchronize()
        out = {}
        for name, start, end, host in self.records:
            out.setdefault(name, []).append({"ms": start.elapsed_time(end), "host_ms": host * 1e3})
        self.records = []
        return out


def _installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_sha(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def provenance(args, device_index):
    import tensorrt_llm

    tllm_dir = Path(tensorrt_llm.__file__).resolve().parent
    prov = {
        "host": platform.node(),
        "arch": platform.machine(),
        "container": args.container or os.environ.get("H3_CONTAINER") or os.environ.get("NVIDIA_BUILD_ID"),
        "tensorrt_llm": tensorrt_llm.__version__,
        "tensorrt_llm_file": str(tllm_dir),
        "tensorrt_llm_commit": _git_sha(tllm_dir.parent),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.version.split()[0],
        "versions": {
            p: _installed(p)
            for p in ("flash_attn_4", "nvidia-cutlass-dsl", "diffusers", "transformers", "triton", "flashinfer-python", "nvidia-cudnn-frontend", "nvidia-modelopt")
        },
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "device_properties": {k: getattr(torch.cuda.get_device_properties(0), k) for k in ("multi_processor_count", "total_memory")},
        "gpu_state": subprocess.run(
            ["nvidia-smi", "-i", str(device_index), "--query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm,clocks.max.mem", "--format=csv"],
            capture_output=True,
            text=True,
        ).stdout,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env": {k: os.environ.get(k) for k in ("FA_DISABLE_2CTA", "TLLM_FMHA_LIBS", "TORCHINDUCTOR_CACHE_DIR", "PYTHONPATH")},
        "sys_path": sys.path,
    }
    try:
        from flash_attn.cute import interface as fa_interface
        from flash_attn.cute import utils as fa_utils

        prov["attention_artifact"] = {
            "flash_attn_cute_interface": fa_interface.__file__,
            "sha256": hashlib.sha256(Path(fa_interface.__file__).read_bytes()).hexdigest()[:16],
            "fa4_disable_2cta_effective_fwd": bool(fa_utils._get_disable_2cta_default(is_fwd=True)),
            "fa4_disable_2cta_env_flag": bool(getattr(fa_utils, "_fa_disable_2cta_enabled", None)),
            "fa4_disable_2cta_cuda12": bool(getattr(fa_utils, "_fa_disable_2cta_cuda12", None)),
        }
    except Exception as exc:  # noqa: BLE001 - provenance only
        prov["attention_artifact"] = {"error": repr(exc)}
    return prov


def load_keyframes(request, request_path):
    from PIL import Image

    keyframes, anchors = [], []
    for entry in request.get("keyframes") or []:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = request_path.parent / path
        keyframes.append(Image.open(path).convert("RGB"))
        anchors.append(entry["anchor"])
    return keyframes, tuple(anchors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint directory (BF16, or a ModelOpt NVFP4 / FP8 export: the recipe comes from its quantization_config)")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default=None, help="free-form name of this arm (defaults to the output directory name)")
    parser.add_argument("--attention-backend", default=None, help="VisualGenArgs attention backend; default = pipeline default")
    parser.add_argument("--qk-dtype", default=None, help="quantized attention recipe (cuDNN FP8: fp8 / fp8); SM 107 needs H3_UNGATE_CUDNN=1 with the probe shim on PYTHONPATH")
    parser.add_argument("--v-dtype", default=None)
    parser.add_argument("--quant-blocks", default=None, help="SageAttention block sizes Q,K,V of the quant recipe (TRTLLM backend), e.g. 1,16,1; unset = per-tensor (0,0,0)")
    parser.add_argument("--sol", default=None, help="Sol-Attn: TAU,CUTOFF[,THRESH] -> sparse_attention_config {algorithm: sol_attn, tau, disabled_until_timestep: CUTOFF (steps with normalized timestep >= CUTOFF run dense; 'none' = no dense prefix), thresh_type: THRESH (diag)}; SM 107 needs H3_UNGATE_TRTLLM=1 with the probe shim")
    parser.add_argument("--quant-algo", default=None, help="dynamic quantization of a BF16 checkpoint (FP8, FP8_BLOCK_SCALES, NVFP4); leave unset for BF16 or a pre-quantized export")
    parser.add_argument("--compile", choices=["on", "off"], default="on")
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="off")
    parser.add_argument("--ulysses", type=int, default=1, help="ulysses_size (single process launch only; >1 needs the VisualGen multi-worker launch)")
    parser.add_argument(
        "--pipeline-warmup",
        choices=["on", "off"],
        default="on",
        help="run the pipeline's own warmup at the request canvas (the shipping path: torch.compile, AutoTuner tactics and CUDA-graph capture happen there); off = skip it and rely on --warmup generations only, which leaves the AutoTuner at its fallback tactics",
    )
    parser.add_argument("--warmup", type=int, default=1, help="warm-up generations, excluded from every number")
    parser.add_argument("--timed", type=int, default=3)
    parser.add_argument(
        "--interleave-graphs",
        type=int,
        default=0,
        metavar="ROUNDS",
        help="with --cuda-graph on: ROUNDS pairs of timed generations alternating CUDA-graph replay and eager "
        "(the runner is disabled for the eager ones, same pipeline, same compiled blocks, same padded layout); "
        "the order flips every round; replaces --timed",
    )
    parser.add_argument("--profile", action="store_true", help="bracket the first timed generation with cudaProfilerApi for nsys")
    parser.add_argument("--container", default=None, help="container tag for provenance")
    parser.add_argument("--save-output", choices=["on", "off"], default="on")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    request = json.loads(args.request.read_text())
    device_index = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])

    import tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 as h3_pipeline
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import align_num_frames
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import MiniMaxH3TransformerBlock
    from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
    from tensorrt_llm.visual_gen.args import VisualGenArgs

    request["num_frames"] = align_num_frames(request["num_frames"])
    results = {
        "label": args.label or args.output.name,
        "request": request,
        "request_file": str(args.request.resolve()),
        "model": str(Path(args.model).resolve()),
        "config": {
            "attention_backend": args.attention_backend,
            "quant_algo": args.quant_algo,
            "compile": args.compile,
            "cuda_graph": args.cuda_graph,
            "ulysses": args.ulysses,
            "warmup": args.warmup,
            "timed": args.timed,
            "profile": args.profile,
        },
        "provenance": provenance(args, device_index),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    def write():
        (args.output / "results.json").write_text(json.dumps(results, indent=2, default=str))

    write()

    layouts = []
    original_build = h3_pipeline.build_packed_sequence

    def logged_build(*a, **k):
        layout = original_build(*a, **k)
        layouts.append(
            {
                "sequence_length": layout.sequence_length,
                "num_condition_video_rows": layout.num_condition_video_rows,
                "num_condition_audio_rows": layout.num_condition_audio_rows,
                "video_tokens": int(layout.video_indices.numel()),
                "audio_tokens": int(layout.audio_indices.numel()),
                "text_tokens": int(layout.text_indices.numel()),
            }
        )
        return layout

    h3_pipeline.build_packed_sequence = logged_build

    vg = {
        "model": args.model,
        "torch_compile_config": {"enable": args.compile == "on"},
        "cuda_graph_config": {"enable": args.cuda_graph == "on"},
        "compilation_config": {
            "skip_warmup": args.pipeline_warmup == "off",
            "resolutions": [[request["height"], request["width"]]],
            "num_frames": [request["num_frames"]],
        },
    }
    results["config"]["pipeline_warmup"] = args.pipeline_warmup
    if args.attention_backend:
        vg["attention_config"] = {"backend": args.attention_backend}
    if args.qk_dtype or args.v_dtype:
        quant = {"qk_dtype": args.qk_dtype, "v_dtype": args.v_dtype}
        if args.quant_blocks:
            qb, kb, vb = (int(x) for x in args.quant_blocks.split(","))
            quant.update({"q_block_size": qb, "k_block_size": kb, "v_block_size": vb})
        vg.setdefault("attention_config", {})["quant_attention_config"] = quant
        results["config"]["quant_attention_config"] = vg["attention_config"]["quant_attention_config"]
        results["config"]["cudnn_ungate"] = "#19738 6cbec8f" if os.environ.get("H3_UNGATE_CUDNN") == "1" else None
    if args.sol:
        parts = args.sol.split(",")
        sparse = {"algorithm": "sol_attn", "tau": float(parts[0]), "thresh_type": parts[2] if len(parts) > 2 else "diag"}
        if len(parts) > 1 and parts[1].lower() != "none":
            sparse["disabled_until_timestep"] = float(parts[1])
        vg.setdefault("attention_config", {})["sparse_attention_config"] = sparse
        results["config"]["sparse_attention_config"] = sparse
    results["config"]["study_overrides"] = {k: os.environ.get(k) for k in ("H3_UNGATE_CUDNN", "H3_UNGATE_TRTLLM", "H3_UNGATE_SAGE", "TLLM_FMHA_LIBS", "CUTE_DSL_ARCH", "H3_SOL_STATS")}
    if args.quant_algo:
        vg["quant_config"] = {"quant_algo": args.quant_algo, "dynamic": True}
    if args.ulysses > 1:
        vg["parallel_config"] = {"ulysses_size": args.ulysses, "parallel_vae_size": args.ulysses}
    # Keep the AutoTuner's choices (NVFP4 GEMM tactics among them) next to the timings: the
    # warmup tunes by default, and a block fixture that did not tune measures the fallback tactic.
    autotuner_cache = os.environ.setdefault("TLLM_AUTOTUNER_CACHE_PATH", str(args.output / "autotuner_cache.json"))
    results["autotune"] = {"cache_path": autotuner_cache}
    t0 = time.perf_counter()
    pipeline = PipelineLoader(VisualGenArgs(**vg)).load(skip_warmup=args.pipeline_warmup == "off")
    torch.cuda.synchronize()
    results["model_load_seconds"] = time.perf_counter() - t0  # includes the pipeline warmup when on
    transformer = pipeline.transformer
    attn_module = next(m for m in transformer.modules() if type(m).__name__ == "MiniMaxH3Attention")
    inner_backend = getattr(attn_module.attn, "inner", attn_module.attn)
    try:
        from tensorrt_llm._torch.visual_gen.attention.backends import flash_attn4 as fa4_backend

        fa4_env_explicit = bool(getattr(fa4_backend, "_FA_DISABLE_2CTA_FROM_ENV", False))
    except ImportError:
        fa4_env_explicit = None
    results["resolved"] = {
        "attention_backend": type(attn_module.attn).__name__,
        "attention_inner": type(inner_backend).__name__,
        # The H3 request for FA4's 2-CTA kernel (None = FA4's default) and whether an explicit
        # FA_DISABLE_2CTA in the environment overrode it; the env value alone cannot tell, since
        # the backend sets it at import.
        "fa4_use_2cta": getattr(inner_backend, "use_2cta", None),
        "fa4_env_explicit": fa4_env_explicit,
        "fa4_use_2cta_default": getattr(transformer, "fa4_use_2cta", None),
        "quant_algo": str(getattr(transformer.model_config.quant_config, "quant_algo", None)),
        "dynamic_weight_quant": bool(getattr(transformer.model_config, "dynamic_weight_quant", False)),
        "exclude_modules": list(getattr(transformer.model_config.quant_config, "exclude_modules", None) or []),
        "cuda_graph_runners": sorted(getattr(pipeline, "_cuda_graph_runners", {}).keys()),
        "transformer_forward": repr(transformer.forward),
        "num_blocks": len(transformer.transformer_blocks),
        "block_type": type(transformer.transformer_blocks[0]).__name__,
        # Sol-Attn / SageAttention resolution (TRTLLM backend): the route class, its SolParams and the
        # quant recipe the backend holds, plus the FMHA libraries the core registered.
        "sol_params": repr(getattr(inner_backend, "sol_params", None)),
        "timestep_cutoff": getattr(inner_backend, "timestep_cutoff", None),
        "quant_attention_config": repr(getattr(inner_backend, "quant_attention_config", None)),
        "fmha_libs": [type(f).__name__ for f in getattr(getattr(inner_backend, "_fmha_manager", None), "fmha_libs", [])],
    }
    write()

    timer = EventTimer()
    timer.wrap(pipeline, "_encode_prompt", "text_encode", nvtx="h3/prompt_encode")
    timer.wrap(pipeline, "_encode_keyframes", "keyframe_encode", nvtx="h3/keyframe_encode")
    timer.wrap(transformer, "prepare_static_context", "static_context", nvtx="h3/static_context")
    # Wrapping the instance attribute keeps a CUDA-graph runner's wrap underneath.
    timer.wrap(transformer, "forward", "transformer_step", nvtx="h3/transformer")
    timer.wrap(pipeline, "_decode_video", "video_decode", nvtx="h3/video_decode")
    timer.wrap(pipeline, "_decode_audio", "audio_decode", nvtx="h3/audio_decode")
    blocks = [m for m in transformer.modules() if isinstance(m, MiniMaxH3TransformerBlock)]

    def push(name):
        def hook(module, a, k):
            torch.cuda.nvtx.range_push(name)

        return hook

    def pop(module, a, k, output):
        torch.cuda.nvtx.range_pop()

    for i, block in enumerate(blocks):
        block.register_forward_pre_hook(push(f"h3/block{i}"), with_kwargs=True)
        block.register_forward_hook(pop, with_kwargs=True)

    keyframes, anchors = load_keyframes(request, args.request)
    call = dict(
        prompt=request["prompt"],
        seed=int(request.get("seed", 42)),
        height=request["height"],
        width=request["width"],
        num_frames=request["num_frames"],
        frame_rate=float(request.get("frame_rate", 24)),
        num_inference_steps=int(request["num_inference_steps"]),
    )
    if keyframes:
        call["keyframes"] = keyframes
        call["keyframe_anchors"] = anchors
    results["workflow"] = "fl2va" if keyframes else "t2va"

    sampler = NvmlSampler(device_index)
    gens = []
    steps_csv = ["kind,generation,step,ms"]
    last_output = None
    runners = list(getattr(pipeline, "_cuda_graph_runners", {}).values())
    if args.interleave_graphs and not runners:
        raise SystemExit("--interleave-graphs needs --cuda-graph on (no CUDA-graph runner installed)")

    def set_graph_mode(mode):
        # "graph": replay the captured graphs; "eager": the same compiled model without the runner.
        for runner in runners:
            runner.enabled = mode == "graph"

    plan = [("warmup", i, None) for i in range(args.warmup)]
    if args.interleave_graphs:
        plan.append(("warmup", args.warmup, "eager"))  # one eager warm-up next to the graph one
        for rnd in range(args.interleave_graphs):
            order = ("eager", "graph") if rnd % 2 == 0 else ("graph", "eager")
            plan.extend(("timed", 2 * rnd + k, mode) for k, mode in enumerate(order))
    else:
        plan.extend(("timed", i, None) for i in range(args.timed))
    for kind, i, mode in plan:
        if True:  # keep the body's indentation stable
            if mode is not None:
                set_graph_mode(mode)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timer.records = []
            sampler.start()
            profiled = args.profile and kind == "timed" and i == 0
            if profiled:
                torch.cuda.profiler.start()
            t = time.perf_counter()
            with torch.cuda.nvtx.range(f"h3/generation/{kind}{i}"):
                out = pipeline.forward(**call)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            if profiled:
                torch.cuda.profiler.stop()
            samples = sampler.stop()
            components = timer.collect()
            step_ms = [r["ms"] for r in components.get("transformer_step", [])]
            for s, ms in enumerate(step_ms):
                steps_csv.append(f"{kind},{i},{s},{ms:.3f}")
            comp_ms = {name: sum(r["ms"] for r in recs) for name, recs in components.items()}
            accounted = sum(comp_ms.values())
            gen = {
                "kind": kind,
                "index": i,
                "graph_mode": mode,
                "wall_s": wall,
                "pipeline_phases_s": {"pre_denoise": out.pre_denoise, "denoise": out.denoise, "post_denoise": out.post_denoise},
                "components_ms": comp_ms,
                "unaccounted_ms": wall * 1e3 - accounted,
                "transformer_steps": len(step_ms),
                "step_ms": step_ms,
                "step_ms_median": sorted(step_ms)[len(step_ms) // 2] if step_ms else None,
                "step_ms_mean_excl_first": (sum(step_ms[1:]) / len(step_ms[1:])) if len(step_ms) > 1 else None,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
                "nvml": summarize_samples(samples),
                "nvml_samples": samples,
                "profiled": profiled,
                "video_shape": list(out.video.shape) if getattr(out, "video", None) is not None else None,
            }
            gens.append(gen)
            print(json.dumps({k: v for k, v in gen.items() if k not in ("nvml_samples", "step_ms")}), flush=True)
            if kind == "timed":
                last_output = out
            else:
                del out
            results["layouts"] = layouts
            results["generations"] = gens
            write()

    timed = [g for g in gens if g["kind"] == "timed"]
    if timed:
        walls = sorted(g["wall_s"] for g in timed)
        steps = sorted(ms for g in timed for ms in g["step_ms"][1:])
        results["summary"] = {
            "timed_generations": len(timed),
            "wall_s_median": walls[len(walls) // 2],
            "wall_s_min": walls[0],
            "step_ms_median_excl_first": steps[len(steps) // 2] if steps else None,
            "components_ms_median": {
                name: sorted(g["components_ms"].get(name, 0.0) for g in timed)[len(timed) // 2]
                for name in sorted({n for g in timed for n in g["components_ms"]})
            },
        }
    if args.interleave_graphs and timed:
        by_mode = {}
        for g in timed:
            by_mode.setdefault(g["graph_mode"], []).append(g)

        def med(values):
            values = sorted(values)
            return values[len(values) // 2]

        results["interleave"] = {
            mode: {
                "n": len(gs),
                "order": [g["index"] for g in gs],
                "wall_s_median": med([g["wall_s"] for g in gs]),
                "wall_s_min": min(g["wall_s"] for g in gs),
                "wall_s_all": [g["wall_s"] for g in gs],
                "step_ms_median_excl_first": med([ms for g in gs for ms in g["step_ms"][1:]]),
                "step_ms_min": min(ms for g in gs for ms in g["step_ms"][1:]),
                "sm_mhz_median": med([g["nvml"]["sm_mhz_median"] for g in gs if g["nvml"]]) if any(g["nvml"] for g in gs) else None,
                "power_w_median": med([g["nvml"]["power_w_median"] for g in gs if g["nvml"]]) if any(g["nvml"] for g in gs) else None,
            }
            for mode, gs in by_mode.items()
        }
        print("| mode | n | gen wall s median / min | step ms median / min | SM MHz | W |")
        print("|---|---|---|---|---|---|")
        for mode, r in results["interleave"].items():
            print(f"| {mode} | {r['n']} | {r['wall_s_median']:.3f} / {r['wall_s_min']:.3f} | {r['step_ms_median_excl_first']:.1f} / {r['step_ms_min']:.1f} | {r['sm_mhz_median']} | {r['power_w_median']} |")
    (args.output / "steps.csv").write_text("\n".join(steps_csv) + "\n")
    tc = getattr(getattr(pipeline, "pipeline_config", None), "torch_compile", None)
    results["autotune"]["enabled"] = bool(getattr(tc, "enable_autotune", False))
    cache_file = Path(autotuner_cache)
    if cache_file.is_file():
        try:
            cache = json.loads(cache_file.read_text())
            # Tuned entries live under "shared" and "rank_<i>"; "metadata" is not one.
            tuned = {}
            for section, entries in cache.items():
                if section != "metadata" and isinstance(entries, dict):
                    tuned.update(entries)
            tactics = {}
            for key, entry in tuned.items():
                tactic = str(entry.get("tactic")) if isinstance(entry, dict) else str(entry)
                tactics.setdefault(tactic, []).append(key[:160])
            results["autotune"]["entries"] = len(tuned)
            results["autotune"]["tactics"] = {t: len(keys) for t, keys in tactics.items()}
            results["autotune"]["tactic_keys"] = tactics
            results["autotune"]["preview"] = json.dumps(cache)[:2000]
        except (OSError, ValueError) as exc:
            results["autotune"]["error"] = repr(exc)
    else:
        results["autotune"]["entries"] = 0
    if last_output is not None and args.save_output == "on":
        video = torch.as_tensor(last_output.video).cpu()
        audio = None if getattr(last_output, "audio", None) is None else torch.as_tensor(last_output.audio).cpu()
        torch.save({"video": video, "audio": audio, "frame_rate": getattr(last_output, "frame_rate", None)}, args.output / "output.pt")
    results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write()
    print("H3_E2E_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
