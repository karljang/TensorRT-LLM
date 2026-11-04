"""Lever-22 study shim (§657 box 52): the base study shim plus three overrides for Sol-Attn +
SageAttention through the TRTLLM backend on SM 107. Lives on PYTHONPATH in front of the base
probe directory so VisualGen's spawned worker processes pick it up too.

First the base shim (`H3_BASE_SHIM`, default `<study root>/hecate/probe/sitecustomize.py`:
H3_PROBE_DIR step recording, H3_UNGATE_CUDNN #19738 carry, H3_TRTLLMGEN_FP8) is executed as
`sitecustomize`, so everything it provides keeps working. Then, lazily on module import:

H3_UNGATE_TRTLLM=1  MiniMaxH3Pipeline.__init__: the TRTLLM backend gate lists SM 107 next to
                    SM 100 and 103 (the base shim's `_lift_h3_trtllm_gate`, on its own switch;
                    lever 21's name, ¶15559).
H3_UNGATE_SAGE=1    tensorrt_llm.visual_gen.args: the TRTLLM quant recipe table is keyed by SM
                    and has no SM 107 row, so SM 107 reads as 100 there (the base shim's
                    `_allow_trtllm_fp8_config`); the Sage recipe then reaches the backend, where
                    the trtllm-gen Sm107a Sage cubins (dense calls) and PrimsTS block-sparse
                    (sparse calls, `sage_quantize`) serve it.
H3_SOL_STATS=<json> SOLTrtllmAttention.block_sparse_attn_predict: per sparse call, the fraction
                    of exact (kept) KV blocks from `exact_block_bits` (`[B, Hkv, nQ, ceil(nK/32)]`
                    uint32, bit r of word w = block 32w+r), the layer index and the normalized
                    timestep, accumulated on the device and written at exit (per call and per
                    timestep, mean over layers). Dense calls (dense prefix) are counted too, as
                    kept fraction 1. Use with --compile off: the wrapper is Python side effects.
"""

import atexit
import importlib.abc
import importlib.util
import json
import os
import sys

_STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BASE_SHIM = os.environ.get("H3_BASE_SHIM") or os.path.join(_STUDY_ROOT, "hecate", "probe", "sitecustomize.py")
_H3_PIPELINE_MODULE = "tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3"
_VG_ARGS_MODULE = "tensorrt_llm.visual_gen.args"
_SOL_BACKEND_MODULE = "tensorrt_llm._torch.visual_gen.attention.backends.sparse.sol.backend"
_UNGATE_TRTLLM = os.environ.get("H3_UNGATE_TRTLLM") == "1"
_UNGATE_SAGE = os.environ.get("H3_UNGATE_SAGE") == "1"
_SOL_STATS = os.environ.get("H3_SOL_STATS")


def _load_base():
    """Execute the base study shim under the `sitecustomize` name; return its namespace."""
    if not os.path.isfile(_BASE_SHIM):
        print(f"[probe22 pid{os.getpid()}] base shim not found at {_BASE_SHIM}; lever-22 overrides only", flush=True)
        return None
    spec = importlib.util.spec_from_file_location("sitecustomize_base", _BASE_SHIM)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


_base = _load_base()


class _Finder(importlib.abc.MetaPathFinder):
    """Patch one module after its normal loader has executed it, then retire."""

    def __init__(self, matches, attr, patch):
        self._matches, self._attr, self._patch = matches, attr, patch

    def find_spec(self, name, path, target=None):
        if not self._matches(name):
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        exec_module = loader.exec_module
        patch = self._patch
        attr = self._attr

        def patched_exec(module):
            exec_module(module)
            if hasattr(module, attr):
                patch(module)

        loader.exec_module = patched_exec
        return spec


def _lift_h3_trtllm_gate(module):
    if _base is not None and hasattr(_base, "_lift_h3_trtllm_gate"):
        _base._lift_h3_trtllm_gate(module)
        return
    import inspect
    import textwrap

    cls = module.MiniMaxH3Pipeline
    source = textwrap.dedent(inspect.getsource(cls.__init__))
    needle = "((10, 0), (10, 3))"
    if needle not in source:
        print(f"[probe22 pid{os.getpid()}] H3 TRTLLM gate not found in {module.__name__}; left as is", flush=True)
        return
    source = source.replace(needle, "((10, 0), (10, 3), (10, 7))").replace("super()", f"super({cls.__name__}, self)")
    namespace = dict(module.__dict__)
    exec(compile(source, module.__file__, "exec"), namespace)  # noqa: S102 - local carry of a one-tuple change
    cls.__init__ = namespace["__init__"]
    print(f"[probe22 pid{os.getpid()}] H3 TRTLLM SM gate lifted to (10, 7) in {module.__name__}", flush=True)


def _allow_sage_recipe(module):
    if _base is not None and hasattr(_base, "_allow_trtllm_fp8_config"):
        _base._allow_trtllm_fp8_config(module)
        return
    real = module.get_sm_version

    def get_sm_version():
        sm = real()
        return 100 if sm == 107 else sm

    module.get_sm_version = get_sm_version
    print(f"[probe22 pid{os.getpid()}] TRTLLM quant recipe table: SM 107 reads as 100 in {module.__name__}", flush=True)


def _sol_stats(module):
    import torch

    cls = module.SOLTrtllmAttention
    original = cls.block_sparse_attn_predict
    calls = []  # (layer, timestep tensor or float, kept tensor, total int, dense bool)
    shifts = torch.arange(32, dtype=torch.int64)

    def kept_blocks(bits, num_kv_blocks):
        words = bits.to(torch.int64)  # uint32 -> int64, no sign issue
        expanded = (words.unsqueeze(-1) >> shifts.to(words.device)) & 1  # [..., words, 32]
        expanded = expanded.flatten(-2)[..., :num_kv_blocks]  # drop final-word padding bits
        return expanded.sum()

    def wrapper(self, q, k, v, metadata, forward_args):
        out = original(self, q, k, v, metadata, forward_args)
        ts = forward_args.timestep
        if isinstance(ts, torch.Tensor):
            ts = ts.detach().reshape(-1)[:1].float()
        if out is None:
            calls.append((self.layer_idx, ts, None, None, True))
            return out
        bits = out.exact_block_bits
        seq_len = metadata.max_seq_len
        num_kv_blocks = -(-seq_len // out.kv_block_size)
        total = bits.shape[0] * bits.shape[1] * bits.shape[2] * num_kv_blocks
        calls.append((self.layer_idx, ts, kept_blocks(bits, num_kv_blocks), total, False))
        return out

    cls.block_sparse_attn_predict = wrapper

    def dump():
        rows = []
        for layer, ts, kept, total, dense in calls:
            t = float(ts.item()) if isinstance(ts, torch.Tensor) else (None if ts is None else float(ts))
            rows.append({"layer": int(layer), "timestep": t, "dense": dense,
                         "kept": None if kept is None else int(kept.item()), "total": total,
                         "kept_fraction": 1.0 if dense else (int(kept.item()) / total if total else None)})
        by_t = {}
        for r in rows:
            by_t.setdefault(r["timestep"], []).append(r["kept_fraction"])
        per_step = [{"timestep": t, "calls": len(v), "kept_fraction_mean": sum(v) / len(v),
                     "dense": all(x == 1.0 for x in v)} for t, v in by_t.items()]
        per_step.sort(key=lambda d: -(d["timestep"] if d["timestep"] is not None else -1))
        sparse_steps = [d for d in per_step if not d["dense"]]
        summary = {
            "pid": os.getpid(), "calls": len(rows), "steps": len(per_step),
            "dense_steps": sum(d["dense"] for d in per_step), "sparse_steps": len(sparse_steps),
            "kept_fraction_sparse_mean": (sum(d["kept_fraction_mean"] for d in sparse_steps) / len(sparse_steps)) if sparse_steps else None,
            "timesteps": [d["timestep"] for d in per_step],
            "per_step": per_step, "per_call": rows,
        }
        path = _SOL_STATS if _SOL_STATS.endswith(".json") else os.path.join(_SOL_STATS, f"sol-stats-pid{os.getpid()}.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=1)
        print(f"[probe22 pid{os.getpid()}] sol stats: {len(rows)} calls, {summary['dense_steps']} dense / {summary['sparse_steps']} sparse steps, "
              f"kept fraction (sparse) {summary['kept_fraction_sparse_mean']} -> {path}", flush=True)

    atexit.register(dump)
    print(f"[probe22 pid{os.getpid()}] sol stats hook on {module.__name__}.SOLTrtllmAttention", flush=True)


if _UNGATE_TRTLLM:
    sys.meta_path.insert(0, _Finder(lambda n: n == _H3_PIPELINE_MODULE, "MiniMaxH3Pipeline", _lift_h3_trtllm_gate))
if _UNGATE_SAGE:
    sys.meta_path.insert(0, _Finder(lambda n: n == _VG_ARGS_MODULE, "AttentionConfig", _allow_sage_recipe))
if _SOL_STATS:
    sys.meta_path.insert(0, _Finder(lambda n: n == _SOL_BACKEND_MODULE, "SOLTrtllmAttention", _sol_stats))
