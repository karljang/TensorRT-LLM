"""Per-process patches for the §628 study; lives on PYTHONPATH so VisualGen's spawned
worker processes pick them up too. Each patch is applied lazily when its module is
imported, so interpreter start-up is unchanged.

H3_PROBE_DIR: record every MiniMaxH3Scheduler.step call (model velocity, sample, next sample).
H3_UNGATE_CUDNN=1: carry TensorRT-LLM PR #19738 (head 6cbec8f), the cuDNN quantized-attention
hardware gate accepting SM 100, 103 and 107 on the tensors' device (same patch as h3_ungate.py).
H3_TRTLLMGEN_FP8=1: carry PR #20005 + #20836 in the small: the visual_gen TRTLLM backend routes the
per-tensor FP8 recipe (q=fp8, v=fp8, block sizes 0) to trtllm::fmha_dense_fp8 (h3_trtllmgen.py,
extension at H3_TRTLLMGEN_SO) instead of requiring PrimTS, and the MiniMax-H3 pipeline's TRTLLM
gate accepts SM 107 next to SM 100 and 103.
H3_UNGATE_TRTLLM=1: only that pipeline gate lift (PR #21278), for the BF16 TRTLLM backend route of lever 21
(Sol-Attn through the predictor op + PrimsTS block-sparse FMHA); sets pipeline module attribute
H3_UNGATE_TRTLLM_SHIM, which h3_e2e_bench.py records.
"""

import importlib.abc
import importlib.util
import os
import sys

_PROBE_DIR = os.environ.get("H3_PROBE_DIR")
_MODULE_PREFIX = "diffusers.schedulers.scheduling_minimax_h3"
_UNGATE = os.environ.get("H3_UNGATE_CUDNN") == "1"
_CUDNN_MODULES = (
    "tensorrt_llm._torch.visual_gen.attention.backends.cudnn",
    "tensorrt_llm._torch.visual_gen.attention_backend.cudnn",
)
_ALLOWED_SM = (100, 103, 107)
_TRTLLMGEN = os.environ.get("H3_TRTLLMGEN_FP8") == "1"
_VG_TRTLLM_MODULE = "tensorrt_llm._torch.visual_gen.attention.backends.trtllm"
_H3_PIPELINE_MODULE = "tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3"
_STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _per_tensor_fp8(cfg):
    return (cfg is not None and cfg.qk_dtype == "fp8" and cfg.v_dtype == "fp8"
            and (cfg.q_block_size, cfg.k_block_size, cfg.v_block_size) == (0, 0, 0))


def _route_trtllmgen(module):
    """visual_gen TrtllmAttention: per-tensor FP8 through trtllm::fmha_dense_fp8 (PR #20005/#20836
    wiring reduced to the H3 case: dense self-attention, uniform sequence lengths, BF16 output)."""
    import math

    if _STUDY_ROOT not in sys.path:
        sys.path.insert(0, _STUDY_ROOT)
    import h3_trtllmgen

    cls = module.TrtllmAttention
    original_init = cls.__init__
    original_forward = cls.forward

    def __init__(self, *args, quant_attention_config=None, **kwargs):
        if not _per_tensor_fp8(quant_attention_config):
            original_init(self, *args, quant_attention_config=quant_attention_config, **kwargs)
            return
        # Construct unquantized (skips the PrimTS requirement), then attach the recipe.
        original_init(self, *args, quant_attention_config=None, **kwargs)
        self.quant_attention_config = quant_attention_config
        if not h3_trtllmgen.is_supported(self.head_dim, self.num_heads):
            raise RuntimeError("trtllm::fmha_dense_fp8: no trtllm-gen dense FP8 kernel for head_dim="
                               f"{self.head_dim}, num_heads={self.num_heads} on this device")
        self._h3_trtllmgen = True
        print(f"[sitecustomize pid{os.getpid()}] TrtllmAttention layer {self.layer_idx}: per-tensor FP8 -> "
              f"trtllm::fmha_dense_fp8 ({h3_trtllmgen.load()})", flush=True)

    def forward(self, q, k, v, batch_size, seq_len, *args, seq_len_kv=None, **kwargs):
        if not getattr(self, "_h3_trtllmgen", False):
            return original_forward(self, q, k, v, batch_size, seq_len, *args, seq_len_kv=seq_len_kv, **kwargs)
        if k is None or v is None:
            raise ValueError("trtllm::fmha_dense_fp8 needs separate q, k and v")
        if seq_len_kv is not None and seq_len_kv != seq_len:
            raise ValueError("trtllm::fmha_dense_fp8 serves self-attention only")
        sm_scale = 1.0 / (math.sqrt(self.head_dim) * getattr(self, "q_scaling", 1.0))
        shape = (batch_size, seq_len, self.num_heads, self.head_dim)
        out = h3_trtllmgen.attention(q.reshape(shape), k.reshape(shape), v.reshape(shape), sm_scale)
        return out.view(batch_size, seq_len, -1)

    cls.__init__ = __init__
    cls.forward = forward
    print(f"[sitecustomize pid{os.getpid()}] trtllmgen route (#20005/#20836) applied to {module.__name__}", flush=True)


def _allow_trtllm_fp8_config(module):
    """visual_gen/args.py AttentionConfig: the TRTLLM recipe table is keyed by SM (PrimTS on 100 / 103
    only) through the module-global get_sm_version, its only use in the file; report 100 on SM 107 so
    the per-tensor FP8 recipe reaches the backend, where the route above serves it."""
    real = module.get_sm_version

    def get_sm_version():
        sm = real()
        return 100 if sm == 107 else sm

    module.get_sm_version = get_sm_version
    print(f"[sitecustomize pid{os.getpid()}] TRTLLM quant recipe table: SM 107 reads as 100 in {module.__name__}", flush=True)


def _lift_h3_trtllm_gate(module):
    """MiniMaxH3Pipeline.__init__: the TRTLLM backend gate lists SM 107 next to 100 and 103."""
    import inspect
    import textwrap

    cls = module.MiniMaxH3Pipeline
    source = textwrap.dedent(inspect.getsource(cls.__init__))
    needle = "((10, 0), (10, 3))"
    if needle not in source:
        print(f"[sitecustomize pid{os.getpid()}] H3 TRTLLM gate not found in {module.__name__}; left as is", flush=True)
        return
    source = source.replace(needle, "((10, 0), (10, 3), (10, 7))").replace("super()", f"super({cls.__name__}, self)")
    namespace = dict(module.__dict__)
    exec(compile(source, module.__file__, "exec"), namespace)  # noqa: S102 - local carry of a one-tuple change
    cls.__init__ = namespace["__init__"]
    module.H3_UNGATE_TRTLLM_SHIM = "pipeline gate ((10, 0), (10, 3)) -> + (10, 7); PR #21278 578eafd893 carried as a study shim"
    print(f"[sitecustomize pid{os.getpid()}] H3 TRTLLM SM gate lifted to (10, 7) in {module.__name__}", flush=True)


def _ungate(module):
    import torch

    def check_hardware_compatibility(cls, device, quant_dtype=None):
        if quant_dtype is None:
            return
        major, minor = torch.cuda.get_device_capability(device)
        if major * 10 + minor not in _ALLOWED_SM:
            raise RuntimeError(
                "cuDNN quantized attention requires an NVIDIA Blackwell- or Rubin-class GPU (SM 100, 103 or 107)."
            )

    module.CuDNNAttention.check_hardware_compatibility = classmethod(check_hardware_compatibility)
    print(f"[sitecustomize pid{os.getpid()}] cudnn_ungate #19738 applied to {module.__name__}", flush=True)


def _patch(module):
    import torch

    cls = module.MiniMaxH3Scheduler
    original = cls.step
    state = {"schedulers": {}}

    def step(self, model_output, timestep, sample, return_dict=True):
        out = original(self, model_output, timestep, sample, return_dict)
        prev = out.prev_sample if hasattr(out, "prev_sample") else out[0]
        sched_ids = state["schedulers"]
        key = sched_ids.setdefault(id(self), len(sched_ids))
        count = state.setdefault(("count", key), 0)
        state[("count", key)] = count + 1
        ts = timestep
        if isinstance(ts, torch.Tensor):
            ts = ts.detach().float().cpu().reshape(-1)[0].item()
        os.makedirs(_PROBE_DIR, exist_ok=True)
        path = os.path.join(_PROBE_DIR, f"pid{os.getpid()}_s{key}_step{count:02d}.pt")
        torch.save(
            {
                "timestep": float(ts),
                "step_index": self.step_index,
                "model_output": model_output.detach().cpu(),
                "sample": sample.detach().cpu(),
                "prev_sample": prev.detach().cpu(),
            },
            path,
        )
        return out

    cls.step = step


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


if _PROBE_DIR:
    sys.meta_path.insert(0, _Finder(lambda n: n.startswith(_MODULE_PREFIX), "MiniMaxH3Scheduler", _patch))
if _UNGATE:
    sys.meta_path.insert(0, _Finder(lambda n: n in _CUDNN_MODULES, "CuDNNAttention", _ungate))
if _TRTLLMGEN:
    sys.meta_path.insert(0, _Finder(lambda n: n == _VG_TRTLLM_MODULE, "TrtllmAttention", _route_trtllmgen))
    sys.meta_path.insert(0, _Finder(lambda n: n == _H3_PIPELINE_MODULE, "MiniMaxH3Pipeline", _lift_h3_trtllm_gate))
    sys.meta_path.insert(0, _Finder(lambda n: n == "tensorrt_llm.visual_gen.args", "AttentionConfig", _allow_trtllm_fp8_config))
# H3_UNGATE_TRTLLM=1 (lever 21, Sol-Attn through the BF16 TRTLLM backend on SM 107): only the pipeline's
# TRTLLM SM gate is lifted (PR #21278); the FP8 recipe routing of H3_TRTLLMGEN_FP8 stays off.
if os.environ.get("H3_UNGATE_TRTLLM") == "1" and not _TRTLLMGEN:
    sys.meta_path.insert(0, _Finder(lambda n: n == _H3_PIPELINE_MODULE, "MiniMaxH3Pipeline", _lift_h3_trtllm_gate))
