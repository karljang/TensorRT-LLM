"""Lever-22 study shim (§657 box 52): the study shim of `hecate/probe/sitecustomize.py` (lever 21's
copy: H3_PROBE_DIR step recording, H3_UNGATE_CUDNN #19738 carry, H3_TRTLLMGEN_FP8, H3_UNGATE_TRTLLM
pipeline SM gate) plus one override for SageAttention through the TRTLLM backend on SM 107. Put this
directory in front of the base probe directory on PYTHONPATH; VisualGen's spawned worker processes
pick both up. The base shim is executed first under the `sitecustomize` name (`H3_BASE_SHIM`,
default `<study root>/hecate/probe/sitecustomize.py`), so everything it provides keeps working.

H3_UNGATE_SAGE=1  tensorrt_llm.visual_gen.args: the TRTLLM quant recipe table (SAGE_RECIPES /
                  PRIMSTS_RECIPES in `_validate_quant_attention_config`) is keyed by SM and has no
                  SM 107 row, so every Sage recipe is rejected at config time on Rubin. SM 107 reads
                  as 100 there (the base shim's `_allow_trtllm_fp8_config`, on its own switch); the
                  recipe then reaches the backend, where the trtllm-gen Sm107a Sage cubins serve the
                  dense calls and PrimsTS block-sparse (`sage_quantize` in the attention core) the
                  sparse Sol calls. Sets `tensorrt_llm.visual_gen.args.H3_UNGATE_SAGE_SHIM`.
"""

import importlib.abc
import importlib.util
import os
import sys

_STUDY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BASE_SHIM = os.environ.get("H3_BASE_SHIM") or os.path.join(_STUDY_ROOT, "hecate", "probe", "sitecustomize.py")
_VG_ARGS_MODULE = "tensorrt_llm.visual_gen.args"
_UNGATE_SAGE = os.environ.get("H3_UNGATE_SAGE") == "1"


def _load_base():
    """Execute the base study shim under the `sitecustomize` name; return its namespace."""
    if not os.path.isfile(_BASE_SHIM):
        print(f"[probe22 pid{os.getpid()}] base shim not found at {_BASE_SHIM}; lever-22 override only", flush=True)
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


def _allow_sage_recipe(module):
    if _base is not None and hasattr(_base, "_allow_trtllm_fp8_config"):
        _base._allow_trtllm_fp8_config(module)
    else:
        real = module.get_sm_version

        def get_sm_version():
            sm = real()
            return 100 if sm == 107 else sm

        module.get_sm_version = get_sm_version
        print(f"[probe22 pid{os.getpid()}] TRTLLM quant recipe table: SM 107 reads as 100 in {module.__name__}", flush=True)
    module.H3_UNGATE_SAGE_SHIM = "visual_gen/args.py TRTLLM recipe table: SM 107 reads as 100 (lever 22 study shim)"


if _UNGATE_SAGE:
    sys.meta_path.insert(0, _Finder(lambda n: n == _VG_ARGS_MODULE, "AttentionConfig", _allow_sage_recipe))
