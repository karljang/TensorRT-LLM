#!/usr/bin/env bash
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --partition=batch-xdr
#SBATCH --constraint=cr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=03:55:00
#SBATCH --job-name=coreai_comparch_trtllm-visualgen:sweep-nvfp4-lever22
#SBATCH --output=/lustre/fsw/coreai_comparch_trtllm/kanghwanj/h3/sweep-nvfp4/logs/%x-%j.log
# §657 lever 22 (box 52, brief ¶15558, ®436): Sol-Attn + SageAttention FP8 Q/K through the TRTLLM backend in the pin
# container trtllm-pin-c8ee74b9-sm107.sqsh (#20224 lets SOLTrtllmAttention carry the Sage recipe), U1, against the pin's own
# FA4 2-CTA (p-l12) and cuDNN FP8 (p-f8) arms in-job (pin arms compare only with pin arms, ¶15005 / ¶15006).
# Arms (NVFP4 checkpoint + the final overlay chain when it dry-runs clean on the pin; else BF16 checkpoint, no overlays,
# H3_OVERLAY_MODE records which):
#   p-l12        FA4 2-CTA, BF16 attention (the fallback row)
#   p-f8         cuDNN FP8 attention (H3_UNGATE_CUDNN shim; the recipe's attention)
#   p-tdense     TRTLLM backend, dense (backend-swap control vs p-l12)
#   p-tsage-16   TRTLLM dense + Sage fp8/fp8 (1,16,1) (Sage cost / gain on the dense path, 19k only)
#   p-sol-t2-dB  Sol tau 2, dense prefix B (~10 of 27 steps), BF16 PrimsTS block-sparse (Sage off)
#   p-solsage-16 the same Sol + Sage fp8/fp8 (1,16,1)   p-solsage-4: (1,4,1), 19k only
#   p-sol-t2-dA / p-solsage-16-dA  the 4-dense-step cutoff A pair (second measured job, lead ¶15812)
# Phases (H3_PARTS): probe (DSL version, ops, one eager Sol + Sage generation with lever 21's --sol-probe -> the 27 normalized
# timesteps and the achieved sparsity; derives the cutoffs A (4 dense) / B (10 dense) unless H3_SOL_CUT_B is given), e2e (GPU 0: 19k
# x 2 rotated passes, 104k x 1, 38k x 1 last), nsys (GPU 0, timed generation only, 19k + 104k, p-f8 / p-sol-t2-dB /
# p-solsage-16), side (GPUs 1-3: gate seeds 42 / 42-rerun / 7 / 123 for every arm in H3_GATE_ARMS, ship prompts for
# H3_SHIP_ARMS), cmp (hashes, bitcmp p1 vs p2 per arm, gate seed 42 vs rerun).
# Study overrides, all logged in results.json: H3_UNGATE_TRTLLM=1 (pipeline SM gate), H3_UNGATE_SAGE=1 (recipe table),
# H3_UNGATE_CUDNN=1 (cuDNN FP8 arm), TLLM_FMHA_LIBS=+prims_ts, CUTE_DSL_ARCH=sm_100f when the pin's DSL < 4.8.
# Pin fixes applied in every overlay mode (H3_PIN_FIXES): lever21-defect (a) study/patches/lever22-sol-kernels-pin.patch
# (sparse/sol/kernels.py do_not_specialize num_blocks, needed under torch.compile, ¶15584; 5b7125bc39b) and defect (c)
# study/patches/lever22-prims-ts-sm107-pin.patch (fmha/prims_ts.py SUPPORTED_SMS gains 107, else the manager never registers
# and study/patches/lever22-qknorm-fp4-pin.patch (transformer_minimax_h3.py _use_fused_qk_norm_rope reads the Fp4QuantizedTensor codes that the
# lever-9 producer fusion hands in; lever 8, which carries that hunk in rc1, rejects on the pin: job 976231 died on .is_cuda).
# PrimsTSBlockSparse on SM 107, ¶15776). The pin already carries #20224's @torch.compiler.disable (lever 21's defect (b)).
set -uo pipefail
export H=/lustre/fsw/coreai_comparch_trtllm/kanghwanj/h3
export S=$H/sweep-nvfp4
export WT=${H3_WT:-$S/wt/lever22-solsage}                 # worktree of sweep/nvfp4/lever22-solsage: study/{e2e-17216,hecate,jobs}
export R=${H3_RESULTS:-$S/results/lever22-solsage/R200}
export SQSH=${H3_SQSH:-$H/containers/trtllm-pin-c8ee74b9-sm107.sqsh}
export MODEL_BF16=$H/models/MiniMax-H3
export MODEL_NVFP4=$H/models/MiniMax-H3-NVFP4
export H3_PARTS=${H3_PARTS:-"probe e2e nsys side cmp"}
export H3_OVERLAY_MODE=${H3_OVERLAY_MODE:-auto}            # auto | on | off  (off = BF16 checkpoint, no overlays)
export H3_E2E_ARMS=${H3_E2E_ARMS:-"p-l12 p-f8 p-tdense p-sol-t2-dB p-solsage-16"}
export H3_E2E_ARMS_19K=${H3_E2E_ARMS_19K:-"p-l12 p-f8 p-tdense p-tsage-16 p-sol-t2-dB p-solsage-16 p-solsage-4"}
export H3_SEQS=${H3_SEQS:-"s19k s104k s38k"}
export H3_E2E_PASSES=${H3_E2E_PASSES:-2}                   # s19k; the long shapes run one pass
export H3_NSYS_SEQS=${H3_NSYS_SEQS:-"s19k s104k"}
export H3_NSYS_ARMS=${H3_NSYS_ARMS:-"p-f8 p-sol-t2-dB p-solsage-16"}
export H3_GATE_ARMS=${H3_GATE_ARMS:-"p-l12 p-f8 p-sol-t2-dB p-solsage-16 p-solsage-4"}
export H3_GATE_RERUN_ARM=${H3_GATE_RERUN_ARM:-p-solsage-16}
export H3_SHIP_ARMS=${H3_SHIP_ARMS:-"p-solsage-16 p-f8"}
export H3_SHIP_PROMPTS=${H3_SHIP_PROMPTS:-$H/study/hecate/requests/ship-prompts-544x960x124.json}
export H3_SOL_TAU=${H3_SOL_TAU:-2}
export H3_SOL_CUT_A=${H3_SOL_CUT_A:-0.9877}                # lever 21's cutoffs (¶15574): t_k = 12s/(1+11s), s = 1 - k/27; A = 4 dense steps,
export H3_SOL_CUT_B=${H3_SOL_CUT_B:-0.9566}                # B = 10 dense steps; the probe's timestep log confirms them (results.json sol.dense_steps)
export H3_SOL_DENSE_A=${H3_SOL_DENSE_A:-4}                 # dense-prefix lengths the cutoffs are derived for
export H3_SOL_DENSE_B=${H3_SOL_DENSE_B:-10}
export PDIR=$S/study/patches/recipe
export H3_MANIFEST=${H3_MANIFEST:-$PDIR/final-overlays.txt}
if [[ -z "${H3_PATCHES:-}" && $H3_OVERLAY_MODE != off ]]; then
  H3_PATCHES=""
  while read -r n sha name _; do
    f=$PDIR/$name
    [[ -f $f ]] || { echo "MANIFEST: missing $f"; exit 2; }
    have=$(sha256sum "$f" | cut -c1-16); [[ $have == "$sha" ]] || { echo "MANIFEST: sha mismatch for $name: manifest $sha, file $have"; exit 2; }
    H3_PATCHES="$H3_PATCHES $f"
  done < <(grep -E "^[0-9]+[[:space:]]+[0-9a-f]{16}[[:space:]]" "$H3_MANIFEST" | sort -n)
  export H3_PATCHES; echo "manifest $(sha256sum "$H3_MANIFEST" | cut -c1-16) -> $(echo $H3_PATCHES | wc -w) overlays verified"
fi
export H3_PIN_FIXES=${H3_PIN_FIXES:-"$WT/study/patches/lever22-sol-kernels-pin.patch $WT/study/patches/lever22-prims-ts-sm107-pin.patch $WT/study/patches/lever22-qknorm-fp4-pin.patch"}   # applied to the pin package in every overlay mode
for f in $H3_PIN_FIXES; do [[ -f $f ]] || { echo "PIN_FIXES: missing $f"; exit 2; }; done
export H3_MARKERS=${H3_MARKERS:-"FUSED_QUANT H3_LEVER10_FOLD async_ulysses=use_async_ulysses _l8_norm_rope_tensor recv_view_dtype _l8_rank _use_fused_fp8_qk _out_proj fused_out_quant"}
export H3_L14_MARKERS=${H3_L14_MARKERS:-"H3_L14_SCHED H3_L14_VAE_FP8 H3_L14_VAE_TILE"}
mkdir -p "$R/e2e" "$R/gate" "$R/ship-u1" "$R/probe" "$H/cache"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) arms=$H3_E2E_ARMS arms19k=$H3_E2E_ARMS_19K seqs=$H3_SEQS passes=$H3_E2E_PASSES nsys=$H3_NSYS_ARMS@$H3_NSYS_SEQS gate=$H3_GATE_ARMS ship=$H3_SHIP_ARMS parts=$H3_PARTS overlays=$H3_OVERLAY_MODE sqsh=$SQSH wt=$WT results=$R"
scontrol show node "$(hostname)" | grep -E "NodeName|ActiveFeatures" | head -2
nvidia-smi --query-gpu=index,name,power.limit,clocks.max.sm,memory.total,driver_version --format=csv
[[ -d $WT/study/e2e-17216 && -f $WT/study/hecate/probe22/sitecustomize.py ]] || { echo "WORKTREE incomplete: $WT (need study/e2e-17216, study/hecate/probe22)"; exit 2; }
for m in "$MODEL_BF16" "$MODEL_NVFP4"; do [[ -d $m/transformer && -f $m/text_encoder/config.json ]] || { echo "MODEL dir incomplete: $m"; ls -la "$m"; exit 2; }; done
# The worktree harness copies need the study root's shared files beside them (idempotent symlinks, git-ignored).
ln -sfn $H/study/h3_ungate.py $WT/study/h3_ungate.py
ln -sfn $H/study/e2e-17216/requests $WT/study/e2e-17216/requests
for f in requests export_clips.py quality_compare.py; do [[ -e $H/study/hecate/$f ]] && ln -sfn $H/study/hecate/$f $WT/study/hecate/$f; done
( cd $WT && git rev-parse HEAD && git status --short study | head -20 ) | tee "$R/worktree-head.txt"

PROLOGUE='
set -uo pipefail
has() { [[ " $H3_PARTS " == *" $1 "* ]]; }
export HF_HUB_OFFLINE=1 TLLM_DISABLE_MPI=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4
ulimit -c 0
unset FA_DISABLE_2CTA H3_UNGATE_CUDNN H3_UNGATE_TRTLLM H3_UNGATE_SAGE
export PYTHONPATH=$WT/study/hecate/probe22:$WT/study/hecate/probe${PYTHONPATH:+:$PYTHONPATH}   # lever-22 shim chains lever 21'"'"'s shim (both in the worktree)
pkg=$(python -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])" | tail -n 1)
echo "[job] $(hostname) sm=$(python -c "import torch; print(torch.cuda.get_device_capability())" | tail -n 1) tensorrt_llm $(python -c "import tensorrt_llm; print(tensorrt_llm.__version__)" 2>/dev/null | tail -n 1) site=$pkg"
dsl=$(python -c "import importlib.metadata as m; print(m.version(\"nvidia-cutlass-dsl\"))" 2>/dev/null | tail -n 1)
# 4.8.0.dev0 (the rc1 image) already targets SM 107 natively; forcing sm_100f makes the PrimsTS Sage kernel fail NVVM ("unsupported operation", job 975192), so compare the release tuple, not the dev-tagged version
if python -c "import sys; from packaging.version import Version as V; sys.exit(0 if V(\"$dsl\").release[:2] < (4, 8) else 1)" 2>/dev/null; then export CUTE_DSL_ARCH=sm_100f; fi
echo "[job] nvidia-cutlass-dsl $dsl CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-unset}"
dryrun_patches() {  # whole chain on a scratch copy of the pin package; 0 clean, 1 reject
  local scratch; scratch=$(mktemp -d); mkdir -p "$scratch/tensorrt_llm/_torch"; cp -r "$pkg/tensorrt_llm/_torch/visual_gen" "$pkg/tensorrt_llm/_torch/attention" "$scratch/tensorrt_llm/_torch/"
  local rc=0 p
  for p in "$@"; do
    if ! patch -p1 -d "$scratch" --forward --batch < "$p" > "$R/patch-dryrun-$(basename "$p").log" 2>&1 || grep -qE "FAILED|rejects" "$R/patch-dryrun-$(basename "$p").log"; then
      echo "[patch] CHAIN DRY RUN: reject at $(basename "$p")"; grep -E "FAILED|rejects|Hunk" "$R/patch-dryrun-$(basename "$p").log" | head -5; rc=1; break; fi
  done
  rm -r "$scratch"; return $rc
}
apply_patches() {
  local p
  for p in "$@"; do
    patch -p1 -d "$pkg" --forward --batch < "$p" > "$R/patch-$(basename "$p").log" 2>&1 || { echo "[patch] APPLY FAILED: $p"; cat "$R/patch-$(basename "$p").log"; exit 2; }
    grep -qE "FAILED|rejects" "$R/patch-$(basename "$p").log" && { echo "[patch] REJECT in apply: $p"; cat "$R/patch-$(basename "$p").log"; exit 2; }
    echo "[patch] $(basename "$p") applied"
  done
  local M=$pkg/tensorrt_llm/_torch/visual_gen/models/minimax_h3 m missing=""
  for m in $H3_MARKERS; do grep -q "$m" "$M/transformer_minimax_h3.py" || missing="$missing $m"; done
  for m in $H3_L14_MARKERS; do grep -q "$m" "$M/pipeline_minimax_h3.py" || missing="$missing $m"; done
  grep -q "lever19" "$H3_MANIFEST" && { grep -q "H3_CUDNN_PREFIX_INPLACE" "$pkg/tensorrt_llm/_torch/visual_gen/attention/backends/cudnn.py" || missing="$missing H3_CUDNN_PREFIX_INPLACE(cudnn.py)"; }
  [[ -z "$missing" ]] && echo "[patch] markers present" || { echo "[patch] MARKERS MISSING:$missing"; exit 2; }
  python -c "import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]; print(\"[patch] composed modules parse\")" "$M/transformer_minimax_h3.py" "$M/pipeline_minimax_h3.py" || exit 2
  echo "[provenance] manifest $(sha256sum $H3_MANIFEST | cut -c1-16) $(basename $H3_MANIFEST)"
}
setenv() { export CUDA_VISIBLE_DEVICES=$1 TORCHINDUCTOR_CACHE_DIR=$H/cache/pin-inductor-$1 TRITON_CACHE_DIR=$H/cache/pin-triton-$1 CUDA_CACHE_PATH=$H/cache/pin-cuda-$1 FLASHINFER_WORKSPACE_BASE=$H/cache/pin-flashinfer-$1; }
L14="H3_L14_SCHED=1 H3_L14_VAE_FP8=1 H3_L14_VAE_TILE=288x272x32"
NVFP4_ON="H3_FUSED_QUANT=1 H3_FUSED_QUANT_WARPS=4 H3_LEVER10_FOLD=1 H3_OUT_QUANT=1 H3_CUDNN_PREFIX_INPLACE=1 $L14"
L8R="H3_FP8_SCALE_FUSION=dynamic H3_FP8_FUSION_SPLIT=1"
TRT="H3_UNGATE_TRTLLM=1 TLLM_FMHA_LIBS=+prims_ts"
SAGE="H3_UNGATE_SAGE=1"
req_e2e() { case $1 in s19k) echo t2va-544x960x124-27.json;; s38k) echo t2va-768x1344x124-27.json;; s104k) echo t2va-768x1344x345-27.json;; esac; }
'

srun --ntasks=1 --mpi=none --container-image="$SQSH" --container-name=lever22 --container-mounts="$H:$H" --container-workdir="$WT/study/e2e-17216" \
  --no-container-entrypoint --container-writable --export=ALL bash -c "$PROLOGUE"'
# --- overlays: the final chain on the pin, or the BF16 checkpoint without overlays (brief ¶15558) ---
mode=$H3_OVERLAY_MODE
if [[ $mode != off ]]; then
  if dryrun_patches $H3_PATCHES; then echo "[patch] chain dry run clean ($(echo $H3_PATCHES | wc -w) overlays) on the pin package"; mode=on
  elif [[ $mode == on ]]; then echo "[patch] chain rejected and H3_OVERLAY_MODE=on: abort"; exit 2
  else echo "[patch] chain rejected on the pin package: every arm runs the BF16 checkpoint without overlays (H3_OVERLAY_MODE=auto -> off)"; mode=off; fi
fi
if [[ $mode == on ]]; then apply_patches $H3_PATCHES; MODEL=$MODEL_NVFP4; COMMON=$NVFP4_ON; else MODEL=$MODEL_BF16; COMMON="H3_CUDNN_PREFIX_INPLACE=0"; fi
# Pin fixes (every mode): the Sol predictor kernel fix the compiled TRTLLM Sol arms need (¶15584 defect 1).
if dryrun_patches $H3_PIN_FIXES; then
  for p in $H3_PIN_FIXES; do
    patch -p1 -d "$pkg" --forward --batch < "$p" > "$R/patch-$(basename "$p").log" 2>&1 && ! grep -qE "FAILED|rejects" "$R/patch-$(basename "$p").log" || { echo "[pinfix] APPLY FAILED: $p"; cat "$R/patch-$(basename "$p").log"; exit 2; }
    echo "[pinfix] $(basename "$p") applied"
  done
  grep -q "do_not_specialize" "$pkg/tensorrt_llm/_torch/visual_gen/attention/backends/sparse/sol/kernels.py" || { echo "[pinfix] marker missing in sol/kernels.py"; exit 2; }
  grep -q "SUPPORTED_SMS = frozenset({100, 103, 107})" "$pkg/tensorrt_llm/_torch/attention/backends/fmha/prims_ts.py" || { echo "[pinfix] marker missing in fmha/prims_ts.py (SM 107)"; exit 2; }
  grep -q "_l22_qk_norm_input" "$M/transformer_minimax_h3.py" || { echo "[pinfix] marker missing in transformer_minimax_h3.py (Fp4 input to the fused QK-norm check)"; exit 2; }
else echo "[pinfix] rejected on the pin package: abort (the Sol arms need both: predictor kernel under compile, PrimTS registration on SM 107)"; exit 2; fi
echo "overlay_mode=$mode model=$MODEL pin_fixes=$(for p in $H3_PIN_FIXES; do basename $p; done | tr "\n" " ")" | tee "$R/overlay-mode.txt"
M=$pkg/tensorrt_llm/_torch/visual_gen/models/minimax_h3
sha256sum ${H3_PATCHES:-} $H3_PIN_FIXES "$pkg"/tensorrt_llm/_torch/visual_gen/attention/backends/sparse/sol/kernels.py "$M"/transformer_minimax_h3.py "$M"/pipeline_minimax_h3.py "$pkg"/tensorrt_llm/_torch/visual_gen/attention/backends/trtllm.py "$pkg"/tensorrt_llm/_torch/visual_gen/attention/backends/sparse/sol/backend.py "$pkg"/tensorrt_llm/_torch/attention/backends/fmha/prims_ts_block_sparse.py "$pkg"/tensorrt_llm/_torch/attention/backends/fmha/prims_ts.py "$pkg"/tensorrt_llm/visual_gen/args.py $WT/study/hecate/probe22/sitecustomize.py $WT/study/hecate/probe/sitecustomize.py $WT/study/e2e-17216/h3_e2e_bench.py $WT/study/hecate/gen_quality.py 2>/dev/null | tee "$R/overlay-sha256.txt" | cut -c1-16,65-
tag="$(basename "$SQSH") + overlays:$mode $( [[ $mode == on ]] && sha256sum $H3_MANIFEST | cut -c1-16 )"

arm_backend() {  # arm -> harness attention flags
  case $1 in
  p-l12)        echo "--attention-backend FA4";;
  p-f8)         echo "--attention-backend CUDNN --qk-dtype fp8 --v-dtype fp8";;
  p-tdense)     echo "--attention-backend TRTLLM";;
  p-tsage-16)   echo "--attention-backend TRTLLM --qk-dtype fp8 --v-dtype fp8 --quant-blocks 1,16,1";;
  p-sol-t2-dA)  echo "--attention-backend TRTLLM --sol $H3_SOL_TAU,$H3_SOL_CUT_A";;
  p-sol-t2-dB)  echo "--attention-backend TRTLLM --sol $H3_SOL_TAU,$H3_SOL_CUT_B";;
  p-solsage-16) echo "--attention-backend TRTLLM --sol $H3_SOL_TAU,$H3_SOL_CUT_B --qk-dtype fp8 --v-dtype fp8 --quant-blocks 1,16,1";;
  p-solsage-16-dA) echo "--attention-backend TRTLLM --sol $H3_SOL_TAU,$H3_SOL_CUT_A --qk-dtype fp8 --v-dtype fp8 --quant-blocks 1,16,1";;
  p-solsage-4)  echo "--attention-backend TRTLLM --sol $H3_SOL_TAU,$H3_SOL_CUT_B --qk-dtype fp8 --v-dtype fp8 --quant-blocks 1,4,1";;
  p-solsage-1)  echo "--attention-backend TRTLLM --sol $H3_SOL_TAU,$H3_SOL_CUT_B --qk-dtype fp8 --v-dtype fp8 --quant-blocks 1,1,1";;
  esac; }
arm_variant() {  # arm -> gen_quality variant string
  case $1 in
  p-l12)        echo "FA4:2cta=1";;
  p-f8)         echo "CUDNN:q=fp8:v=fp8";;
  p-tdense)     echo "TRTLLM";;
  p-tsage-16)   echo "TRTLLM:q=fp8:v=fp8:blk=1,16,1";;
  p-sol-t2-dA)  echo "TRTLLM:sol=$H3_SOL_TAU:dut=$H3_SOL_CUT_A";;
  p-sol-t2-dB)  echo "TRTLLM:sol=$H3_SOL_TAU:dut=$H3_SOL_CUT_B";;
  p-solsage-16) echo "TRTLLM:sol=$H3_SOL_TAU:dut=$H3_SOL_CUT_B:q=fp8:v=fp8:blk=1,16,1";;
  p-solsage-16-dA) echo "TRTLLM:sol=$H3_SOL_TAU:dut=$H3_SOL_CUT_A:q=fp8:v=fp8:blk=1,16,1";;
  p-solsage-4)  echo "TRTLLM:sol=$H3_SOL_TAU:dut=$H3_SOL_CUT_B:q=fp8:v=fp8:blk=1,4,1";;
  p-solsage-1)  echo "TRTLLM:sol=$H3_SOL_TAU:dut=$H3_SOL_CUT_B:q=fp8:v=fp8:blk=1,1,1";;
  esac; }
arm_env() {  # arm -> study overrides (the non-attention levers are identical across arms)
  case $1 in
  p-l12)       echo "$COMMON H3_FP8_SCALE_FUSION=off";;
  p-f8)        echo "$COMMON H3_UNGATE_CUDNN=1 $( [[ $mode == on ]] && echo $L8R || echo H3_FP8_SCALE_FUSION=off )";;
  p-tdense|p-sol-*) echo "$COMMON H3_FP8_SCALE_FUSION=off $TRT";;
  p-tsage-*|p-solsage-*) echo "$COMMON H3_FP8_SCALE_FUSION=off $TRT $SAGE";;
  esac; }

# --- probe (GPU 0): versions, ops, one eager Sol + Sage generation with the sparsity hook; cutoffs from its timestep log ---
if has probe; then
  setenv 0
  echo "== probe $(date -Is)"
  python - <<EOF 2>&1 | tee "$R/probe/env.txt"
import importlib.metadata as m, torch, tensorrt_llm, inspect
print("tensorrt_llm", tensorrt_llm.__version__, tensorrt_llm.__file__)
print("device", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
for p in ("nvidia-cutlass-dsl", "flashinfer-python", "nvidia-cudnn-frontend", "triton"):
    try: print(p, m.version(p))
    except Exception as e: print(p, "missing", e)
print("ops: sage_quantize", hasattr(torch.ops.trtllm, "sage_quantize"), "visual_gen_sol_predictor", hasattr(torch.ops.trtllm, "visual_gen_sol_predictor"))
from tensorrt_llm._torch.attention.backends.fmha import registry
print("fmha libs default", registry.DEFAULT_FMHA_LIBS)
from tensorrt_llm._torch.visual_gen.attention.backends.sparse.sol import backend as sol
src = inspect.getsource(sol.SOLTrtllmAttention)
print("SOLTrtllmAttention mentions quant_attention_config (a rejection in older trees):", "quant_attention_config" in src)
from tensorrt_llm._torch.attention.backends.prims_ts import context
print("prims_ts supported cc", context._SUPPORTED_COMPUTE_CAPABILITIES)
from tensorrt_llm._torch.attention.backends.fmha import prims_ts as pts
print("PrimsTSFmha.SUPPORTED_SMS", sorted(pts.PrimsTSFmha.SUPPORTED_SMS))
EOF
  out=$R/probe/solsage-eager-s19k; mkdir -p "$out"
  echo "[probe] eager Sol + Sage (no dense prefix) with --sol-probe start $(date -Is)"
  env $(arm_env p-solsage-16) python -u h3_e2e_bench.py --model "$MODEL" --request "requests/$(req_e2e s19k)" --output "$out" --label probe-solsage-16-eager \
    --attention-backend TRTLLM --sol $H3_SOL_TAU,none --sol-probe --qk-dtype fp8 --v-dtype fp8 --quant-blocks 1,16,1 --compile off --cuda-graph off --warmup 1 --timed 1 --save-output off --container "$tag" > "$out.log" 2>&1
  echo "[probe] rc=$? $(date -Is)"; grep -E "H3_E2E_BENCH_DONE|probe22|sitecustomize|sol_probe|\"sol\"|Error|error" "$out.log" | grep -v "error_" | tail -8 | cut -c1-300
  if python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get(\"timesteps\") and d.get(\"sol_probe\") else 1)" "$out/results.json" 2>/dev/null; then
    read -r cutA cutB <<< "$(python - "$out/results.json" "$H3_SOL_DENSE_A" "$H3_SOL_DENSE_B" <<EOF
import json, sys
d = json.load(open(sys.argv[1])); ts = sorted(d["timesteps"], reverse=True)   # descending noise in [0, 1]
def cut(n):  # cutoff so that the first n steps (largest timesteps) run dense: midpoint between step n and n+1
    return (ts[n - 1] + ts[n]) / 2 if 0 < n < len(ts) else (ts[0] + 1e-6 if n >= len(ts) else 0.0)
print(f"{cut(int(sys.argv[2])):.6f} {cut(int(sys.argv[3])):.6f}")
EOF
)"
    [[ -z $H3_SOL_CUT_A ]] && H3_SOL_CUT_A=$cutA; [[ -z $H3_SOL_CUT_B ]] && H3_SOL_CUT_B=$cutB
    python -c "import json,sys; d=json.load(open(sys.argv[1])); p=d[\"sol_probe\"]; print(\"[probe] timesteps\", [round(t,4) for t in d[\"timesteps\"]]); print(\"[probe] route\", p[\"route\"], \"calls\", p[\"calls\"], \"kept fraction (all sparse, mean)\", p[\"kept_fraction_mean_all_sparse_calls\"]); print(\"[probe] per step\", [round(v[\"kept_fraction_mean\"],3) for k,v in sorted(p[\"by_transformer_call\"].items(), key=lambda kv:int(kv[0]))])" "$out/results.json"
  else
    echo "[probe] no timesteps / sol_probe in results.json: the Sol + Sage path did not run; see $out.log"; tail -30 "$out.log" | cut -c1-240
    [[ -z $H3_SOL_CUT_B ]] && { echo "LEVER22: no cutoff and no probe: abort"; exit 5; }
  fi
  echo "cutA=$H3_SOL_CUT_A (dense $H3_SOL_DENSE_A) cutB=$H3_SOL_CUT_B (dense $H3_SOL_DENSE_B) tau=$H3_SOL_TAU" | tee "$R/probe/cutoffs.txt"
  export H3_SOL_CUT_A H3_SOL_CUT_B
  # kernel names of the eager Sol + Sage generation (sparse phase only, no dense prefix): the Sage quantize kernels must appear
  which nsys >/dev/null 2>&1 || export PATH=$PATH:/opt/nvidia/nsight-systems/current/bin:/usr/local/cuda/bin
  outn=$R/probe/solsage-eager-s19k-nsys; mkdir -p "$outn"
  env $(arm_env p-solsage-16) nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx -o "$outn/profile" --force-overwrite true \
    python -u h3_e2e_bench.py --model "$MODEL" --request "requests/$(req_e2e s19k)" --output "$outn" --label probe-solsage-16-eager-nsys \
    --attention-backend TRTLLM --sol $H3_SOL_TAU,none --qk-dtype fp8 --v-dtype fp8 --quant-blocks 1,16,1 --compile off --cuda-graph off --warmup 1 --timed 1 --profile --save-output off --container "$tag" > "$outn.log" 2>&1
  echo "[probe] nsys rc=$? $(date -Is)"
  nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o "$outn/kernsum" "$outn/profile.nsys-rep" > "$outn/nsys-stats.log" 2>&1
  head -n 12 "$outn/kernsum_cuda_gpu_kern_sum.csv" 2>/dev/null | cut -c1-220; grep -i -c -E "sage|quant" "$outn/kernsum_cuda_gpu_kern_sum.csv" 2>/dev/null | sed "s/^/[probe] sage|quant kernel rows: /"
  rm -f "$outn/profile.nsys-rep"
fi
[[ -z ${H3_SOL_CUT_B:-} ]] && { echo "LEVER22: H3_SOL_CUT_B unset (probe skipped?): set it or run the probe"; exit 5; }
[[ -z ${H3_SOL_CUT_A:-} ]] && H3_SOL_CUT_A=$H3_SOL_CUT_B

arms=($H3_E2E_ARMS); arms19k=($H3_E2E_ARMS_19K); seqs=($H3_SEQS)
e2e_arm() {  # seq arm pass
  local seq=$1 arm=$2 pass=$3 name=$2-$1-p$3; mkdir -p "$R/e2e/$seq"
  echo "[e2e gpu0 $seq] $name start $(date -Is) env=$(arm_env "$arm") flags=$(arm_backend "$arm")"
  env $(arm_env "$arm") python -u h3_e2e_bench.py --model "$MODEL" --request "requests/$(req_e2e "$seq")" --output "$R/e2e/$seq/$name" --label "$name" \
    $(arm_backend "$arm") --compile on --cuda-graph off --warmup 1 --timed 3 --container "$tag" > "$R/e2e/$seq/$name.log" 2>&1
  echo "[e2e gpu0 $seq] $name rc=$? $(date -Is)"; grep -E "H3_E2E_BENCH_DONE|wall_s_median|probe22|Error|error" "$R/e2e/$seq/$name.log" | grep -v "error_" | tail -3 | cut -c1-240
}
e2e_nsys() {  # seq arm
  local seq=$1 arm=$2 name=$2-$1-nsys; mkdir -p "$R/e2e/$seq/$name"
  which nsys >/dev/null 2>&1 || export PATH=$PATH:/opt/nvidia/nsight-systems/current/bin:/usr/local/cuda/bin
  echo "[e2e gpu0 $seq] nsys $name start $(date -Is)"
  env $(arm_env "$arm") nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx --cuda-graph-trace=node -o "$R/e2e/$seq/$name/profile" --force-overwrite true \
    python -u h3_e2e_bench.py --model "$MODEL" --request "requests/$(req_e2e "$seq")" --output "$R/e2e/$seq/$name" --label "$name" \
      $(arm_backend "$arm") --compile on --cuda-graph off --warmup 1 --timed 1 --profile --save-output off --container "$tag" > "$R/e2e/$seq/$name.log" 2>&1
  echo "[e2e gpu0 $seq] nsys $name rc=$? $(date -Is)"
  nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv -o "$R/e2e/$seq/$name/kernsum" "$R/e2e/$seq/$name/profile.nsys-rep" > "$R/e2e/$seq/$name/nsys-stats.log" 2>&1
  head -n 8 "$R/e2e/$seq/$name/kernsum_cuda_gpu_kern_sum.csv" 2>/dev/null | cut -c1-220
  rm -f "$R/e2e/$seq/$name/profile.nsys-rep"   # the sqlite + csv carry the data; keep lustre small
}
side_gate() {  # arm seed [suffix]  (GPU set by the caller)
  local arm=$1 seed=$2 suffix=${3:-} out=$R/gate/$1/seed$2${3:-}; mkdir -p "$R/gate/$arm"
  echo "[gate gpu$CUDA_VISIBLE_DEVICES] $arm seed $seed$suffix start $(date -Is)"
  ( cd $WT/study/hecate && env $(arm_env "$arm") python -u gen_quality.py --model "$MODEL" --request requests/quality-544x960x124.json --variant "$(arm_variant "$arm")" \
      --compile on --cuda-graph off --pipeline-warmup on --output "$out" --seed "$seed" --steps 28 > "$out.log" 2>&1 )
  echo "[gate gpu$CUDA_VISIBLE_DEVICES] $arm seed $seed$suffix rc=$? $(date -Is) steps=$(ls "$out/steps" 2>/dev/null | wc -l)"; grep -E "H3_QUALITY_GEN_DONE|Error|error" "$out.log" | grep -v "error_" | tail -2 | cut -c1-240
}
side_ship() {  # arm  (12 ship prompts at U1, gen_quality_batch.py layout p*/<arm>/output.pt)
  local arm=$1
  echo "[ship gpu$CUDA_VISIBLE_DEVICES] $arm start $(date -Is)"
  ( cd $WT/study/hecate && env $(arm_env "$arm") python -u gen_quality_batch.py --model "$MODEL" --prompts "$H3_SHIP_PROMPTS" --variant "$(arm_variant "$arm")" --compile on \
      --ulysses 1 --root "$R/ship-u1" --arm "$arm" > "$R/ship-u1/$arm.log" 2>&1 )
  echo "[ship gpu$CUDA_VISIBLE_DEVICES] $arm rc=$? $(date -Is) clips=$(ls -d "$R"/ship-u1/p*/"$arm"/output.pt 2>/dev/null | wc -l)"; grep -E "H3_QUALITY_BATCH_DONE|Error|error" "$R/ship-u1/$arm.log" | grep -v "error_\|ProcessGroupNCCL" | tail -2 | cut -c1-240
}
pids=()
if has side; then
  cp "$H3_SHIP_PROMPTS" "$R/ship-u1/prompts.json"
  gate=($H3_GATE_ARMS); ship=($H3_SHIP_ARMS)
  # three lanes: every gate arm x seeds 42 / 7 / 123 spread by seed, the rerun of the gate arm, then the ship arms
  ( setenv 1; for a in "${gate[@]}"; do side_gate "$a" 42; done; side_gate "$H3_GATE_RERUN_ARM" 42 -rerun; [[ -n ${ship[2]:-} ]] && side_ship "${ship[2]}" ) & pids+=($!)
  ( setenv 2; for a in "${gate[@]}"; do side_gate "$a" 7; done; side_ship "${ship[0]}" ) & pids+=($!)
  ( setenv 3; for a in "${gate[@]}"; do side_gate "$a" 123; done; [[ -n ${ship[1]:-} ]] && side_ship "${ship[1]}" ) & pids+=($!)
fi
if has e2e; then
  setenv 0
  for seq in "${seqs[@]}"; do
    list=("${arms[@]}"); passes=1; [[ $seq == s19k ]] && { list=("${arms19k[@]}"); passes=$H3_E2E_PASSES; }
    for pass in $(seq "$passes"); do
      order=("${list[@]}"); (( pass % 2 == 0 )) && order=($(printf "%s\n" "${list[@]}" | tac))
      echo "[e2e $seq pass $pass] order: ${order[*]}"; for arm in "${order[@]}"; do e2e_arm "$seq" "$arm" "$pass"; done
    done
  done
fi
if has nsys; then setenv 0; for seq in $H3_NSYS_SEQS; do for arm in $H3_NSYS_ARMS; do e2e_nsys "$seq" "$arm"; done; done; fi
if (( ${#pids[@]} )); then echo "[side] waiting for GPU 1-3 work $(date -Is)"; wait "${pids[@]}"; echo "[side] done $(date -Is)"; fi
if has side && [[ -f $WT/study/hecate/export_clips.py ]]; then
  echo "== clip export $(date -Is)"
  ( export CUDA_VISIBLE_DEVICES=""; cd $WT/study/hecate; out=$R/ship-u1/clips; mkdir -p "$out"; cp "$R"/ship-u1/prompts.json "$R"/ship-u1/*.provenance.json "$out"/ 2>/dev/null
    for p in "$R"/ship-u1/p*/; do pid=$(basename "$p"); python -u export_clips.py "$p" $H3_SHIP_ARMS --out "$out/$pid" 2>&1 | grep -v "^$" | tail -n 2; done
    du -sh "$out"; ls "$out" | head -20 )
fi
if has cmp; then
  echo "== hashes and bitcmp $(date -Is)"; export CUDA_VISIBLE_DEVICES=""
  [[ -f $S/study/jobs/hash_outputs.py ]] && python -u $S/study/jobs/hash_outputs.py "$R/e2e" "$R/gate" "$R/ship-u1" --out "$R/hashes.txt" 2>&1 | sed "s|$R/||" | tail -40
  B=$S/study/jobs/bitcmp.py
  for seq in "${seqs[@]}"; do
    list=("${arms[@]}"); [[ $seq == s19k ]] && list=("${arms19k[@]}")
    for arm in "${list[@]}"; do
      [[ -d $R/e2e/$seq/$arm-$seq-p1 && -d $R/e2e/$seq/$arm-$seq-p2 ]] && { echo "[cmp] $arm p1 vs p2 ($seq):"; python -u $B "$R/e2e/$seq/$arm-$seq-p1" "$R/e2e/$seq/$arm-$seq-p2" --out "$R/e2e/$seq/bitcmp-$arm-p1p2.json" 2>&1 | tail -1; }
    done
  done
  g=$R/gate/$H3_GATE_RERUN_ARM; [[ -d $g/seed42 && -d $g/seed42-rerun ]] && { echo "[cmp] gate $H3_GATE_RERUN_ARM seed 42 vs rerun:"; python -u $B "$g/seed42" "$g/seed42-rerun" --out "$g/bitcmp-seed42.json" 2>&1 | tail -1; }
  [[ -f $S/study/jobs/summarize_u8_arms.py ]] && python -u $S/study/jobs/summarize_u8_arms.py "$R" "$R" --control p-l12 2>&1 | tail -40
fi
du -sh $R/* 2>/dev/null
echo "[lever22] done $(date -Is)"
'
echo "step rc=$? $(date -Is)"
echo "end=$(date -Is)"
echo SWEEP_LEVER22_DONE
