#!/usr/bin/env bash
# §657 lever 22, second measured job (lead ¶15817): NVFP4 Linears in the pin container with the recipe overlays that apply
# (lever22-nvfp4-pin-overlays.txt: levers 1, 9, 10, 14; #21275 / 5 / 8 / 13 / 17 / 19 reject on the pin). Arms p-l12 (FA4 1-CTA,
# the pin default), p-f8 (cuDNN FP8, unfused scales), p-sol-t2-dB, p-solsage-16 at 19k x2 / 104k / 38k; nsys p-f8 + p-solsage-16;
# gate seeds for all four, ship clips for p-solsage-16, p-f8, p-l12 (judge pairs for ®462). No probe (cutoffs from job 975237).
S=/lustre/fsw/coreai_comparch_trtllm/kanghwanj/h3/sweep-nvfp4; WT=$S/wt/lever22-solsage
export H3_RESULTS=$S/results/lever22-solsage/R200-nvfp4
export H3_PARTS="e2e nsys side cmp"
export H3_OVERLAY_MODE=on
export H3_MANIFEST=$WT/study/patches/lever22-nvfp4-pin-overlays.txt
export H3_MARKERS="FUSED_QUANT H3_LEVER10_FOLD"
export H3_E2E_ARMS="p-l12 p-f8 p-sol-t2-dB p-solsage-16"
export H3_E2E_ARMS_19K="$H3_E2E_ARMS"
export H3_SEQS="s19k s104k s38k"
export H3_NSYS_SEQS="s19k s104k"
export H3_NSYS_ARMS="p-f8 p-solsage-16"
export H3_GATE_ARMS="p-l12 p-f8 p-sol-t2-dB p-solsage-16"
export H3_GATE_RERUN_ARM=p-solsage-16
export H3_SHIP_ARMS="p-solsage-16 p-f8 p-l12"
cd "$S" && sbatch --export=ALL,H3_RESULTS,H3_PARTS,H3_OVERLAY_MODE,H3_MANIFEST,H3_MARKERS,H3_E2E_ARMS,H3_E2E_ARMS_19K,H3_SEQS,H3_NSYS_SEQS,H3_NSYS_ARMS,H3_GATE_ARMS,H3_GATE_RERUN_ARM,H3_SHIP_ARMS \
  "$WT/study/jobs/lever22-solsage-r200.sbatch"
