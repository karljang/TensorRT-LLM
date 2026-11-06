#!/usr/bin/env bash
# §657 lever 22, second measured job (lead ¶15812): the 4-dense-step cutoff-A pair beside the job-1 arms, 19k x2 + 104k, no probe
# (cutoffs fixed from job 975237's timesteps), nsys + gate seeds + ship clips for the dA Sage arm. Run from $S on Hecate.
S=/lustre/fsw/coreai_comparch_trtllm/kanghwanj/h3/sweep-nvfp4
export H3_RESULTS=$S/results/lever22-solsage/R200-job2
export H3_PARTS="e2e nsys side cmp"
export H3_E2E_ARMS="p-l12 p-f8 p-sol-t2-dA p-solsage-16-dA p-solsage-16"
export H3_E2E_ARMS_19K="$H3_E2E_ARMS"
export H3_SEQS="s19k s104k"
export H3_NSYS_SEQS="s19k s104k"
export H3_NSYS_ARMS="p-solsage-16-dA"
export H3_GATE_ARMS="p-sol-t2-dA p-solsage-16-dA"
export H3_GATE_RERUN_ARM=p-solsage-16-dA
export H3_SHIP_ARMS="p-solsage-16-dA"
sbatch --export=ALL,H3_RESULTS,H3_PARTS,H3_E2E_ARMS,H3_E2E_ARMS_19K,H3_SEQS,H3_NSYS_SEQS,H3_NSYS_ARMS,H3_GATE_ARMS,H3_GATE_RERUN_ARM,H3_SHIP_ARMS \
  "$S/wt/lever22-solsage/study/jobs/lever22-solsage-r200.sbatch"
