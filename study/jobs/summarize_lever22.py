#!/usr/bin/env python3
# ruff: noqa: E501
"""Summarize a lever-22 (Sol-Attn + SageAttention, pin container) job folder into the result-post tables.

Layout written by study/jobs/lever22-solsage-r200.sbatch:
  <root>/e2e/<shape>/<arm>-<shape>-p<N>/results.json        compiled e2e arms (h3_e2e_bench.py)
  <root>/e2e/<shape>/<arm>-<shape>-nsys/kernsum_cuda_gpu_kern_sum.csv
  <root>/e2e/<shape>/bitcmp-<arm>-p1p2.json
  <root>/probe/<name>/results.json                           eager probes (--sol-probe)
  <root>/probe/<name>-nsys/kernsum_cuda_gpu_kern_sum.csv
  <root>/gate/<arm>/seed<N>/, <root>/gate/<arm>/bitcmp-seed42.json

Usage: summarize_lever22.py ROOT [--base p-l12] [--anchor p-f8] [--part R200]
S = T_base / T_arm on the median generation wall of the arm's passes (in-job); Δ % = (1 - T_arm / T_base) * 100.
Pairs printed below the tables: Sage on/off (p-solsage-16 vs p-sol-t2-dB), dense Sage (p-tsage-16 vs p-tdense),
backend swap (p-tdense vs p-l12).
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

PAIRS = [
    ("Sage on / off on the Sol route", "p-solsage-16", "p-sol-t2-dB"),
    ("Sage (1,4,1) / (1,16,1)", "p-solsage-4", "p-solsage-16"),
    ("Sage on / off on the dense TRTLLM route", "p-tsage-16", "p-tdense"),
    ("backend swap TRTLLM dense / FA4", "p-tdense", "p-l12"),
    ("Sol + Sage / cuDNN FP8", "p-solsage-16", "p-f8"),
]


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def collect_e2e(root):
    arms = {}
    for rj in sorted(root.glob("e2e/*/*/results.json")):
        r = load(rj)
        if not r or "summary" not in r:
            continue
        shape = rj.parent.parent.name
        name = rj.parent.name
        m = re.match(rf"^(.*)-{shape}-p(\d+)$", name)
        if not m:
            continue
        arm = m.group(1)
        gens = [g for g in r.get("generations", []) if g.get("kind") == "timed"]
        e = arms.setdefault(
            (shape, arm),
            {
                "walls": [],
                "wall_min": [],
                "steps": [],
                "mhz": [],
                "watts": [],
                "passes": 0,
                "sol": r.get("sol"),
                "resolved": r.get("resolved", {}),
                "timesteps": r.get("timesteps"),
                "env": r.get("env", {}),
                "quant": r.get("quant_attention_config")
                or (r.get("attention") or {}).get("quant_attention_config"),
            },
        )
        e["passes"] += 1
        e["walls"].append(r["summary"]["wall_s_median"])
        e["wall_min"].append(r["summary"]["wall_s_min"])
        e["steps"].append(r["summary"].get("step_ms_median_excl_first"))
        e["mhz"] += [g["nvml"]["sm_mhz_median"] for g in gens if g.get("nvml")]
        e["watts"] += [g["nvml"]["power_w_median"] for g in gens if g.get("nvml")]
    return arms


def collect_probes(root):
    probes = {}
    for rj in sorted(root.glob("probe/*/results.json")):
        r = load(rj)
        if not r or "sol_probe" not in r:
            continue
        sp = r["sol_probe"]
        by = sp.get("by_transformer_call", {})

        def at(step):
            e = by.get(str(step + 1))
            return None if e is None else e["kept_fraction_mean"]

        probes[rj.parent.name] = {
            "route": sp.get("route"),
            "calls": sp.get("calls"),
            "mean": sp.get("kept_fraction_mean_all_sparse_calls"),
            "step5": at(5),
            "step20": at(20),
            "refiner_calls": by.get("0", {}).get("layers"),
            "dense_steps": (r.get("sol") or {}).get("dense_steps"),
            "errors": sp.get("errors"),
            "timesteps": r.get("timesteps"),
        }
    return probes


def kernel_rows(csv_path):
    rows = []
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return [], 0.0
    total = sum(float(r.get("Total Time (ns)", 0)) for r in rows)
    return rows, total


def print_kernels(csv_path, top=7):
    rows, total = kernel_rows(csv_path)
    if not rows:
        return
    print("| kernel | total ms | share % | instances |")
    print("|---|---|---|---|")
    for r in rows[:top]:
        t = float(r.get("Total Time (ns)", 0))
        print(
            f"| `{r.get('Name', '')[:100]}` | {t / 1e6:.1f} | {100 * t / total if total else 0:.1f} | {r.get('Instances', 0)} |"
        )
    sage = [r for r in rows if re.search(r"sage|quant", r.get("Name", ""), re.I)]
    if sage:
        print("\nSage / quantize rows:")
        for r in sage[:8]:
            t = float(r.get("Total Time (ns)", 0))
            print(
                f"- `{r.get('Name', '')[:110]}` {t / 1e6:.1f} ms, {100 * t / total if total else 0:.2f} %, {r.get('Instances', 0)} instances"
            )
    else:
        print("\nno kernel row matching sage|quant")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--base", default="p-l12")
    ap.add_argument("--anchor", default="p-f8")
    ap.add_argument("--part", default="R200")
    args = ap.parse_args()
    root = args.root
    arms = collect_e2e(root)
    probes = collect_probes(root)
    shapes = sorted({s for s, _ in arms}, key=lambda s: int(s[1:-1]))
    print(f"## Lever 22 [{args.part}] — {root}\n")
    for shape in shapes:
        base = arms.get((shape, args.base))
        anchor = arms.get((shape, args.anchor))
        tb = med(base["walls"]) if base else None
        ta = med(anchor["walls"]) if anchor else None
        print(f"### {shape}\n")
        print(
            f"| arm | passes | gen wall s median (min) | step ms | S vs {args.base} | Δ % | S vs {args.anchor} | Δ % | dense steps | MHz / W | backend |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for (s, arm), e in sorted(
            arms.items(), key=lambda kv: (kv[0][0], med(kv[1]["walls"]) or 0)
        ):
            if s != shape:
                continue
            t = med(e["walls"])
            S = tb / t if tb and t else None
            d = (1 - t / tb) * 100 if tb and t else None
            Sa = ta / t if ta and t else None
            da = (1 - t / ta) * 100 if ta and t else None
            dense = (e.get("sol") or {}).get("dense_steps")
            res = e.get("resolved", {})
            backend = (
                f"{res.get('attention_backend')} / {res.get('attention_inner')}" if res else ""
            )
            print(
                f"| {arm} | {e['passes']} | {fmt(t)} ({fmt(min(e['wall_min']))}) | {fmt(med(e['steps']), 1)} | {fmt(S)} | {fmt(d, 1)} | {fmt(Sa)} | {fmt(da, 1)} | {dense if dense is not None else '–'} | {fmt(med(e['mhz']), 0)} / {fmt(med(e['watts']), 0)} | {backend} |"
            )
        print()
        print("Pairs (S = T_second / T_first, in-job):")
        for label, a, b in PAIRS:
            ea, eb = arms.get((shape, a)), arms.get((shape, b))
            if ea and eb:
                t1, t2 = med(ea["walls"]), med(eb["walls"])
                print(
                    f"- {label}: {a} {fmt(t1)} s vs {b} {fmt(t2)} s → S {fmt(t2 / t1)}, Δ {fmt((1 - t1 / t2) * 100, 1)} %"
                )
        print()
        spreads = [
            (arm, max(e["walls"]) - min(e["wall_min"]))
            for (s, arm), e in arms.items()
            if s == shape
        ]
        print(
            "Noise (median − min spread, s): "
            + ", ".join(f"{a} {sp:.3f}" for a, sp in sorted(spreads))
            + "\n"
        )
    ts = next((e["timesteps"] for e in arms.values() if e.get("timesteps")), None) or next(
        (p["timesteps"] for p in probes.values() if p.get("timesteps")), None
    )
    if ts:
        print(
            "Attention timesteps seen by the transformer: "
            + " ".join(f"{i}:{t:.4f}" for i, t in enumerate(ts))
            + "\n"
        )
    for name, p in sorted(probes.items()):
        print(
            f"probe {name}: route {p['route']}, calls {p['calls']}, kept blocks mean {fmt(p['mean'])} (step 5 {fmt(p['step5'])}, step 20 {fmt(p['step20'])}), "
            f"dense steps {p['dense_steps']}, refiner predictor calls {p['refiner_calls']}, errors {p['errors'][:2] if p.get('errors') else 'none'}"
        )
    print()
    bits = sorted(root.glob("e2e/*/bitcmp-*.json")) + sorted(root.glob("gate/*/bitcmp-*.json"))
    if bits:
        print("### bitcmp\n")
        for b in bits:
            r = load(b) or {}
            s = r.get("summary", r)
            verdict = s.get("bit_identical")
            detail = (
                f"{s.get('tensors_equal')} equal / {s.get('tensors_unequal')} unequal tensors"
                if "tensors_equal" in s
                else json.dumps(s)[:160]
            )
            print(
                f"- {b.parent.name} {b.stem}: {'bit-identical' if verdict else 'differs'} ({detail})"
            )
        print()
    for kcsv in sorted(root.glob("e2e/*/*-nsys/kernsum_cuda_gpu_kern_sum.csv")) + sorted(
        root.glob("probe/*-nsys/kernsum_cuda_gpu_kern_sum.csv")
    ):
        print(f"### kernels {kcsv.parent.parent.name} {kcsv.parent.name}\n")
        print_kernels(kcsv)
    for arm_dir in sorted(root.glob("gate/*")):
        if arm_dir.is_dir():
            seeds = sorted(
                d.name for d in arm_dir.iterdir() if d.is_dir() and (d / "output.pt").exists()
            )
            print(f"gate {arm_dir.name}: {', '.join(seeds) or 'no output.pt'}")
    ship = sorted(root.glob("ship-u1/p*/*/output.pt"))
    if ship:
        by_arm = {}
        for s in ship:
            by_arm[s.parent.name] = by_arm.get(s.parent.name, 0) + 1
        print("ship clips: " + ", ".join(f"{a} {n}" for a, n in sorted(by_arm.items())))


if __name__ == "__main__":
    main()
