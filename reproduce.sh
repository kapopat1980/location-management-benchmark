#!/bin/bash
# Regenerates every number, table input and figure used in the manuscript (about 1.5 hours on one core).
# Run from the repository root.  Pre-computed outputs are already in results/ and figs/.
set -e
mkdir -p results figs
# --- Benchmark A: hexagonal grid (Sections 7.1-7.9, Appendix D) ---
for s in 1 2 3 4 5; do python3 sim.py 120 5000 1500 0.02 $s results/main_s$s.json; done
for lam in 0.005 0.01 0.05 0.1; do
  for s in 1 2 3; do python3 sim.py 120 5000 1500 $lam $s results/lam${lam}_s$s.json; done
done
python3 analyze.py && python3 analyze2.py        # fixed-parameter comparison, per-class tables, figures
python3 tune_baselines.py && python3 analyze3.py # threshold-tuned baselines (Section 7.8)
python3 eval_selftune.py                         # self-tuning radius (Section 7.9)
python3 fig_tuned.py && python3 fig_main.py && python3 fig_design.py   # main-text figures
for l in 0.005 0.02 0.1; do
  python3 dev_v2.py $l 101; python3 dev_v3.py $l 101; python3 dev_v4.py $l 101; python3 dev_v5.py $l 101   # Appendix D
done
# --- Benchmark B: square grid, earlier AFLU simulator vendored in third_party/ (Section 7.10, Appendix E) ---
python3 bench_square.py && python3 bench_str_holdout.py && python3 bench_summary.py
python3 aflu_audit.py                            # reduced single-seed audit that exposed the look-ahead
# --- Benchmark C: real GPS trajectories (GeoLife, not redistributed; download it from Microsoft Research and unzip
#     "Geolife Trajectories 1.3" into ./geolife/) -------------------------------------------------------------
if [ -d "geolife/Geolife Trajectories 1.3/Data" ]; then
  python3 geolife_prep.py                                   # -> results/geolife_cells.npz (derived data; do not redistribute)
  for lam in 0.005 0.01 0.02 0.05 0.1; do for s in 1 2 3; do python3 run_real.py $lam $s results/real_lam${lam}_s$s.json; done; done
fi
python3 analyze_real.py                                     # -> results/real_summary.json, figs/fig_real.png (works from stored results too)
