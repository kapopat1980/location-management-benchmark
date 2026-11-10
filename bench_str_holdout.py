"""Held-out confirmation of the self-tuning radius (STR) on Benchmark B: seeds 11-13, which were not used to design STR
or to produce any other Benchmark B result.  Compares STR with distance-based registration with ring-by-ring paging whose
threshold D = 1-12 is chosen in hindsight for each seed and call-to-mobility ratio."""
import sys, json, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_square as B
from lu_sim import *

N, T = 220, 1600
out = {}
for seed in [11, 12, 13]:
    rng = np.random.default_rng(seed)
    traj, moved, profiles, home = simulate_mobility(N, T, rng)
    pm = np.array([PROFILE_PARAMS[p]["p_move"] for p in profiles])
    for cmr in [0.25, 0.5, 1, 2, 4, 8, 16]:
        calls = simulate_calls(N, T, cmr, pm, rng); calls[:BURN_IN] = False
        grid = [(B.tot(B.run_trigger_sdf(traj, moved, calls, "dist", d)) / (N * T), d) for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]]
        best = min(grid); s = B.tot(B.run_str(traj, moved, calls, seed)) / (N * T)
        out[f"{seed}_{cmr}"] = {"str": s, "best_tuned": best[0], "best_D": best[1], "d2": dict((d, c) for c, d in grid)[2]}
        print(seed, cmr, round(s, 3), best, flush=True)
json.dump(out, open("results/bench_str_holdout.json", "w"), indent=1)
