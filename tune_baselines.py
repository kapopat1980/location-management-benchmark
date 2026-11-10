"""Tuned-baseline check: sweep the threshold of distance-based (D) and movement-based (M) registration
on the SAME trajectories and calls as the main/sweep experiments (identical seeds), and report the
best fixed threshold per regime. Does not touch UPAL or any earlier result."""
import numpy as np, json, glob, sys
import sim
from sim import *

def run_baselines(N=120, T=5000, W=1500, lam_scale=0.02, seed=0):
    rng = np.random.default_rng(seed)
    pos, cls = gen_traj(N, T, rng)
    lam_u = lam_scale * rng.uniform(0.5, 1.5, size=N)
    calls = rng.random((N, T)) < lam_u[:, None]
    schemes = []
    for d in [1, 2, 3, 4, 5, 6, 7, 8]: schemes.append(DBR(N, T, pos[:, 0], d, True))
    for m in [2, 3, 4, 5, 6, 7, 8]: schemes.append(MB(N, T, pos[:, 0], m))
    for t in range(1, T):
        prev = pos[:, t-1]; cur = pos[:, t]; call = calls[:, t]
        for sc in schemes: sc.step(t, cur, prev, call, t >= W, None)
    steps = T - W; out = {}
    for sc in schemes:
        a = sc.acc
        cost = ((10.0 * a.regs + 1.0 * a.cells) / steps * 1000)
        out[sc.name] = {"cost": float(cost.mean()), "cls": [float(cost[cls == c].mean()) for c in range(3)]}
    return out

if __name__ == "__main__":
    res = {}
    for lam, seeds in [(0.005, [1,2,3]), (0.01, [1,2,3]), (0.02, [1,2,3,4,5]), (0.05, [1,2,3]), (0.1, [1,2,3])]:
        runs = [run_baselines(lam_scale=lam, seed=s) for s in seeds]
        res[str(lam)] = {n: {"cost": float(np.mean([r[n]["cost"] for r in runs])),
                             "sd": float(np.std([r[n]["cost"] for r in runs], ddof=1)),
                             "cls": np.mean([r[n]["cls"] for r in runs], axis=0).tolist(),
                             "per_seed": [r[n]["cost"] for r in runs]} for n in runs[0]}
        best = min(res[str(lam)], key=lambda n: res[str(lam)][n]["cost"])
        print(lam, "best:", best, round(res[str(lam)][best]["cost"], 1), flush=True)
    json.dump(res, open("results/tuned.json", "w"), indent=1)
