"""Evaluation-seed run of the self-tuning radius scheme (DBR + per-context bandit, ring paging) on the SAME
trajectories/calls as the main and call-rate experiments. Compared offline with results/tuned.json."""
import json, numpy as np
from sim import *
res = {}
for lam, seeds in [(0.005, [1,2,3]), (0.01, [1,2,3]), (0.02, [1,2,3,4,5]), (0.05, [1,2,3]), (0.1, [1,2,3])]:
    costs = []; cls_costs = []
    for seed in seeds:
        N, T, W = 120, 5000, 1500
        rng = np.random.default_rng(seed); pos, cls = gen_traj(N, T, rng)
        lam_u = lam * rng.uniform(0.5, 1.5, size=N); calls = rng.random((N, T)) < lam_u[:, None]
        prof = Profile(N); p0 = pos[:, 0]
        sc = DBRBandit(N, T, p0, prof, bandit=True, selective=True, kappa=1.0, name="selftune", seed=seed, arms=(1,2,3,4,5,6,7,8))
        for t in range(1, T):
            prev = pos[:, t-1]; cur = pos[:, t]; call = calls[:, t]; prof.update(prev, cur, call.astype(float))
            sc.step(t, cur, prev, call, t >= W, prof)
        a = sc.acc; cost = (10.0 * a.regs + a.cells) / (T - W) * 1000
        costs.append(float(cost.mean())); cls_costs.append([float(cost[cls == c].mean()) for c in range(3)])
    res[str(lam)] = {"per_seed": costs, "mean": float(np.mean(costs)), "sd": float(np.std(costs, ddof=1)), "cls": np.mean(cls_costs, axis=0).tolist()}
    print(lam, round(np.mean(costs), 1), flush=True)
json.dump(res, open("results/selftune.json", "w"), indent=1)
