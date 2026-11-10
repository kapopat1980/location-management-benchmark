import sys, numpy as np
from sim import *
lam = float(sys.argv[1]); seed = int(sys.argv[2]); N, T, W = 60, 4000, 1500
rng = np.random.default_rng(seed); pos, cls = gen_traj(N, T, rng)
lam_u = lam * rng.uniform(0.5, 1.5, size=N); calls = rng.random((N, T)) < lam_u[:, None]
prof = Profile(N); p0 = pos[:, 0]; arms = (1, 2, 3, 4, 5, 6, 7, 8)
sch = [DBR(N, T, p0, d, True, name=f"DBR-{d}") for d in (3, 5, 8)]
sch.append(DBRBandit(N, T, p0, prof, bandit=True, selective=True, kappa=1.0, name="DBR + per-context bandit", seed=seed, arms=arms))
for t in range(1, T):
    prev = pos[:, t-1]; cur = pos[:, t]; call = calls[:, t]; prof.update(prev, cur, call.astype(float))
    for s in sch: s.step(t, cur, prev, call, t >= W, prof)
steps = T - W
for s in sch:
    a = s.acc; cost = (10 * s.kappa * a.regs + a.cells) / steps * 1000
    print(f"lam={lam} {s.name:26s} cost={cost.mean():7.1f} ped={cost[cls==0].mean():6.0f} veh={cost[cls==1].mean():6.0f} com={cost[cls==2].mean():6.0f}", flush=True)
