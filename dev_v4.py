import sys, numpy as np
from sim import *
lam = float(sys.argv[1]); seed = int(sys.argv[2]); N, T, W = 60, 4000, 1500
rng = np.random.default_rng(seed); pos, cls = gen_traj(N, T, rng)
lam_u = lam * rng.uniform(0.5, 1.5, size=N); calls = rng.random((N, T)) < lam_u[:, None]
prof = Profile2(N); p0 = pos[:, 0]; DA = (2, 3, 4, 5, 6, 7, 8); th = (0.5, 0.8, 0.95)
sch = [DBR(N, T, p0, 5, True, name="DBR-5"), DBR(N, T, p0, 3, True, name="DBR-3"), DBR(N, T, p0, 8, True, name="DBR-8"),
       UPALH(N, T, p0, prof, bandit=False, fixedD=5, name="H fixed D5", seed=seed, hmax=40, thetas=th),
       UPALH(N, T, p0, prof, bandit=True, name="H bandit", seed=seed, hmax=40, thetas=th, arms=DA),
       UPALH(N, T, p0, prof, bandit=True, direction=True, name="H bandit+dir", seed=seed, hmax=40, thetas=th, arms=DA)]
for t in range(1, T):
    prev = pos[:, t-1]; cur = pos[:, t]; call = calls[:, t]; prof.update(prev, cur, call.astype(float))
    for s in sch: s.step(t, cur, prev, call, t >= W, prof)
steps = T - W
for s in sch:
    a = s.acc; cost = (10 * s.kappa * a.regs + a.cells) / steps * 1000
    print(f"lam={lam} {s.name:14s} cost={cost.mean():7.1f} ped={cost[cls==0].mean():6.0f} veh={cost[cls==1].mean():6.0f} com={cost[cls==2].mean():6.0f} regs={a.regs.mean()/steps*1000:6.1f} cells/call={a.cells.sum()/a.calls.sum():5.1f} delay={a.rounds.sum()/a.calls.sum():4.2f}", flush=True)
