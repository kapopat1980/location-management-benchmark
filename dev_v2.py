# Development experiments for UPAL v2 on DEVELOPMENT seeds (101, 102) -- disjoint from evaluation seeds 1-5.
import sys, numpy as np
from sim import *
def build(N, T, pos0, prof, seed):
    sc = [DBR(N, T, pos0, 5, True, name="DBR-5"), DBR(N, T, pos0, 7, True, name="DBR-7")]
    sc.append(UPAL(N, T, pos0, prof, bandit=True, selective=True, name="v1 (H15,2rd)", seed=seed))
    sc.append(UPAL(N, T, pos0, prof, bandit=True, selective=True, name="H40,2rd", seed=seed, hmax=40))
    sc.append(UPAL(N, T, pos0, prof, bandit=True, selective=True, name="H15,4rd", seed=seed, thetas=(0.5, 0.8, 0.95)))
    sc.append(UPAL(N, T, pos0, prof, bandit=True, selective=True, name="H40,4rd", seed=seed, hmax=40, thetas=(0.5, 0.8, 0.95)))
    sc.append(UPAL(N, T, pos0, prof, bandit=True, selective=True, name="H80,4rd", seed=seed, hmax=80, thetas=(0.5, 0.8, 0.95)))
    sc.append(UPAL(N, T, pos0, prof, bandit=True, selective=True, name="H80,6rd", seed=seed, hmax=80, thetas=(0.3, 0.5, 0.7, 0.85, 0.95)))
    return sc
lam = float(sys.argv[1]); seed = int(sys.argv[2])
N, T, W = 60, 4000, 1500
rng = np.random.default_rng(seed); pos, cls = gen_traj(N, T, rng)
lam_u = lam * rng.uniform(0.5, 1.5, size=N); calls = rng.random((N, T)) < lam_u[:, None]
prof = Profile(N); sch = build(N, T, pos[:, 0], prof, seed)
for t in range(1, T):
    prev = pos[:, t-1]; cur = pos[:, t]; call = calls[:, t]; prof.update(prev, cur, call.astype(float))
    for s in sch: s.step(t, cur, prev, call, t >= W, prof)
steps = T - W
for s in sch:
    a = s.acc; cost = (10 * s.kappa * a.regs + a.cells) / steps * 1000
    print(f"lam={lam} {s.name:16s} cost={cost.mean():7.1f} regs={a.regs.mean()/steps*1000:6.1f} cells/call={a.cells.sum()/a.calls.sum():6.1f} delay={a.rounds.sum()/a.calls.sum():4.2f}", flush=True)
