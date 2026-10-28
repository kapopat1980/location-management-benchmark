"""Evaluate the schemes on the GeoLife cell sequences (real trajectories, simulated calls).
usage: python3 run_real.py lam0 seed out.json [max_users] [cap_slots]"""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim
from sim import *

class AccR:
    """Accounting with a per-user, per-slot measurement mask (first W active slots of every user are warm-up)."""
    def __init__(s, N, T):
        s.regs = np.zeros(N); s.cells = np.zeros(N); s.rounds = np.zeros(N); s.calls = np.zeros(N); s.cur_meas = None
        s.reg_ts = np.zeros(T); s.pg_ts = np.zeros(T)
    def reg(s, mask, t, meas):
        s.regs[mask & s.cur_meas] += 1
    def page(s, mask, cells_, rounds_, t, meas):
        m = mask & s.cur_meas
        ce = np.broadcast_to(np.asarray(cells_, dtype=float), mask.shape); ro = np.broadcast_to(np.asarray(rounds_, dtype=float), mask.shape)
        s.calls[m] += 1; s.cells[m] += ce[m]; s.rounds[m] += ro[m]
sim.Acc = AccR

def load(path="results/geolife_cells.npz", min_slots=2000, cap=15000, W=1000, max_users=None):
    z = np.load(path); users = [u for u in z["users"] if len(z["c_" + u]) >= min_slots]
    users = sorted(users, key=lambda u: -len(z["c_" + u]))[:max_users] if max_users else sorted(users)
    N = len(users); T = min(cap, max(len(z["c_" + u]) for u in users))
    pos = np.zeros((N, T), dtype=np.int64); active = np.zeros((N, T), dtype=bool); reset = np.zeros((N, T), dtype=bool)
    for i, u in enumerate(users):
        c = z["c_" + u].astype(np.int64)[:T]; r = z["r_" + u][:T]; n = len(c)
        pos[i, :n] = c; pos[i, n:] = c[-1]; active[i, :n] = True; reset[i, :n] = r
    meas = active & (np.cumsum(active, axis=1) > W)
    return users, pos, active, reset, meas

def reanchor(sc, mask, cur, t):
    nm = type(sc).__name__; idxs = np.where(mask)[0]
    if nm == "TA9": sc.reg[mask] = ta_id[cur[mask]]
    elif nm == "MB": sc.anchor[mask] = cur[mask]; sc.cnt[mask] = 0
    elif nm == "DBR": sc.anchor[mask] = cur[mask]
    elif nm == "DirTAL":
        for u in idxs: sc.prev[u] = cur[u]; sc.anchor[u] = cur[u]; sc._build(u)
    elif nm in ("UPAL", "DBRBandit"):
        for u in idxs:
            sc._end_segment(u, t, 0.0)          # a segment cut by the end of a recording session is still a valid (censored) cost/duration sample
            sc.anchor[u] = cur[u]; sc._build(u, t, explicit=False)
    else: raise ValueError(nm)

def build(N, T, p0, prof, seed):
    sch = [TA9(N, T, p0), MB(N, T, p0, 3), DBR(N, T, p0, 2, True), DBR(N, T, p0, 1, False, name="Ring TAL (7 cells)"), DirTAL(N, T, p0),
           UPAL(N, T, p0, prof, bandit=True, selective=True, name="UPAL (full)", seed=seed)]
    sch += [DBR(N, T, p0, d, True, name=f"DBR sweep D={d}") for d in range(1, 9)]
    sch += [MB(N, T, p0, m) for m in range(2, 9)]
    sch.append(DBRBandit(N, T, p0, prof, bandit=True, selective=True, kappa=1.0, name="STR", seed=seed, arms=tuple(range(1, 9))))
    return sch

def run(lam0, seed, max_users=None, cap=15000, W=1000):
    users, pos, active, reset, meas = load(max_users=max_users, cap=cap, W=W)
    N, T = pos.shape; rng = np.random.default_rng(seed)
    lam_u = lam0 * rng.uniform(0.5, 1.5, size=N); calls = (rng.random((N, T)) < lam_u[:, None]) & active
    prof = Profile(N); sch = build(N, T, pos[:, 0], prof, seed); t0 = time.time()
    for t in range(1, T):
        cur = pos[:, t]; rs = reset[:, t]; prev = np.where(rs, cur, pos[:, t - 1]); call = calls[:, t]
        prof.update(prev, cur, call.astype(float))
        for sc in sch:
            sc.acc.cur_meas = meas[:, t]
            if rs.any(): reanchor(sc, rs, cur, t)
            sc.step(t, cur, prev, call, True, prof)
        if t % 3000 == 0: print("  t", t, round(time.time() - t0), "s", flush=True)
    out = {"lam0": lam0, "seed": seed, "N": N, "T": T, "W": W, "users": list(users), "meas_slots": meas.sum(1).tolist(), "schemes": {}}
    for sc in sch:
        a = sc.acc; out["schemes"][sc.name] = {"kappa": sc.kappa, "regs": a.regs.tolist(), "cells": a.cells.tolist(), "rounds": a.rounds.tolist(), "calls": a.calls.tolist()}
    out["runtime_s"] = time.time() - t0
    return out

if __name__ == "__main__":
    lam0 = float(sys.argv[1]); seed = int(sys.argv[2]); fn = sys.argv[3]
    mu = int(sys.argv[4]) if len(sys.argv) > 4 else None; cap = int(sys.argv[5]) if len(sys.argv) > 5 else 15000
    r = run(lam0, seed, mu, cap); json.dump(r, open(fn, "w")); print("done", fn, round(r["runtime_s"]), "s")
