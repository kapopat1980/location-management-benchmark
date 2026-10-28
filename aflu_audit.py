"""Audit of the AFLU simulator (github.com/kapopat1980/aflu-repo) for comparison fairness.
Uses the repo's own mobility, call and baseline code unchanged; adds (a) an AFLU variant whose paging
histogram excludes the user's current cell (no look-ahead), and (b) a distance-based baseline with
ring-by-ring (shortest-distance-first) paging."""
import sys, json, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
import lu_sim as L
from lu_sim import *

def run_afle_noleak(traj, moved, calls, rng, eta=0.10, decay=0.02, Tn=8.0, Dn=3.0, Mn=3.0, theta0=1.0):
    n_ticks_p1, n_users, _ = traj.shape
    theta = np.full(n_users, theta0); last_update_tick = np.zeros(n_users, dtype=int)
    last_pos = traj[0].copy(); move_count = np.zeros(n_users, dtype=int)
    hist = np.zeros((n_users, GRID, GRID)); interval_pg_cost = np.zeros(n_users)
    lu_events = 0; total_pg_cost = 0.0; TH_LO, TH_HI = 0.25, 3.0
    ar = np.arange(n_users)
    for t in range(1, n_ticks_p1):
        move_count += moved[t].astype(int)
        hist_prev = hist.copy() if calls[t].any() else hist          # histogram as known BEFORE locating the user
        hist *= (1 - decay); hist[ar, traj[t][:, 0], traj[t][:, 1]] += decay
        elapsed = t - last_update_tick; dist = grid_dist(traj[t], last_pos)
        S = elapsed / Tn + dist / Dn + move_count / Mn; do_update = S >= theta
        callers = np.where(calls[t])[0]
        if len(callers):
            radius = np.maximum(1, np.maximum(dist[callers], move_count[callers]))
            cost_now = np.zeros(len(callers))
            for k, u in enumerate(callers):
                r = int(radius[k]); lp = last_pos[u]
                x0, x1 = max(0, lp[0] - r), min(GRID, lp[0] + r + 1); y0, y1 = max(0, lp[1] - r), min(GRID, lp[1] + r + 1)
                sub = hist_prev[u, x0:x1, y0:y1]; tc = traj[t][u]
                tp = sub[tc[0] - x0, tc[1] - y0]; cost_now[k] = max(int((sub >= tp).sum()), 1)
            total_pg_cost += cost_now.sum(); interval_pg_cost[callers] += cost_now
        if do_update.any():
            idx = np.where(do_update)[0]; lu_events += len(idx)
            ratio = interval_pg_cost[idx] / LU_COST
            step = np.exp(-eta * np.clip(np.log(np.maximum(ratio, 1e-3)), -2, 2))
            theta[idx] = np.clip(theta[idx] * step, TH_LO, TH_HI)
            last_update_tick[idx] = t; last_pos[idx] = traj[t][idx]; move_count[idx] = 0; interval_pg_cost[idx] = 0.0
    return lu_events, total_pg_cost

def run_distance_sdf(traj, moved, calls, D):
    n_ticks_p1, n_users, _ = traj.shape
    last_pos = traj[0].copy(); lu = 0; pg = 0.0
    for t in range(1, n_ticks_p1):
        d = grid_dist(traj[t], last_pos); upd = d >= D
        if upd.any(): lu += int(upd.sum()); last_pos[upd] = traj[t][upd]
        c = np.where(calls[t])[0]
        if len(c):
            dd = grid_dist(traj[t][c], last_pos[c]); pg += np.minimum(2 * dd * dd + 2 * dd + 1, GRID * GRID).sum()   # all cells within L1 ring dd (SDF), grid clipping ignored
    return lu, pg

def best(fn, cands, traj, moved, calls):
    b = None
    for th in cands:
        lu, pg = fn(traj, moved, calls, th); tot = LU_COST * lu + pg
        if b is None or tot < b[0]: b = (tot, th)
    return b

N, T, seed = 150, 2200, 7
rng = np.random.default_rng(seed)
traj, moved, profiles, home = simulate_mobility(N, T, rng)
pm = np.array([PROFILE_PARAMS[p]["p_move"] for p in profiles])
den = N * (T - BURN_IN); out = {}
for cmr in [0.25, 1, 4]:
    calls = simulate_calls(N, T, cmr, pm, rng); calls[:BURN_IN] = False
    bd = best(run_distance_based, [1,2,3,4,5,6,7], traj, moved, calls)
    bm = best(run_movement_based, [1,2,3,4,5,6,8], traj, moved, calls)
    bp = best(run_profile_based, [0.02,0.04,0.06,0.08,0.10,0.15,0.20], traj, moved, calls)
    bs = best(run_distance_sdf, [1,2,3,4,5,6,7,8,10,12], traj, moved, calls)
    lu_a, pg_a, _ = run_afle_adaptive(traj, moved, calls, rng); a_leak = LU_COST * lu_a + pg_a
    lu_n, pg_n = run_afle_noleak(traj, moved, calls, rng); a_fix = LU_COST * lu_n + pg_n
    out[cmr] = dict(dbr_blanket=bd, mb_blanket=bm, profile=bp, dbr_sdf=bs, aflu_original=a_leak, aflu_noleak=a_fix)
    best_classic = min(bd[0], bm[0], bp[0])
    print(f"CMR={cmr}: per-user-tick cost  best-classic(blanket)={best_classic/den:.3f}  DBR+SDF*={bs[0]/den:.3f} (D={bs[1]})  AFLU(original)={a_leak/den:.3f}  AFLU(no look-ahead)={a_fix/den:.3f}  | AFLU orig vs classic {100*(a_leak/best_classic-1):+.1f}%  AFLU fixed vs classic {100*(a_fix/best_classic-1):+.1f}%  AFLU fixed vs DBR+SDF {100*(a_fix/bs[0]-1):+.1f}%", flush=True)
json.dump({str(k): v for k, v in out.items()}, open("results/aflu_audit.json", "w"), indent=1)
