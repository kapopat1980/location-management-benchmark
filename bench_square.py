"""Benchmark B: fair re-evaluation on the AFLU square-grid simulator (github.com/kapopat1980/aflu-repo, unchanged
mobility, call and classical-baseline code).  Adds ring-by-ring (shortest-distance-first, SDF) paging for every
trigger, a leak-free profile-based baseline, a leak-free AFLU, and AFLU's self-tuning trigger with SDF paging."""
import sys, json, time, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
import lu_sim as L
from lu_sim import *

G = GRID
# number of grid cells within L1 distance <= d of (x, y)
xs, ys = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
DMAX = 2 * (G - 1)
DCNT = np.zeros((G, G, DMAX + 2), dtype=np.int64)
for d in range(DMAX + 2):
    for x in range(G):
        for y in range(G):
            DCNT[x, y, d] = (np.abs(xs - x) + np.abs(ys - y) <= d).sum()

def sdf_cost(last_pos, true_pos):
    d = np.minimum(grid_dist(true_pos, last_pos), DMAX + 1)
    return DCNT[last_pos[:, 0], last_pos[:, 1], d].astype(float)

def run_trigger_sdf(traj, moved, calls, kind, th):
    T1, N, _ = traj.shape; last_pos = traj[0].copy(); mc = np.zeros(N, dtype=int); lt = np.zeros(N, dtype=int); lu = 0; pg = 0.0
    for t in range(1, T1):
        mc += moved[t].astype(int)
        if kind == "dist": upd = grid_dist(traj[t], last_pos) >= th
        elif kind == "move": upd = mc >= th
        else: upd = (t - lt) >= th
        if upd.any():
            lu += int(upd.sum()); last_pos[upd] = traj[t][upd]; mc[upd] = 0; lt[upd] = t
        c = np.where(calls[t])[0]
        if len(c): pg += sdf_cost(last_pos[c], traj[t][c]).sum()
    return lu, pg

def run_profile_noleak(traj, moved, calls, p_thresh, decay=0.01):
    T1, N, _ = traj.shape; hist = np.zeros((N, G, G)); lu = 0; pg = 0.0; ar = np.arange(N)
    for t in range(1, T1):
        c = np.where(calls[t])[0]
        if len(c):                                      # network profile BEFORE the current cell is known
            hp = hist[c]; tot = hp.reshape(len(c), -1).sum(1); tot[tot == 0] = 1
            probs = hp / tot[:, None, None]; tc = traj[t][c]
            tp = probs[np.arange(len(c)), tc[:, 0], tc[:, 1]]
            pg += (probs >= tp[:, None, None]).reshape(len(c), -1).sum(1).sum()
        hist *= (1 - decay); hist[ar, traj[t][:, 0], traj[t][:, 1]] += decay
        totals = hist.reshape(N, -1).sum(1); totals[totals == 0] = 1
        cur = hist[ar, traj[t][:, 0], traj[t][:, 1]] / totals
        upd = cur < p_thresh; lu += int(upd.sum())
    return lu, pg

def run_aflu(traj, moved, calls, paging, eta=0.10, decay=0.02, Tn=8.0, Dn=3.0, Mn=3.0, theta0=1.0):
    """paging: 'noleak' (rank in prior histogram, ring-restricted) or 'sdf' (ring-by-ring, no profile)."""
    T1, N, _ = traj.shape; theta = np.full(N, theta0); lut = np.zeros(N, dtype=int); last_pos = traj[0].copy()
    mc = np.zeros(N, dtype=int); hist = np.zeros((N, G, G)); ipc = np.zeros(N); lu = 0; pg = 0.0; ar = np.arange(N)
    for t in range(1, T1):
        mc += moved[t].astype(int)
        hp = hist.copy() if (paging == "noleak" and calls[t].any()) else None
        hist *= (1 - decay); hist[ar, traj[t][:, 0], traj[t][:, 1]] += decay
        el = t - lut; dist = grid_dist(traj[t], last_pos); S = el / Tn + dist / Dn + mc / Mn; upd = S >= theta
        c = np.where(calls[t])[0]
        if len(c):
            if paging == "sdf": cost = sdf_cost(last_pos[c], traj[t][c])
            else:
                r = np.maximum(1, np.maximum(dist[c], mc[c])); cost = np.zeros(len(c))
                for k, u in enumerate(c):
                    rr = int(r[k]); lp = last_pos[u]
                    x0, x1 = max(0, lp[0]-rr), min(G, lp[0]+rr+1); y0, y1 = max(0, lp[1]-rr), min(G, lp[1]+rr+1)
                    sub = hp[u, x0:x1, y0:y1]; tc = traj[t][u]; tp = sub[tc[0]-x0, tc[1]-y0]
                    cost[k] = max(int((sub >= tp).sum()), 1)
            pg += cost.sum(); ipc[c] += cost
        if upd.any():
            idx = np.where(upd)[0]; lu += len(idx); ratio = ipc[idx] / LU_COST
            step = np.exp(-eta * np.clip(np.log(np.maximum(ratio, 1e-3)), -2, 2))
            theta[idx] = np.clip(theta[idx] * step, 0.25, 3.0)
            lut[idx] = t; last_pos[idx] = traj[t][idx]; mc[idx] = 0; ipc[idx] = 0.0
    return lu, pg


def run_str(traj, moved, calls, seed, arms=(1, 2, 3, 4, 5, 6, 7, 8), gamma=0.9995, eps=0.08, d0=3):
    """Self-tuning radius (STR): distance-based registration with ring-by-ring paging whose radius D is chosen by a
    discounted, context-pooled cost bandit (same design as on Benchmark A).  Because calls do not re-anchor the last
    reported cell in this simulator, a decision interval ends at a registration or at a call and the radius is
    re-selected at both (at a call the network can reconfigure the threshold with the paging signalling at no extra
    cost); otherwise a large explored radius would lock a user in for the rest of the run.  An interval costs the
    registration that started it (U) plus the paging cost of the call that ended it.  The bandit learns only after the
    burn-in, during which D = d0.  Registration when the distance from the last reported cell reaches D."""
    T1, N, _ = traj.shape; rng = np.random.default_rng(seed + 777)
    last_pos = traj[0].copy(); Du = np.full(N, d0); m_hat = np.full(N, 0.2)
    seg_t0 = np.zeros(N, dtype=int); seg_cost = np.zeros(N); seg_arm = np.full(N, -1); seg_ctx = np.zeros(N, dtype=int)
    Sc = np.zeros((6, len(arms))); Sd = np.zeros((6, len(arms))); nn = np.zeros((6, len(arms))); lu = 0; pg = 0.0
    def ctx_of(u, t):
        b = 0 if m_hat[u] < 0.15 else (1 if m_hat[u] < 0.5 else 2)
        return b * 2 + (1 if t >= 300 else 0)
    def choose(x):
        low = np.where(nn[x] < 2)[0]
        if len(low): return int(low[0])
        if rng.random() < eps: return int(rng.integers(len(arms)))
        return int(np.argmin(Sc[x] / np.maximum(Sd[x], 1e-9)))
    def close_and_reselect(u, t):
        if t < BURN_IN: return
        if seg_arm[u] >= 0:
            x, a = seg_ctx[u], seg_arm[u]
            Sc[x, a] = gamma * Sc[x, a] + seg_cost[u]; Sd[x, a] = gamma * Sd[x, a] + max(1, t - seg_t0[u]); nn[x, a] += 1
        x = ctx_of(u, t); a = choose(x); Du[u] = arms[a]; seg_arm[u] = a; seg_ctx[u] = x
    for t in range(1, T1):
        m_hat += (moved[t].astype(float) - m_hat) / 400.0
        for u in np.where(grid_dist(traj[t], last_pos) >= Du)[0]:
            close_and_reselect(u, t)
            last_pos[u] = traj[t][u]; seg_t0[u] = t; seg_cost[u] = LU_COST; lu += 1
        c = np.where(calls[t])[0]
        if len(c):
            cost = sdf_cost(last_pos[c], traj[t][c]); pg += cost.sum()
            for k, u in enumerate(c):
                seg_cost[u] += cost[k]
                close_and_reselect(u, t)
                seg_t0[u] = t; seg_cost[u] = 0.0
    return lu, pg


def tot(r): return LU_COST * r[0] + r[1]
def best(fn, cands, *a):
    out = [(tot(fn(*a, th)), th) for th in cands]; return min(out)

if __name__ == "__main__":
    N, T = 220, 1600
    out = {}; t0 = time.time()
    for seed in [1, 2, 3]:
        rng = np.random.default_rng(seed)
        traj, moved, profiles, home = simulate_mobility(N, T, rng)
        pm = np.array([PROFILE_PARAMS[p]["p_move"] for p in profiles])
        for cmr in [0.25, 0.5, 1, 2, 4, 8, 16]:
            calls = simulate_calls(N, T, cmr, pm, rng); calls[:BURN_IN] = False
            den = N * T; r = {}
            r["time_blanket"] = best(run_time_based, [3,5,8,12,16,20,28,36], traj, moved, calls)
            r["dist_blanket"] = best(run_distance_based, [1,2,3,4,5,6,7], traj, moved, calls)
            r["move_blanket"] = best(run_movement_based, [1,2,3,4,5,6,8], traj, moved, calls)
            r["profile_leaky"] = best(run_profile_based, [0.02,0.04,0.06,0.08,0.10,0.15,0.20], traj, moved, calls)
            r["time_sdf"] = best(lambda *a: run_trigger_sdf(*a[:3], "time", a[3]), [3,5,8,12,16,20,28,36], traj, moved, calls)
            r["dist_sdf"] = best(lambda *a: run_trigger_sdf(*a[:3], "dist", a[3]), [1,2,3,4,5,6,7,8,10,12], traj, moved, calls)
            r["move_sdf"] = best(lambda *a: run_trigger_sdf(*a[:3], "move", a[3]), [1,2,3,4,5,6,8,10,12], traj, moved, calls)
            r["profile_noleak"] = best(run_profile_noleak, [0.02,0.04,0.06,0.08,0.10,0.15,0.20], traj, moved, calls)
            lu, pg, _ = run_afle_adaptive(traj, moved, calls, rng); r["aflu_original"] = (LU_COST * lu + pg, None)
            r["aflu_noleak"] = (tot(run_aflu(traj, moved, calls, "noleak")), None)
            r["aflu_trigger_sdf"] = (tot(run_aflu(traj, moved, calls, "sdf")), None)
            r["str"] = (tot(run_str(traj, moved, calls, seed)), None)
            out[f"{seed}_{cmr}"] = {k: [v[0] / den, v[1]] for k, v in r.items()}
            print(seed, cmr, {k: round(v[0] / den, 3) for k, v in r.items()}, round(time.time() - t0), "s", flush=True)
            json.dump(out, open("results/bench_square.json", "w"), indent=1)
