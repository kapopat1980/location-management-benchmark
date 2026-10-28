"""
Dynamic location management simulator (hexagonal cellular grid).
Compares classical schemes with the proposed UPAL scheme on identical
trajectories and call arrivals (common random numbers).
"""
import numpy as np, json, sys, time

# ------------------------------------------------------------------ grid
R = 13
cells = [(q, r) for q in range(-R, R + 1) for r in range(-R, R + 1) if abs(q + r) <= R]
idx = {c: i for i, c in enumerate(cells)}
C = len(cells)
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
nbr = np.zeros((C, 6), dtype=np.int64)
for i, (q, r) in enumerate(cells):
    for d, (dq, dr) in enumerate(DIRS):
        nbr[i, d] = idx.get((q + dq, r + dr), i)          # reflecting boundary
QQ = np.array([c[0] for c in cells]); RR = np.array([c[1] for c in cells])
dq = QQ[:, None] - QQ[None, :]; dr = RR[:, None] - RR[None, :]
D = ((np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) // 2).astype(np.int64)
XY = np.stack([QQ + RR / 2.0, RR * np.sqrt(3) / 2.0], axis=1)
within = np.stack([(D <= d).sum(1) for d in range(0, 10)], axis=1)  # cells within ring d
nbr7 = np.concatenate([np.arange(C)[:, None], nbr], axis=1)          # self + 6 nbrs
ta_key = (QQ + R) // 3 * 100 + (RR + R) // 3
_, ta_id = np.unique(ta_key, return_inverse=True)
ta_size = np.bincount(ta_id)
CENTER = idx[(0, 0)]

# ------------------------------------------------------------ mobility
def gen_traj(N, T, rng):
    n_ped = int(0.4 * N); n_veh = int(0.3 * N); n_com = N - n_ped - n_veh
    cls = np.array([0] * n_ped + [1] * n_veh + [2] * n_com)
    pos = np.zeros((N, T), dtype=np.int64)
    disc = np.where(D[CENTER] <= 11)[0]
    cur = rng.choice(disc, size=N)
    home = np.zeros(N, dtype=np.int64); work = np.zeros(N, dtype=np.int64)
    for u in np.where(cls == 2)[0]:
        h = rng.choice(np.where(D[CENTER] <= 6)[0])
        cand = np.where((D[h] >= 9) & (D[h] <= 13))[0]
        home[u] = h; work[u] = rng.choice(cand); cur[u] = h
    head = rng.integers(0, 6, size=N)
    P = 500
    off = rng.integers(0, P, size=N)
    jit = rng.integers(-10, 11, size=(N, T // P + 3))
    ar = np.arange(N)
    com = cls == 2; veh = cls == 1; ped = cls == 0
    for t in range(T):
        pos[:, t] = cur
        r = rng.random(N)
        nxt = cur.copy()
        # pedestrians: lazy random walk
        d = rng.integers(0, 6, size=N)
        m = ped & (r < 0.2)
        nxt[m] = nbr[cur[m], d[m]]
        # vehicles: persistent heading
        chg = veh & (rng.random(N) < 0.15)
        head[chg] = (head[chg] + rng.choice([-1, 1], size=chg.sum())) % 6
        mv = veh & (r < 0.95)
        tgt = nbr[cur, head]
        blocked = mv & (tgt == cur)
        head[blocked] = (head[blocked] + 3) % 6
        tgt = nbr[cur, head]
        nxt[mv] = tgt[mv]
        # commuters: goal-directed daily routine
        phase = (t + off + jit[ar, t // P]) % P
        target = np.where((phase >= 220) & (phase < 470), work, home)
        atg = cur == target
        nd = D[nbr[cur], target[:, None]]
        best = np.argmin(nd + rng.random((N, 6)) * 0.5, axis=1)
        gt = nbr[cur, best]
        rn = nbr[cur, rng.integers(0, 6, size=N)]
        u2 = rng.random(N)
        sc = np.where(atg, np.where(u2 < 0.05, rn, cur),
                      np.where(u2 < 0.9, gt, np.where(u2 < 0.95, rn, cur)))
        nxt[com] = sc[com]
        cur = nxt
    return pos, cls

# ------------------------------------------------------------ accounting
class Acc:
    def __init__(s, N, T):
        s.regs = np.zeros(N); s.cells = np.zeros(N); s.rounds = np.zeros(N); s.calls = np.zeros(N)
        s.reg_ts = np.zeros(T); s.pg_ts = np.zeros(T)
    def reg(s, mask, t, meas):
        n = int(mask.sum())
        if n:
            s.reg_ts[t] += n
            if meas: s.regs[mask] += 1
    def page(s, mask, cells_, rounds_, t, meas):
        s.pg_ts[t] += float(np.asarray(cells_)[mask].sum()) if np.ndim(cells_) else float(cells_) * mask.sum()
        if meas:
            s.calls[mask] += 1
            s.cells[mask] += np.asarray(cells_)[mask] if np.ndim(cells_) else cells_
            s.rounds[mask] += np.asarray(rounds_)[mask] if np.ndim(rounds_) else rounds_

# ------------------------------------------------------------ baselines
class TA9:
    name = "TA-9 (static)"; kappa = 1.0
    def __init__(s, N, T, pos0): s.reg = ta_id[pos0]; s.acc = Acc(N, T)
    def step(s, t, pos, prev, call, meas, prof):
        cur = ta_id[pos]; ch = cur != s.reg
        s.acc.reg(ch, t, meas); s.reg = np.where(ch, cur, s.reg)
        if call.any(): s.acc.page(call, ta_size[s.reg].astype(float), 1, t, meas)

class MB:
    kappa = 1.0
    def __init__(s, N, T, pos0, M=3):
        s.M = M; s.name = f"Movement-based (M={M}, SDF)"
        s.cnt = np.zeros(N, dtype=np.int64); s.anchor = pos0.copy(); s.acc = Acc(N, T)
    def step(s, t, pos, prev, call, meas, prof):
        s.cnt += (pos != prev); trig = s.cnt >= s.M
        s.acc.reg(trig, t, meas); s.anchor[trig] = pos[trig]; s.cnt[trig] = 0
        if call.any():
            d = D[s.anchor, pos]
            s.acc.page(call, within[s.anchor, np.minimum(d, 9)].astype(float), d + 1, t, meas)
            s.anchor[call] = pos[call]; s.cnt[call] = 0

class DBR:
    kappa = 1.0
    def __init__(s, N, T, pos0, Dth=2, sdf=True, name=None):
        s.Dth = Dth; s.sdf = sdf
        s.name = name or f"Distance-based (D={Dth}{', SDF' if sdf else ''})"
        s.anchor = pos0.copy(); s.acc = Acc(N, T)
    def step(s, t, pos, prev, call, meas, prof):
        trig = D[s.anchor, pos] > s.Dth
        s.acc.reg(trig, t, meas); s.anchor[trig] = pos[trig]
        if call.any():
            d = D[s.anchor, pos]
            if s.sdf: s.acc.page(call, within[s.anchor, np.minimum(d, 9)].astype(float), d + 1, t, meas)
            else: s.acc.page(call, within[s.anchor, s.Dth].astype(float), 1, t, meas)
            s.anchor[call] = pos[call]

class DirTAL:
    """Direction-sector flexible TAL (re-implementation of the concept of mobility-prediction TALs)."""
    name = "Direction-sector TAL"; kappa = 1.0
    def __init__(s, N, T, pos0):
        s.N = N; s.anchor = pos0.copy(); s.prev = pos0.copy(); s.acc = Acc(N, T)
        s.inT = np.zeros((N, C), dtype=bool); s.size = np.zeros(N)
        for u in range(N): s._build(u)
    def _build(s, u):
        a = s.anchor[u]; p = s.prev[u]
        cand = np.where(D[a] <= 2)[0]
        keep = D[a, cand] <= 1
        if p != a:
            v = XY[a] - XY[p]; nv = np.linalg.norm(v)
            w = XY[cand] - XY[a]; nw = np.linalg.norm(w, axis=1); nw[nw == 0] = 1
            keep = keep | ((w @ v) / (nv * nw) >= 0.5)
        A = cand[keep]
        s.inT[u, :] = False; s.inT[u, A] = True; s.size[u] = len(A)
    def step(s, t, pos, prev, call, meas, prof):
        out = ~s.inT[np.arange(s.N), pos]
        s.acc.reg(out, t, meas)
        for u in np.where(out)[0]:
            s.prev[u] = s.anchor[u]; s.anchor[u] = pos[u]; s._build(u)
        if call.any():
            s.acc.page(call, s.size.copy(), 1, t, meas)
            for u in np.where(call)[0]:
                s.prev[u] = s.anchor[u]; s.anchor[u] = pos[u]; s._build(u)

# ------------------------------------------------- learning profile (UE side)
class Profile:
    ALPHA = 4.0
    def __init__(s, N):
        s.N = N; s.cnt = np.zeros((N, C, 7), dtype=np.float32)
        s.m_hat = np.full(N, 0.2); s.lam_hat = np.full(N, 0.02); s.ntot = np.zeros(N)
        s.ar = np.arange(N)
    def update(s, prev, pos, call):
        moved = pos != prev
        match = (nbr[prev] == pos[:, None])
        di = np.where(moved, match.argmax(1) + 1, 0)
        np.add.at(s.cnt, (s.ar, prev, di), 1.0)
        s.m_hat += (moved - s.m_hat) / 400.0
        s.lam_hat += (call - s.lam_hat) / 600.0
        s.ntot += 1
    def rows(s, u):
        c = s.cnt[u].astype(np.float64); m = min(max(s.m_hat[u], 0.02), 0.98)
        prior = np.array([1 - m] + [m / 6.0] * 6)
        n = c.sum(1, keepdims=True)
        return (c + s.ALPHA * prior) / (n + s.ALPHA)

# ------------------------------------------------------------ proposed UPAL
ARMS = np.array([0.50, 0.70, 0.85, 0.95, 0.99])
HMAX = 15

class UPAL:
    """Uncertainty-aware Predictive Adaptive TAL.
       bandit: tune coverage level online;  selective: two-round probability-ordered paging."""
    def __init__(s, N, T, pos0, prof, bandit=True, selective=True, rho=0.90, kappa=1.2, name="UPAL", seed=0, hmax=15, thetas=(0.85,), arms=None):
        s.N = N; s.prof = prof; s.bandit = bandit; s.selective = selective; s.rho = rho; s.hmax = hmax; s.thetas = tuple(thetas); s.arms = ARMS if arms is None else np.asarray(arms, dtype=float)
        s.kappa = kappa; s.name = name; s.acc = Acc(N, T)
        s.rng = np.random.default_rng(seed + 12345)
        s.inT = np.zeros((N, C), dtype=bool)
        s.anchor = pos0.copy(); s.last_t = np.zeros(N, dtype=np.int64)
        s.A = [None] * N; s.Pk = [None] * N; s.mvec = [None] * N; s.H = np.ones(N, dtype=np.int64)
        s.seg_t0 = np.zeros(N, dtype=np.int64); s.seg_cost = np.zeros(N)
        s.seg_arm = np.zeros(N, dtype=np.int64); s.seg_ctx = np.zeros(N, dtype=np.int64)
        s.Sc = np.zeros((6, len(s.arms))); s.Sd = np.zeros((6, len(s.arms))); s.nn = np.zeros((6, len(s.arms)))
        s.eps = 0.08; s.gamma = 0.9995
        s.sizes_at_call = []
        for u in range(N): s._build(u, 0, explicit=True, first=True)
    def _ctx(s, u):
        m = s.prof.m_hat[u]; b = 0 if m < 0.15 else (1 if m < 0.5 else 2)
        return b * 2 + (1 if s.prof.ntot[u] >= 300 else 0)
    def _choose(s, ctx):
        if not s.bandit: return None
        low = np.where(s.nn[ctx] < 2)[0]
        if len(low): return int(low[0])
        if s.rng.random() < s.eps: return int(s.rng.integers(len(s.arms)))
        return int(np.argmin(s.Sc[ctx] / np.maximum(s.Sd[ctx], 1e-9)))
    def _build(s, u, t, explicit, first=False):
        prof = s.prof; start = s.anchor[u]
        H = int(np.clip(round(1.0 / max(prof.lam_hat[u], 1e-3)), 3, s.hmax))
        P = prof.rows(u); p = np.zeros(C); p[start] = 1.0
        Pk = np.zeros((H, C)); m = np.zeros(C)
        for k in range(H):
            p = np.bincount(nbr7.ravel(), weights=(p[:, None] * P).ravel(), minlength=C)
            Pk[k] = p; m += p
        m[start] += 1e-6
        ctx = s._ctx(u); arm = s._choose(ctx)
        rho = s.rho if arm is None else s.arms[arm]
        order = np.argsort(-m); cs = np.cumsum(m[order]) / m.sum()
        n = int(np.searchsorted(cs, rho)) + 1
        A = order[:n]
        if start not in A: A = np.append(A, start)
        s.inT[u, :] = False; s.inT[u, A] = True
        s.A[u] = A; s.Pk[u] = Pk.astype(np.float32); s.mvec[u] = m; s.H[u] = H
        s.last_t[u] = t; s.seg_t0[u] = t
        s.seg_cost[u] = s.kappa * 10.0 if explicit else 0.0
        s.seg_arm[u] = -1 if arm is None else arm; s.seg_ctx[u] = ctx
    def _end_segment(s, u, t, extra):
        if not s.bandit or s.seg_arm[u] < 0: return
        c = s.seg_cost[u] + extra; d = max(1, t - s.seg_t0[u]); x, a = s.seg_ctx[u], s.seg_arm[u]
        s.Sc[x, a] = s.gamma * s.Sc[x, a] + c; s.Sd[x, a] = s.gamma * s.Sd[x, a] + d; s.nn[x, a] += 1
    def _page(s, u, posu, t):
        A = s.A[u]
        if not s.selective: return float(len(A)), 1
        tau = t - s.last_t[u]; H = s.H[u]
        if tau <= 0: qv = np.zeros(C); qv[s.anchor[u]] = 1.0
        elif tau <= H: qv = s.Pk[u][tau - 1]
        else: qv = s.mvec[u]
        q = qv[A]; tot = q.sum()
        if tot <= 0: return float(len(A)), 1
        o = np.argsort(-q); cs = np.cumsum(q[o]) / tot
        cuts = []
        for th in s.thetas:
            n = int(np.searchsorted(cs, th)) + 1
            n = min(max(n, (cuts[-1] + 1) if cuts else 1), len(A))
            if n >= len(A): break
            cuts.append(n)
        cuts.append(len(A))
        k = int(np.where(A[o] == posu)[0][0]) if posu in A else len(A) - 1
        for r, n in enumerate(cuts):
            if k < n: return float(n), r + 1
        return float(len(A)), len(cuts)
    def step(s, t, pos, prev, call, meas, prof):
        out = ~s.inT[np.arange(s.N), pos]
        s.acc.reg(out, t, meas)
        for u in np.where(out)[0]:
            s._end_segment(u, t, 0.0)
            s.anchor[u] = pos[u]; s._build(u, t, explicit=True)
        if call.any():
            cells_ = np.zeros(s.N); rounds_ = np.zeros(s.N)
            for u in np.where(call)[0]:
                c, r = s._page(u, pos[u], t); cells_[u] = c; rounds_[u] = r
                if meas: s.sizes_at_call.append(len(s.A[u]))
                s._end_segment(u, t, c)
                s.anchor[u] = pos[u]; s._build(u, t, explicit=False)
            s.acc.page(call, cells_, rounds_, t, meas)

# ------------------------------------------------------------------ run
def build_schemes(N, T, pos0, prof, seed):
    return [
        TA9(N, T, pos0),
        MB(N, T, pos0, 3),
        DBR(N, T, pos0, 2, True),
        DBR(N, T, pos0, 1, False, name="Ring TAL (7 cells)"),
        DirTAL(N, T, pos0),
        UPAL(N, T, pos0, prof, bandit=False, selective=False, name="UPAL-A1 (predictive TAL only)", seed=seed),
        UPAL(N, T, pos0, prof, bandit=False, selective=True, name="UPAL-A2 (+selective paging)", seed=seed),
        UPAL(N, T, pos0, prof, bandit=True, selective=True, name="UPAL (full)", seed=seed),
    ]

def run(N=120, T=5000, W=1500, lam_scale=0.02, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    pos, cls = gen_traj(N, T, rng)
    lam_u = lam_scale * rng.uniform(0.5, 1.5, size=N)
    calls = rng.random((N, T)) < lam_u[:, None]
    prof = Profile(N)
    schemes = build_schemes(N, T, pos[:, 0], prof, seed)
    t0 = time.time()
    for t in range(1, T):
        prev = pos[:, t - 1]; cur = pos[:, t]; call = calls[:, t]
        prof.update(prev, cur, call.astype(float))
        meas = t >= W
        for sc in schemes: sc.step(t, cur, prev, call, meas, prof)
        if verbose and t % 1000 == 0: print("t", t, round(time.time() - t0, 1), "s", flush=True)
    out = {"lam_scale": lam_scale, "seed": seed, "N": N, "T": T, "W": W,
           "cls": cls.tolist(), "moves_per_step": float((pos[:, W:] != pos[:, W - 1:-1]).mean()), "schemes": {}}
    for sc in schemes:
        a = sc.acc
        out["schemes"][sc.name] = {
            "kappa": sc.kappa, "regs": a.regs.tolist(), "cells": a.cells.tolist(),
            "rounds": a.rounds.tolist(), "calls": a.calls.tolist(),
            "reg_ts": a.reg_ts.reshape(-1, 250).sum(1).tolist(), "pg_ts": a.pg_ts.reshape(-1, 250).sum(1).tolist(),
        }
    out["runtime_s"] = time.time() - t0
    return out

if __name__ == "__main__":
    N = int(sys.argv[1]); T = int(sys.argv[2]); W = int(sys.argv[3])
    lam = float(sys.argv[4]); seed = int(sys.argv[5]); fn = sys.argv[6]
    res = run(N, T, W, lam, seed, verbose=True)
    json.dump(res, open(fn, "w"))
    print("done", fn, round(res["runtime_s"], 1), "s")


# ================= Direction-aware extension (second-order profile with back-off) =================
NS = C * 7
nxt_state = np.zeros((NS, 7), dtype=np.int64)
for c_ in range(C):
    for h_ in range(7):
        s_ = c_ * 7 + h_
        nxt_state[s_, 0] = s_
        for d_ in range(6):
            nxt_state[s_, d_ + 1] = nbr[c_, d_] * 7 + d_

class Profile2(Profile):
    """First-order counts (parent) plus counts keyed by (cell, last heading); heading 6 = none yet."""
    ALPHA2 = 4.0
    def __init__(s, N):
        super().__init__(N)
        s.cnt2 = np.zeros((N, NS, 7), dtype=np.float32); s.h = np.full(N, 6, dtype=np.int64)
    def update(s, prev, pos, call):
        moved = pos != prev
        match = (nbr[prev] == pos[:, None]); di = np.where(moved, match.argmax(1) + 1, 0)
        np.add.at(s.cnt2, (s.ar, prev * 7 + s.h, di), 1.0)
        s.h = np.where(moved, di - 1, s.h)
        super().update(prev, pos, call)
    def rows2(s, u):
        P1 = s.rows(u)                                   # (C,7) shrunk first-order estimate
        c2 = s.cnt2[u].astype(np.float64); n2 = c2.sum(1, keepdims=True)
        return (c2 + s.ALPHA2 * np.repeat(P1, 7, axis=0)) / (n2 + s.ALPHA2)

class UPALD(UPAL):
    """UPAL with a direction-aware occupancy forecast; everything else identical."""
    def _build(s, u, t, explicit, first=False):
        prof = s.prof; start = s.anchor[u]
        H = int(np.clip(round(1.0 / max(prof.lam_hat[u], 1e-3)), 3, s.hmax))
        P2 = prof.rows2(u); p = np.zeros(NS); p[start * 7 + prof.h[u]] = 1.0
        Pk = np.zeros((H, C)); m = np.zeros(C)
        for k in range(H):
            p = np.bincount(nxt_state.ravel(), weights=(p[:, None] * P2).ravel(), minlength=NS)
            marg = p.reshape(C, 7).sum(1); Pk[k] = marg; m += marg
        m[start] += 1e-6
        ctx = s._ctx(u); arm = s._choose(ctx)
        rho = s.rho if arm is None else s.arms[arm]
        order = np.argsort(-m); cs = np.cumsum(m[order]) / m.sum()
        n = int(np.searchsorted(cs, rho)) + 1
        A = order[:n]
        if start not in A: A = np.append(A, start)
        s.inT[u, :] = False; s.inT[u, A] = True
        s.A[u] = A; s.Pk[u] = Pk.astype(np.float32); s.mvec[u] = m; s.H[u] = H
        s.last_t[u] = t; s.seg_t0[u] = t
        s.seg_cost[u] = s.kappa * 10.0 if explicit else 0.0
        s.seg_arm[u] = -1 if arm is None else arm; s.seg_ctx[u] = ctx


class UPALH(UPAL):
    """Hybrid: distance-disc registration area whose radius D is tuned online by the context bandit,
       with probability-ordered multi-round paging from a learned (optionally direction-aware) profile."""
    def __init__(s, *a, direction=False, fixedD=3, **k):
        s.direction = direction; s.fixedD = fixedD
        super().__init__(*a, **k)
    def _build(s, u, t, explicit, first=False):
        prof = s.prof; start = s.anchor[u]
        H = int(np.clip(round(1.0 / max(prof.lam_hat[u], 1e-3)), 3, s.hmax)); Pk = np.zeros((H, C)); m = np.zeros(C)
        if s.direction:
            P2 = prof.rows2(u); p = np.zeros(NS); p[start * 7 + prof.h[u]] = 1.0
            for k in range(H):
                p = np.bincount(nxt_state.ravel(), weights=(p[:, None] * P2).ravel(), minlength=NS)
                marg = p.reshape(C, 7).sum(1); Pk[k] = marg; m += marg
        else:
            P = prof.rows(u); p = np.zeros(C); p[start] = 1.0
            for k in range(H):
                p = np.bincount(nbr7.ravel(), weights=(p[:, None] * P).ravel(), minlength=C); Pk[k] = p; m += p
        m[start] += 1e-6
        ctx = s._ctx(u); arm = s._choose(ctx)
        Dsel = s.fixedD if arm is None else int(s.arms[arm])
        A = np.where(D[start] <= Dsel)[0]
        s.inT[u, :] = False; s.inT[u, A] = True
        s.A[u] = A; s.Pk[u] = Pk.astype(np.float32); s.mvec[u] = m; s.H[u] = H
        s.last_t[u] = t; s.seg_t0[u] = t
        s.seg_cost[u] = s.kappa * 10.0 if explicit else 0.0
        s.seg_arm[u] = -1 if arm is None else arm; s.seg_ctx[u] = ctx


class DBRBandit(UPAL):
    """Distance-based registration whose radius is chosen online per context by the cost bandit;
       shortest-distance-first ring paging (no learned paging, no profile upload)."""
    def _build(s, u, t, explicit, first=False):
        start = s.anchor[u]; ctx = s._ctx(u); arm = s._choose(ctx)
        Dsel = s.fixedD if arm is None else int(s.arms[arm])
        A = np.where(D[start] <= Dsel)[0]
        s.inT[u, :] = False; s.inT[u, A] = True; s.A[u] = A
        s.last_t[u] = t; s.seg_t0[u] = t
        s.seg_cost[u] = s.kappa * 10.0 if explicit else 0.0
        s.seg_arm[u] = -1 if arm is None else arm; s.seg_ctx[u] = ctx
    def _page(s, u, posu, t):
        a = s.anchor[u]; d = int(D[a, posu])
        return float(within[a, min(d, 9)]), d + 1
    fixedD = 3
