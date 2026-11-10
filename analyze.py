import json, glob, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "figure.dpi": 150})
SHORT = {"TA-9 (static)": "TA-9", "Movement-based (M=3, SDF)": "MB-3",
         "Distance-based (D=2, SDF)": "DBR-2", "Ring TAL (7 cells)": "Ring-TAL7",
         "Direction-sector TAL": "Dir-TAL", "UPAL-A1 (predictive TAL only)": "UPAL-A1",
         "UPAL-A2 (+selective paging)": "UPAL-A2", "UPAL (full)": "UPAL"}
ORDER = list(SHORT.keys())
COL = {"TA-9": "#8c8c8c", "MB-3": "#b07aa1", "DBR-2": "#4e79a7", "Ring-TAL7": "#76b7b2",
       "Dir-TAL": "#59a14f", "UPAL-A1": "#f2b98a", "UPAL-A2": "#ec8b4f", "UPAL": "#d1481f"}
U0, V0 = 10.0, 1.0
os.makedirs("figs", exist_ok=True)

def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]

def metrics(r, U=U0, V=V0):
    steps = r["T"] - r["W"]; cls = np.array(r["cls"]); out = {}
    for n in ORDER:
        s = r["schemes"][n]
        regs = np.array(s["regs"]); cells = np.array(s["cells"]); rounds = np.array(s["rounds"]); calls = np.array(s["calls"])
        regc = U * s["kappa"] * regs / steps * 1000; pgc = V * cells / steps * 1000
        out[SHORT[n]] = {"cost": (regc + pgc).mean(), "regcost": regc.mean(), "pgcost": pgc.mean(),
                         "regs": regs.mean() / steps * 1000, "cells_per_call": cells.sum() / calls.sum(),
                         "delay": rounds.sum() / calls.sum(),
                         "cls": [float((regc + pgc)[cls == c].mean()) for c in range(3)]}
    return out

main = load("results/main_s*.json")
M = [metrics(r) for r in main]
names = [SHORT[n] for n in ORDER]
def agg(key, n): 
    v = np.array([m[n][key] for m in M]); return v.mean(), v.std(ddof=1)
summary = {}
for n in names:
    summary[n] = {k: agg(k, n) for k in ["cost", "regcost", "pgcost", "regs", "cells_per_call", "delay"]}
    cl = np.array([m[n]["cls"] for m in M]); summary[n]["cls"] = cl.mean(0).tolist(); summary[n]["cls_sd"] = cl.std(0, ddof=1).tolist()

# paired statistics UPAL vs each baseline
def paired(a, b):
    d = np.array([m[a]["cost"] - m[b]["cost"] for m in M]); rel = np.array([(m[a]["cost"] - m[b]["cost"]) / m[b]["cost"] * 100 for m in M])
    n = len(d); tcrit = stats.t.ppf(0.975, n - 1)
    return {"mean_diff": d.mean(), "rel_mean": rel.mean(), "rel_ci": tcrit * rel.std(ddof=1) / np.sqrt(n),
            "p": stats.ttest_1samp(d, 0).pvalue, "wins": int((d < 0).sum()), "n": n}
pair = {b: paired("UPAL", b) for b in names if b != "UPAL"}

# ---------------------------------------------------------- Fig 1 cost breakdown
fig, ax = plt.subplots(figsize=(6.4, 3.2))
x = np.arange(len(names))
rc = [summary[n]["regcost"][0] for n in names]; pc = [summary[n]["pgcost"][0] for n in names]
tot_sd = [summary[n]["cost"][1] for n in names]
ax.bar(x, rc, color="#4e79a7", label="Registration (update) cost", width=0.62)
ax.bar(x, pc, bottom=rc, color="#f28e2b", label="Paging cost", width=0.62)
ax.errorbar(x, np.array(rc) + np.array(pc), yerr=tot_sd, fmt="none", ecolor="k", capsize=2, lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylabel("Signalling cost per user\nper 1000 steps"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("figs/fig_cost_breakdown.png", dpi=300); plt.close()

# ---------------------------------------------------------- Fig 2 call-rate sweep
lams = [0.005, 0.01, 0.02, 0.05, 0.1]; sweep = {}
for lam in lams:
    rr = main if lam == 0.02 else load(f"results/lam{lam}_s*.json")
    if not rr: continue
    mm = [metrics(r) for r in rr]
    sweep[lam] = {n: (np.mean([m[n]["cost"] for m in mm]), np.std([m[n]["cost"] for m in mm], ddof=1) if len(mm) > 1 else 0.0) for n in names}
    sweep[lam]["_n"] = len(mm)
    sweep[lam]["_regs"] = {n: np.mean([m[n]["regs"] for m in mm]) for n in names}
    sweep[lam]["_cells"] = {n: np.mean([m[n]["cells_per_call"] for m in mm]) for n in names}
    sweep[lam]["_delay"] = {n: np.mean([m[n]["delay"] for m in mm]) for n in names}
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ls = sorted(sweep.keys())
for n in names:
    y = [sweep[l][n][0] for l in ls]
    ax.plot([l * 100 for l in ls], y, marker="o", ms=3.5, lw=2.0 if n == "UPAL" else 1.1, color=COL[n], label=n,
            zorder=5 if n == "UPAL" else 2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Mean call arrivals per user per 100 steps"); ax.set_ylabel("Signalling cost per user\nper 1000 steps")
ax.set_xticks([l * 100 for l in ls]); ax.set_xticklabels([f"{l*100:g}" for l in ls])
ax.legend(frameon=False, ncol=2, fontsize=7.5)
plt.tight_layout(); plt.savefig("figs/fig_call_sweep.png", dpi=300); plt.close()

# ---------------------------------------------------------- Fig 3 per-class cost
fig, ax = plt.subplots(figsize=(6.4, 3.2))
sel = ["TA-9", "MB-3", "DBR-2", "Dir-TAL", "UPAL-A2", "UPAL"]
w = 0.13; cn = ["Pedestrian (random walk)", "Vehicular (persistent heading)", "Commuter (routine)"]
for i, n in enumerate(sel):
    ax.bar(np.arange(3) + (i - 2.5) * w, summary[n]["cls"], w, yerr=summary[n]["cls_sd"], color=COL[n], label=n,
           error_kw={"lw": 0.7, "capsize": 1.5})
ax.set_xticks(np.arange(3)); ax.set_xticklabels(cn); ax.set_ylabel("Signalling cost per user\nper 1000 steps")
ax.legend(frameon=False, ncol=3, fontsize=7.5)
plt.tight_layout(); plt.savefig("figs/fig_per_class.png", dpi=300); plt.close()

# ---------------------------------------------------------- Fig 4 learning curve
fig, ax = plt.subplots(figsize=(6.0, 3.2))
Nn = main[0]["N"]
for n in ["TA-9", "DBR-2", "Dir-TAL", "UPAL-A2", "UPAL"]:
    full = [k for k in ORDER if SHORT[k] == n][0]
    curves = []
    for r in main:
        s = r["schemes"][full]
        c = (U0 * s["kappa"] * np.array(s["reg_ts"]) + V0 * np.array(s["pg_ts"])) / Nn * 4
        curves.append(c)
    cm = np.mean(curves, 0); x = (np.arange(len(cm)) + 0.5) * 250
    ax.plot(x, cm, lw=2 if n == "UPAL" else 1.2, color=COL[n], label=n, marker="o", ms=2.5)
ax.axvline(main[0]["W"], color="k", ls=":", lw=0.8)
ax.text(main[0]["W"] + 40, 1480, "measurement window starts", fontsize=7, va="top")
ax.set_xlabel("Simulation step"); ax.set_ylabel("Signalling cost per user\nper 1000 steps"); ax.legend(frameon=False, ncol=3, fontsize=7.5)
plt.tight_layout(); plt.savefig("figs/fig_learning.png", dpi=300); plt.close()

# ---------------------------------------------------------- Fig 5 U/V sensitivity
Us = [2, 5, 10, 20, 50]; usens = {}
for U in Us:
    mm = [metrics(r, U=U) for r in main]
    usens[U] = {n: np.mean([m[n]["cost"] for m in mm]) for n in names}
fig, ax = plt.subplots(figsize=(6.0, 3.2))
for n in ["TA-9", "MB-3", "DBR-2", "Dir-TAL", "UPAL-A2", "UPAL"]:
    ax.plot(Us, [usens[U][n] / usens[U]["DBR-2"] for U in Us], marker="o", ms=3.5, lw=2 if n == "UPAL" else 1.1, color=COL[n], label=n)
ax.axhline(1.0, color="k", lw=0.6, ls=":")
ax.set_xscale("log"); ax.set_xticks(Us); ax.set_xticklabels([str(u) for u in Us])
ax.set_xlabel("Update-to-paging cost ratio U/V"); ax.set_ylabel("Cost relative to DBR-2")
ax.legend(frameon=False, ncol=3, fontsize=7.5)
plt.tight_layout(); plt.savefig("figs/fig_ratio.png", dpi=300); plt.close()

# ---------------------------------------------------------- dump numbers
def f(x): return float(x)
dump = {"n_seeds": len(main), "moves_per_step": float(np.mean([r["moves_per_step"] for r in main])),
        "summary": {n: {k: ([f(v) for v in summary[n][k]] if k not in ("cls", "cls_sd") else summary[n][k]) for k in summary[n]} for n in names},
        "paired": {b: {k: f(v) for k, v in p.items()} for b, p in pair.items()},
        "sweep": {str(l): {"n": sweep[l]["_n"], "cost": {n: f(sweep[l][n][0]) for n in names},
                           "regs": {n: f(sweep[l]["_regs"][n]) for n in names},
                           "cells": {n: f(sweep[l]["_cells"][n]) for n in names},
                           "delay": {n: f(sweep[l]["_delay"][n]) for n in names}} for l in ls},
        "usens": {str(U): {n: f(usens[U][n]) for n in names} for U in Us}}
json.dump(dump, open("results/summary.json", "w"), indent=1)
print("saved. seeds:", len(main))
