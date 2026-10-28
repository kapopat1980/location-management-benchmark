"""Summarise the GeoLife real-trace experiment: per-user costs, tuned baselines (hindsight and 2-fold cross-fitted over users),
bootstrap confidence intervals over users, and a breakdown by user mobility tercile."""
import json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import run_real as RR

LAMS = ["0.005", "0.01", "0.02", "0.05", "0.1"]
FIX = {"TA-9": "TA-9 (static)", "MB-3": "Movement-based (M=3, SDF)", "DBR-2": "Distance-based (D=2, SDF)", "Ring-TAL7": "Ring TAL (7 cells)",
       "Dir-TAL": "Direction-sector TAL", "UPAL": "UPAL (full)", "STR": "STR"}
rng = np.random.default_rng(2026)

def user_costs(runs, name):
    """mean over call seeds of per-user cost per 1000 measured slots"""
    cs = []
    for r in runs:
        s = r["schemes"][name]; ms = np.array(r["meas_slots"], dtype=float)
        cs.append((10.0 * s["kappa"] * np.array(s["regs"]) + np.array(s["cells"])) / np.maximum(ms, 1) * 1000)
    return np.mean(cs, axis=0)

def boot_rel(x, ref, B=4000):
    n = len(x); idx = rng.integers(0, n, size=(B, n))
    rel = (x[idx].mean(1) / ref[idx].mean(1) - 1) * 100
    return float((x.mean() / ref.mean() - 1) * 100), float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))

# per-user mobility (cell changes per measured slot)
if os.path.exists("results/geolife_cells.npz"):          # converted trajectories are not redistributed (dataset licence)
    users, pos, active, reset, meas = RR.load()
    mv = ((pos[:, 1:] != pos[:, :-1]) & (~reset[:, 1:]) & meas[:, 1:]).sum(1) / np.maximum(meas[:, 1:].sum(1), 1)
    json.dump([float(x) for x in mv], open("results/real_move_frac.json", "w"))
else:
    mv = np.array(json.load(open("results/real_move_frac.json")))
_first = json.load(open(sorted(glob.glob("results/real_lam*_s*.json"))[0]))
users = list(range(len(mv))); slots_measured = int(sum(_first["meas_slots"]))
terc = np.digitize(mv, np.quantile(mv, [1 / 3, 2 / 3]))
out = {"n_users": len(users), "slots_measured": slots_measured, "move_mean": float(mv.mean()), "move_terciles": [float(mv[terc == k].mean()) for k in range(3)], "lam": {}}
fold = np.arange(len(users)) % 2
for lam in LAMS:
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/real_lam{lam}_s*.json"))]
    if not runs: continue
    uc = {k: user_costs(runs, v) for k, v in FIX.items()}
    dsw = {d: user_costs(runs, f"DBR sweep D={d}") for d in range(1, 9)}
    msw = {m: user_costs(runs, f"Movement-based (M={m}, SDF)") for m in range(2, 9)}
    best_d = min(dsw, key=lambda d: dsw[d].mean()); best_m = min(msw, key=lambda m: msw[m].mean())
    dstar_h = dsw[best_d]; mstar_h = msw[best_m]
    dstar_cv = np.zeros(len(users)); mstar_cv = np.zeros(len(users)); chosen = []
    for f in (0, 1):
        tr = fold != f; te = fold == f
        dd = min(dsw, key=lambda d: dsw[d][tr].mean()); mm = min(msw, key=lambda m: msw[m][tr].mean()); chosen.append((dd, mm))
        dstar_cv[te] = dsw[dd][te]; mstar_cv[te] = msw[mm][te]
    rec = {"n_seeds": len(runs), "mean": {k: float(v.mean()) for k, v in uc.items()}, "dbr_star_hindsight": float(dstar_h.mean()), "best_D": best_d,
           "dbr_star_cv": float(dstar_cv.mean()), "cv_D": [c[0] for c in chosen], "mb_star_hindsight": float(mstar_h.mean()), "best_M": best_m, "mb_star_cv": float(mstar_cv.mean()), "cv_M": [c[1] for c in chosen],
           "rel": {}, "users_better": {}}
    for nm, x, ref in [("UPAL_vs_DBR2", uc["UPAL"], uc["DBR-2"]), ("UPAL_vs_DBRstar_h", uc["UPAL"], dstar_h), ("UPAL_vs_DBRstar_cv", uc["UPAL"], dstar_cv),
                       ("STR_vs_DBR2", uc["STR"], uc["DBR-2"]), ("STR_vs_DBRstar_h", uc["STR"], dstar_h), ("STR_vs_DBRstar_cv", uc["STR"], dstar_cv),
                       ("DBRstar_cv_vs_DBR2", dstar_cv, uc["DBR-2"]), ("MBstar_cv_vs_DBRstar_cv", mstar_cv, dstar_cv), ("UPAL_vs_STR", uc["UPAL"], uc["STR"])]:
        rec["rel"][nm] = boot_rel(x, ref)
    rec["users_better"] = {"UPAL_le_DBRstar_cv": float((uc["UPAL"] <= dstar_cv).mean()), "STR_le_DBRstar_cv": float((uc["STR"] <= dstar_cv).mean()), "UPAL_le_DBR2": float((uc["UPAL"] <= uc["DBR-2"]).mean())}
    rec["terciles"] = {}
    for k in range(3):
        m_ = terc == k
        rec["terciles"][str(k)] = {n: float(v[m_].mean()) for n, v in [("DBR-2", uc["DBR-2"]), ("DBR*_cv", dstar_cv), ("UPAL", uc["UPAL"]), ("STR", uc["STR"])]}
    out["lam"][lam] = rec
json.dump(out, open("results/real_summary.json", "w"), indent=1)
# figure: cost vs call rate on the real traces
ls = [l for l in LAMS if l in out["lam"]]; x = [float(l) * 100 for l in ls]
fig, ax = plt.subplots(figsize=(6.2, 3.7))
for lab, f, col, lw, mk in [("Distance-based, fixed D = 2", lambda r: r["mean"]["DBR-2"], "#4e79a7", 1.3, "o"), ("UPAL", lambda r: r["mean"]["UPAL"], "#d1481f", 2.0, "o"),
                            ("Movement-based, tuned M (cross-fitted)", lambda r: r["mb_star_cv"], "#b07aa1", 1.3, "o"), ("Distance-based, tuned D (cross-fitted)", lambda r: r["dbr_star_cv"], "#2a9d8f", 2.0, "o"),
                            ("Self-tuning radius (STR)", lambda r: r["mean"]["STR"], "#1b4332", 2.0, "s")]:
    ax.plot(x, [f(out["lam"][l]) for l in ls], marker=mk, ms=3.8, lw=lw, color=col, label=lab, ls="--" if "STR" in lab else "-")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
ax.set_xlabel("Mean calls per user per 100 slots (1-minute slots)"); ax.set_ylabel("Signalling cost per user\nper 1000 slots")
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("figs/fig_real.png", dpi=300)
for l in ls:
    r = out["lam"][l]; rel = r["rel"]
    print(l, "DBR-2 %.0f UPAL %.0f STR %.0f DBR*cv %.0f (D %s) DBR*h %.0f (D %d) MB*cv %.0f" % (r["mean"]["DBR-2"], r["mean"]["UPAL"], r["mean"]["STR"], r["dbr_star_cv"], r["cv_D"], r["dbr_star_hindsight"], r["best_D"], r["mb_star_cv"]),
          "| UPAL vs DBR*cv %+.1f [%+.1f,%+.1f]" % rel["UPAL_vs_DBRstar_cv"], "| STR vs DBR*cv %+.1f [%+.1f,%+.1f]" % rel["STR_vs_DBRstar_cv"], "| UPAL vs DBR-2 %+.1f [%+.1f,%+.1f]" % rel["UPAL_vs_DBR2"])
