import json, glob, numpy as np
from scipy import stats
T = json.load(open("results/tuned.json"))
def upal_seed_costs(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f)); s = r["schemes"]["UPAL (full)"]; steps = r["T"] - r["W"]
        out.append(float(((10.0 * s["kappa"] * np.array(s["regs"]) + np.array(s["cells"])) / steps * 1000).mean()))
    return np.array(out)
res = {"per_lambda": {}, "perclass_oracle": None}
for lam in ["0.005", "0.01", "0.02", "0.05", "0.1"]:
    pat = f"results/main_s*.json" if lam == "0.02" else f"results/lam{lam}_s*.json"
    u = upal_seed_costs(pat); tl = T[lam]
    dbr = {n: v for n, v in tl.items() if n.startswith("Distance")}
    mb = {n: v for n, v in tl.items() if n.startswith("Movement")}
    bd = min(dbr, key=lambda n: dbr[n]["cost"]); bm = min(mb, key=lambda n: mb[n]["cost"])
    ps = np.array(dbr[bd]["per_seed"])
    d = u - ps; rel = d / ps * 100; n = len(d); tc = stats.t.ppf(0.975, n - 1)
    res["per_lambda"][lam] = {"best_dbr": bd, "best_dbr_cost": dbr[bd]["cost"], "best_dbr_sd": dbr[bd]["sd"], "best_mb": bm, "best_mb_cost": mb[bm]["cost"],
        "upal": float(u.mean()), "upal_sd": float(u.std(ddof=1)), "rel_mean": float(rel.mean()), "rel_ci": float(tc * rel.std(ddof=1) / np.sqrt(n)),
        "p": float(stats.ttest_1samp(d, 0).pvalue), "wins": int((d < 0).sum()), "n": int(n), "cls_best_dbr": dbr[bd]["cls"]}
tl = T["0.02"]; dbr = {n: v for n, v in tl.items() if n.startswith("Distance")}
orc = []
for c in range(3):
    bn = min(dbr, key=lambda n: dbr[n]["cls"][c]); orc.append((bn, dbr[bn]["cls"][c]))
res["perclass_oracle"] = {"best_per_class": orc, "weighted_cost": 0.4 * orc[0][1] + 0.3 * orc[1][1] + 0.3 * orc[2][1]}
s = json.load(open("results/summary.json"))["summary"]["UPAL"]["cls"]; res["upal_cls_002"] = s
json.dump(res, open("results/tuned_summary.json", "w"), indent=1)
for lam, v in res["per_lambda"].items():
    print(lam, v["best_dbr"], round(v["best_dbr_cost"]), "| best MB", v["best_mb"], round(v["best_mb_cost"]), "| UPAL", round(v["upal"]), f"rel {v['rel_mean']:+.1f}±{v['rel_ci']:.1f}% p={v['p']:.3f} wins {v['wins']}/{v['n']}")
print(res["perclass_oracle"], res["upal_cls_002"])
