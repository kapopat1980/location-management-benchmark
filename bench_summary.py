import json, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
d = json.load(open("results/bench_square.json")); cmrs = [0.25, 0.5, 1, 2, 4, 8, 16]
keys = ['time_blanket','dist_blanket','move_blanket','profile_leaky','time_sdf','dist_sdf','move_sdf','profile_noleak','aflu_original','aflu_noleak','aflu_trigger_sdf','str']
seeds = sorted({k.split('_')[0] for k in d}); out = {"seeds": len(seeds), "cmr": {}}
for c in cmrs:
    per = {k: np.array([d[f"{s}_{c}"][k][0] for s in seeds if f"{s}_{c}" in d]) for k in keys}
    bc = np.min([per[k] for k in ['time_blanket','dist_blanket','move_blanket','profile_leaky']], axis=0)
    bs_name = min(['time_sdf','dist_sdf','move_sdf'], key=lambda k: per[k].mean()); bs = per[bs_name]
    def rel(a, b):
        r = (a - b) / b * 100; n = len(r); tc = stats.t.ppf(0.975, n - 1) if n > 1 else 0
        return float(r.mean()), float(tc * r.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    out["cmr"][str(c)] = {"mean": {k: float(v.mean()) for k, v in per.items()}, "best_blanket": float(bc.mean()), "best_sdf": float(bs.mean()), "best_sdf_name": bs_name,
        "orig_vs_blanket": rel(per['aflu_original'], bc), "noleak_vs_blanket": rel(per['aflu_noleak'], bc),
        "noleak_vs_sdf": rel(per['aflu_noleak'], bs), "trig_sdf_vs_sdf": rel(per['aflu_trigger_sdf'], bs), "orig_vs_sdf": rel(per['aflu_original'], bs),
        "str_vs_sdf": rel(per['str'], bs), "str_wins_over_sdf": int((per['str'] <= bs).sum()),
        "wins_sdf_over_aflu_noleak": int((bs < per['aflu_noleak']).sum()), "n": len(bs)}
json.dump(out, open("results/bench_square_summary.json", "w"), indent=1)

import os
if os.path.exists("results/bench_str_holdout.json"):
    h = json.load(open("results/bench_str_holdout.json")); out["holdout"] = {}
    for c in cmrs:
        ks = [k for k in h if k.endswith("_" + str(c))]
        sv = np.array([h[k]["str"] for k in ks]); bv = np.array([h[k]["best_tuned"] for k in ks]); r = (sv / bv - 1) * 100
        tc = stats.t.ppf(0.975, len(r) - 1)
        out["holdout"][str(c)] = {"str": float(sv.mean()), "best_tuned": float(bv.mean()), "rel": [float(r.mean()), float(tc * r.std(ddof=1) / np.sqrt(len(r)))], "wins": int((sv <= bv).sum()), "n": len(r)}
    json.dump(out, open("results/bench_square_summary.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(6.0, 3.4))
cm = out["cmr"]
for lab, f, col, lw in [("Best classical, blanket paging (AFLU paper baseline)", lambda v: v["best_blanket"], "#8c8c8c", 1.2),
                        ("AFLU as originally simulated", lambda v: v["mean"]["aflu_original"], "#e9a23b", 1.4),
                        ("AFLU, look-ahead removed", lambda v: v["mean"]["aflu_noleak"], "#d1481f", 1.8),
                        ("AFLU trigger + ring paging", lambda v: v["mean"]["aflu_trigger_sdf"], "#b07aa1", 1.4),
                        ("Self-tuning radius (STR), ring-by-ring paging", lambda v: v["mean"]["str"], "#1b4332", 1.8), ("Best tuned classical, ring-by-ring paging", lambda v: v["best_sdf"], "#2a9d8f", 2.2)]:
    ax.plot(cmrs, [f(cm[str(c)]) for c in cmrs], marker="o", ms=3.5, lw=lw, color=col, label=lab)
ax.set_xscale("log", base=2); ax.set_xticks(cmrs); ax.set_xticklabels([f"{c:g}" for c in cmrs])
ax.set_xlabel("Call-to-mobility ratio (CMR)"); ax.set_ylabel("Signalling cost per user per tick"); ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.tight_layout(); plt.savefig("figs/fig_benchB.png", dpi=300, bbox_inches="tight")
for c in cmrs:
    v = cm[str(c)]; print(c, v["best_sdf_name"], f"orig vs blanket {v['orig_vs_blanket'][0]:+.1f}  noleak vs blanket {v['noleak_vs_blanket'][0]:+.1f}  noleak vs SDF {v['noleak_vs_sdf'][0]:+.1f}±{v['noleak_vs_sdf'][1]:.1f}  trig+SDF vs SDF {v['trig_sdf_vs_sdf'][0]:+.1f}  seeds SDF wins {v['wins_sdf_over_aflu_noleak']}/{v['n']}")
