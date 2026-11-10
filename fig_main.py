"""Main-text figures for the manuscript: (1) cost vs call rate on Benchmark A including tuned baselines and STR;
(2) cost by mobility class at the main setting."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open("results/summary.json")); T = json.load(open("results/tuned_summary.json")); ST = json.load(open("results/selftune.json"))
lams = ["0.005", "0.01", "0.02", "0.05", "0.1"]; x = [float(l) * 100 for l in lams]
fig, ax = plt.subplots(figsize=(6.2, 3.5))
ax.plot(x, [S["sweep"][l]["cost"]["DBR-2"] for l in lams], marker="o", ms=3.5, lw=1.3, color="#4e79a7", label="Distance-based, fixed D = 2")
ax.plot(x, [S["sweep"][l]["cost"]["UPAL"] for l in lams], marker="o", ms=3.5, lw=2.0, color="#d1481f", label="UPAL (learned lists)")
ax.plot(x, [T["per_lambda"][l]["best_mb_cost"] for l in lams], marker="o", ms=3.5, lw=1.3, color="#b07aa1", label="Movement-based, tuned M")
ax.plot(x, [T["per_lambda"][l]["best_dbr_cost"] for l in lams], marker="o", ms=3.5, lw=2.0, color="#2a9d8f", label="Distance-based, tuned D")
ax.plot(x, [ST[l]["mean"] for l in lams], marker="s", ms=4, lw=2.0, ls="--", color="#1b4332", label="Self-tuning radius (STR)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
ax.set_xlabel("Mean calls per user per 100 slots"); ax.set_ylabel("Signalling cost per user\nper 1000 slots")
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("figs/fig_main_sweep.png", dpi=300); plt.close()
g = T["per_lambda"]["0.02"]["cls_best_dbr"]; o = [v[1] for v in T["perclass_oracle"]["best_per_class"]]; u = T["upal_cls_002"]; s = ST["0.02"]["cls"]
fig, ax = plt.subplots(figsize=(6.2, 3.2)); w = 0.2; names = ["Pedestrian", "Vehicular", "Commuter"]
for k, (lab, v, c) in enumerate([("Tuned D, one value for all (D = 5)", g, "#2a9d8f"), ("Per-class oracle D (3 / 6 / 3)", o, "#8fd3c7"), ("STR", s, "#1b4332"), ("UPAL", u, "#d1481f")]):
    ax.bar(np.arange(3) + (k - 1.5) * w, v, w, color=c, label=lab)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(names); ax.set_ylabel("Signalling cost per user\nper 1000 slots"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig("figs/fig_main_class.png", dpi=300); plt.close()
print("ok")
