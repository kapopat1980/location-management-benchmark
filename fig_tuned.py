import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
S = json.load(open("results/summary.json")); T = json.load(open("results/tuned_summary.json"))
lams = ["0.005", "0.01", "0.02", "0.05", "0.1"]; x = [float(l) * 100 for l in lams]
fig, ax = plt.subplots(figsize=(6.0, 3.3))
ax.plot(x, [S["sweep"][l]["cost"]["DBR-2"] for l in lams], marker="o", ms=3.5, lw=1.2, color="#4e79a7", label="DBR-2 (fixed D = 2)")
ax.plot(x, [T["per_lambda"][l]["best_mb_cost"] for l in lams], marker="o", ms=3.5, lw=1.2, color="#b07aa1", label="MB* (tuned M)")
ax.plot(x, [T["per_lambda"][l]["best_dbr_cost"] for l in lams], marker="o", ms=3.5, lw=2.0, color="#2a9d8f", label="DBR* (tuned D)")
ax.plot(x, [S["sweep"][l]["cost"]["UPAL"] for l in lams], marker="o", ms=3.5, lw=2.0, color="#d1481f", label="UPAL")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
ax.set_xlabel("Mean call arrivals per user per 100 steps"); ax.set_ylabel("Signalling cost per user\nper 1000 steps")
ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig("figs/fig_tuned.png", dpi=300)
