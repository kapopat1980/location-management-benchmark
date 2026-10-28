"""Fig. 1: study design schematic."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.0, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
def box(x, y, w, h, title, lines, fc, ec="#333333", tc="#111111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.2, "\n".join(lines), ha="center", va="top", fontsize=7.3, color=tc, linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#555555"))
box(1, 28, 29, 26, "1  Three benchmarks", ["A: hexagonal grid, 547 cells,", "120 synthetic users, 5 seeds", "", "B: square grid 21×21, 220 users,", "earlier AFLU simulator, 3 seeds", "", "C: real GPS trajectories (GeoLife),", "65 users on the hexagonal grid"], "#eaf1fb")
box(36, 28, 29, 26, "2  Schemes compared", ["Classical triggers: time, movement,", "distance, profile", "fixed settings  vs  tuned per regime", "blanket  vs  ring-by-ring paging", "", "Adaptive: UPAL, AFLU,", "self-tuning radius (STR)"], "#fdf1e3")
box(71, 28, 28, 26, "3  Common conditions", ["identical trajectories and calls", "for every scheme within a run", "", "cost = 10 × updates", "+ 1 × cells paged", "", "paired statistics over seeds (users on C)"], "#eef7ee")
arrow(30.4, 41, 35.6, 41); arrow(65.4, 41, 70.6, 41)
box(1, 1, 29, 22, "4  Fixed parameters", ["do adaptive schemes beat", "textbook baselines?", "(Tables 5–6, Fig. 2)"], "#f7f7f7")
box(36, 1, 29, 22, "5  Tuned baselines", ["tuned ring-paged distance-based", "registration vs UPAL and AFLU;", "audit of AFLU: look-ahead and", "oversized-area paging (Tables 5, 6, 12)"], "#f7f7f7")
box(71, 1, 28, 22, "6  Self-tuning, theory", ["does an online radius match", "the tuned optimum? (Tables 13–14,", "Figs 6–7); why is a tuned radius", "hard to beat? (Section 6.9)"], "#f7f7f7")
arrow(15.5, 27.4, 15.5, 23.8); arrow(50.5, 27.4, 50.5, 23.8); arrow(85, 27.4, 85, 23.8)
plt.tight_layout(pad=0.3); plt.savefig("figs/fig_design.png", dpi=300)
print("ok")
