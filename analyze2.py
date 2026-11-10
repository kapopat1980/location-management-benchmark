import json, glob, numpy as np
U0, V0 = 10.0, 1.0
ORDER = ["TA-9 (static)","Movement-based (M=3, SDF)","Distance-based (D=2, SDF)","Ring TAL (7 cells)","Direction-sector TAL","UPAL-A1 (predictive TAL only)","UPAL-A2 (+selective paging)","UPAL (full)"]
SHORT = dict(zip(ORDER, ["TA-9","MB-3","DBR-2","Ring-TAL7","Dir-TAL","UPAL-A1","UPAL-A2","UPAL"]))
main = [json.load(open(f)) for f in sorted(glob.glob("results/main_s*.json"))]
out = {"perclass": {}, "sweep_sd": {}, "regcost_share": {}}
# per-class decomposition (pooled over seeds)
for n in ORDER:
    sn = SHORT[n]; out["perclass"][sn] = {}
    for c in range(3):
        regs = cells = calls = rounds = 0.0; users = 0
        for r in main:
            cls = np.array(r["cls"]); s = r["schemes"][n]; steps = r["T"] - r["W"]
            m = cls == c
            regs += np.array(s["regs"])[m].sum() / steps * 1000; users += m.sum()
            cells += np.array(s["cells"])[m].sum(); calls += np.array(s["calls"])[m].sum(); rounds += np.array(s["rounds"])[m].sum()
        out["perclass"][sn][c] = {"regs_per_user_per_1000": regs / users, "cells_per_call": cells / calls, "rounds_per_call": rounds / calls}
# sweep sd
for lam in ["0.005", "0.01", "0.05", "0.1"]:
    rr = [json.load(open(f)) for f in sorted(glob.glob(f"results/lam{lam}_s*.json"))]
    out["sweep_sd"][lam] = {}
    for n in ORDER:
        v = []
        for r in rr:
            s = r["schemes"][n]; steps = r["T"] - r["W"]
            v.append(((U0 * s["kappa"] * np.array(s["regs"]) + V0 * np.array(s["cells"])) / steps * 1000).mean())
        out["sweep_sd"][lam][SHORT[n]] = float(np.std(v, ddof=1))
v = {}
for n in ORDER:
    sd = []
    for r in main:
        s = r["schemes"][n]; steps = r["T"] - r["W"]
        sd.append(((U0 * s["kappa"] * np.array(s["regs"]) + V0 * np.array(s["cells"])) / steps * 1000).mean())
    v[SHORT[n]] = float(np.std(sd, ddof=1))
out["sweep_sd"]["0.02"] = v
# share of total users by class
cls = np.array(main[0]["cls"]); out["class_counts"] = [int((cls == c).sum()) for c in range(3)]
json.dump(out, open("results/extra.json", "w"), indent=1)
for sn in ["DBR-2", "UPAL"]:
    for c in range(3):
        p = out["perclass"][sn][c]; print(sn, c, {k: round(x, 2) for k, x in p.items()})
print(out["class_counts"])
print({k: round(x, 1) for k, x in out["sweep_sd"]["0.005"].items()})
