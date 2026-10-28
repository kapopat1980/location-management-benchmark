"""Convert the GeoLife GPS trajectories (Microsoft Research Asia, v1.3) into cell sequences on the benchmark's hexagonal grid.
Cells: 547 hexagonal cells (radius 13 rings), hexagon side SIDE metres (pointy-top, axial coordinates as in sim.py).
Slots: 60 s. A trajectory file is split into continuous segments at raw gaps > 120 s, at positions outside the grid, and at
steps larger than one cell. Each user's segments are concatenated; a reset flag marks the first slot of every segment
(the device is switched on or tracking resumes there)."""
import os, sys, glob, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim                                    # hex grid definition (cells, idx, D, R)

ROOT = sys.argv[1] if len(sys.argv) > 1 else "geolife/Geolife Trajectories 1.3/Data"
SIDE = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/geolife_cells.npz"
SLOT = 60.0; GAP = 120.0; MIN_SLOTS = 10
R = sim.R
lut = -np.ones((2 * R + 1, 2 * R + 1), dtype=np.int64)
for (q, r), i in sim.idx.items(): lut[q + R, r + R] = i

def to_cells(x, y):
    q = (np.sqrt(3) / 3 * x - y / 3) / SIDE; r = (2.0 / 3 * y) / SIDE; s = -q - r
    rq, rr, rs = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(rq - q), np.abs(rr - r), np.abs(rs - s)
    m1 = (dq > dr) & (dq > ds); m2 = (~m1) & (dr > ds)
    rq = np.where(m1, -rr - rs, rq); rr = np.where(m2, -rq - rs, rr)
    qi = rq.astype(int) + R; ri = rr.astype(int) + R
    ok = (qi >= 0) & (qi < 2 * R + 1) & (ri >= 0) & (ri < 2 * R + 1)
    out = -np.ones(len(x), dtype=np.int64); out[ok] = lut[qi[ok], ri[ok]]
    return out

def read_plt(f):
    d = pd.read_csv(f, skiprows=6, header=None, usecols=[0, 1, 4], names=["lat", "lon", "days"], dtype=np.float64)
    return d["lat"].values, d["lon"].values, d["days"].values * 86400.0

# data-driven projection centre: median of points in a Beijing bounding box (sampled files)
files = sorted(glob.glob(os.path.join(ROOT, "*", "Trajectory", "*.plt")))
samp = files[::40]; la, lo = [], []
for f in samp:
    a, b, _ = read_plt(f); m = (a > 39.6) & (a < 40.3) & (b > 115.9) & (b < 116.9); la.append(a[m]); lo.append(b[m])
lat0 = float(np.median(np.concatenate(la))); lon0 = float(np.median(np.concatenate(lo)))
kx = 111320.0 * np.cos(np.radians(lat0)); ky = 110540.0
print("centre", lat0, lon0, "files", len(files), flush=True)

users = sorted(os.listdir(ROOT)); res = {}; stats = {"files": len(files), "segments": 0, "slots": 0, "cut_jump": 0, "cut_outside": 0}
t0 = time.time()
for u in users:
    cells, resets = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, u, "Trajectory", "*.plt"))):
        lat, lon, t = read_plt(f)
        if len(t) < 2: continue
        keep = np.concatenate([[True], np.diff(t) > 0]); lat, lon, t = lat[keep], lon[keep], t[keep]
        x = (lon - lon0) * kx; y = (lat - lat0) * ky
        brk = np.where(np.diff(t) > GAP)[0] + 1; starts = np.concatenate([[0], brk]); ends = np.concatenate([brk, [len(t)]])
        for a, b in zip(starts, ends):
            if b - a < 2: continue
            ts = np.arange(np.ceil(t[a] / SLOT) * SLOT, np.floor(t[b - 1] / SLOT) * SLOT + 1, SLOT)
            if len(ts) < MIN_SLOTS: continue
            c = to_cells(np.interp(ts, t[a:b], x[a:b]), np.interp(ts, t[a:b], y[a:b]))
            # cut at outside-grid slots and at steps larger than one cell
            bad = c < 0; stats["cut_outside"] += int(bad.sum())
            j = np.zeros(len(c), dtype=bool); okp = (~bad[1:]) & (~bad[:-1])
            j[1:][okp] = sim.D[c[1:][okp], c[:-1][okp]] > 1; stats["cut_jump"] += int(j.sum())
            cut = np.where(bad | j)[0]
            bounds = np.concatenate([[0], cut, [len(c)]])
            for s0, s1 in zip(bounds[:-1], bounds[1:]):
                seg = c[s0:s1]; seg = seg[seg >= 0]
                if len(seg) >= MIN_SLOTS:
                    cells.append(seg); r = np.zeros(len(seg), dtype=bool); r[0] = True; resets.append(r); stats["segments"] += 1; stats["slots"] += len(seg)
    if cells: res[u] = (np.concatenate(cells), np.concatenate(resets))
    print(u, (len(res[u][0]) if u in res else 0), round(time.time() - t0), "s", flush=True)
np.savez_compressed(OUT, users=np.array(list(res.keys())), **{f"c_{u}": v[0].astype(np.int16) for u, v in res.items()}, **{f"r_{u}": v[1] for u, v in res.items()})
json.dump({"centre": [lat0, lon0], "side_m": SIDE, "slot_s": SLOT, **stats}, open(OUT.replace(".npz", "_meta.json"), "w"), indent=1)
print(stats)
