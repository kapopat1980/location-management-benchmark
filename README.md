# Location-management benchmark

Code and data for the manuscript *"Tuned distance-based registration with ring paging beats learned and adaptive location
management in simulated cellular networks"* (K. Popat, D. Meva, H. Kukadiya).

The benchmark compares time-, movement- and distance-based location registration, **tuned to each operating
regime and paged ring by ring**, with three adaptive candidates on two independent simulators:

| Candidate | What it is |
|---|---|
| **UPAL** | learned per-user Markov profile, tracking-area list sized from predicted occupancy, two-round probability-ordered paging, bandit-tuned coverage |
| **AFLU** | fused time/distance/movement trigger with a self-tuning per-user threshold and profile-assisted paging (earlier scheme by two of the authors) |
| **STR** | self-tuning radius: distance-based registration whose radius is chosen online per user context by a cost bandit, ring-by-ring paging |

## Headline findings (synthetic mobility; see the manuscript for caveats)

* Against fixed-parameter baselines, UPAL (about 7% at the main setting) and AFLU (20-36%) look better.
* Against distance-based registration with a per-regime threshold and ring-by-ring paging, neither is competitive:
  UPAL costs about 32% more at the main setting (30-44% across call rates) and AFLU 35-61% more on the square grid.
* AFLU's reported gain came from comparing sequential, probability-ordered paging with classical baselines that page a
  whole square area at once; with ring-by-ring paging for the baselines it disappears. The simulator also contained a
  **look-ahead** (the paging histogram includes the user's current cell before the user is located).
* STR needs no offline tuning: it stays within about 6% of the hindsight-tuned radius at every call rate on the hexagonal
  grid and within about 20% on the square grid (held-out check on seeds 11-13 in `bench_str_holdout.py`).
* On 65 real GPS users (GeoLife, Benchmark C) the picture is the same: tuned distance-based registration is cheapest, UPAL costs
  6-106% more and STR 11-19% more than the cross-fitted tuned radius (calls are simulated).

## Layout

| Path | Content |
|---|---|
| `sim.py` | Benchmark A: 547-cell hexagonal grid, mobility models, all baselines, UPAL and variants, STR (`DBRBandit`) |
| `tune_baselines.py`, `analyze3.py` | threshold sweeps for distance-/movement-based registration and paired comparisons |
| `eval_selftune.py` | STR on the evaluation seeds |
| `analyze.py`, `analyze2.py`, `fig_tuned.py`, `fig_main.py`, `fig_design.py` | tables, statistics and figures for Benchmark A |
| `dev_v2.py` ... `dev_v5.py` | exploratory UPAL variants on development seed 101 |
| `bench_square.py`, `bench_summary.py` | Benchmark B: 21x21 square grid, fair baselines, leak-free AFLU |
| `aflu_audit.py` | reduced single-seed audit that exposed the look-ahead |
| `third_party/lu_sim.py` | **unchanged** copy of the earlier AFLU simulator (MIT, see `third_party/LICENSE-AFLU`) |
| `results/`, `figs/` | pre-computed outputs used in the manuscript |
| `reproduce.sh` | regenerates everything (about 1.5 h on one core) |

The README of the earlier AFLU repository, which this repository replaces, stated cost reductions of 20.6-35.9%.
Those figures are superseded by the audit in the manuscript (Section 6.6, Table 12) and should not be relied on.


## Where each table and figure of the manuscript comes from

| Manuscript item | Script(s) | Output |
|---|---|---|
| Table 5 (Benchmark A, main setting), Table 9 and Table 10 (call-rate sweep), Table 7 (UPAL components), Fig. 3 (cost ratio), Fig. 4 (learning curve) | `sim.py`, `analyze.py` | `results/summary.json`, `figs/` |
| Table 8 (per-class breakdown), standard deviations in Table 9 | `analyze2.py` | `results/extra.json` |
| Tuned baselines in Table 5, Fig. 2 | `tune_baselines.py`, `analyze3.py`, `fig_main.py` | `results/tuned.json`, `results/tuned_summary.json` |
| Table 13 (STR vs tuned DBR*), Fig. 2, Fig. 6 | `eval_selftune.py`, `fig_main.py` | `results/selftune.json` |
| Table 11 (exploratory UPAL variants, development seed 101) | `dev_v2.py` ... `dev_v5.py` | `results/dev_v*.log` |
| Table 14 and Fig. 7 (real GPS trajectories) | `run_real.py`, `analyze_real.py` | `results/real_summary.json` |
| Table 6, Table 12 and Fig. 5 (Benchmark B, the AFLU audit and STR on the square grid) | `bench_square.py`, `bench_summary.py` | `results/bench_square*.json` |
| STR held-out confirmation on seeds 11-13 (Section 6.7) | `bench_str_holdout.py` | `results/bench_str_holdout.json` |
| Fig. 1 (study design) | `fig_design.py` | `figs/fig_design.png` |
| single-seed audit that first exposed the look-ahead | `aflu_audit.py` | `results/aflu_audit.json` |


## Real GPS trajectories (Benchmark C) and the GeoLife licence

Benchmark C replays real trajectories from the **GeoLife GPS Trajectories dataset, version 1.3** (Microsoft Research Asia),
converted to cell sequences on the benchmark's hexagonal grid. The dataset is **not included** in this repository: its licence
forbids distributing the software or any derivative work, so the converted trajectories are not shared either. Download it
from the [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=52367), unzip it into `geolife/`,
and `reproduce.sh` will run `geolife_prep.py` and `run_real.py`. The repository contains only aggregated results derived from it.
If you use GeoLife, cite: Y. Zheng, L. Zhang, X. Xie, W.-Y. Ma, *Mining interesting locations and travel sequences from GPS
trajectories*, WWW 2009, pp. 791-800; Y. Zheng, Q. Li, Y. Chen, X. Xie, W.-Y. Ma, *Understanding mobility based on GPS data*,
UbiComp 2008, pp. 312-321; Y. Zheng, X. Xie, W.-Y. Ma, *GeoLife: a collaborative social networking service among user, location
and trajectory*, IEEE Data Eng. Bull. 33(2), 32-40, 2010. The dataset is for non-commercial research use.

| Script | Purpose | Output |
|---|---|---|
| `geolife_prep.py` | GPS points to 60-s cell sequences (hexagon side 1 km), cutting at gaps, off-grid positions and multi-cell jumps | `results/geolife_cells.npz` (not shared) |
| `run_real.py` | all schemes on the real trajectories with simulated calls | `results/real_lam*_s*.json` |
| `analyze_real.py` | per-user costs, cross-fitted tuned baselines, bootstrap intervals over users, Fig. 7 | `results/real_summary.json`, `figs/fig_real.png` |

## Quick start

```bash
pip install -r requirements.txt
python3 sim.py 40 2000 750 0.02 1 /tmp/smoke.json   # ~20 s smoke test (users, ticks, warm-up, call rate, seed, output)
bash reproduce.sh                                     # full reproduction
```

`sim.py` arguments: `N T W lambda seed out.json`; `T` must be a multiple of 250.
All experiments use common random numbers: within a seed, every scheme sees identical trajectories and calls.

## Notes

* All data are synthetic; no personal or external data are used.
* Cost = U x registrations + V x cells paged with U = 10 and V = 1; UPAL carries a 20% registration surcharge
  (profile upload, an assumption). Costs are per user per 1000 slots (Benchmark A) or per user per tick (Benchmark B).
* Tuned baselines are selected in hindsight on the evaluation seeds, which favours them.

## Licence and citation

MIT (see `LICENSE`); `third_party/lu_sim.py` keeps its own notice. Citation metadata: `CITATION.cff`.
After the first release, archive the repository on Zenodo and cite the DOI in the manuscript.
