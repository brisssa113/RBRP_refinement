# RBRP_refinement — reproducibility package

Code, stored raw results and LaTeX source for the paper on **controlling the interpolation-matrix amplification of
RBRP skeletons** (robust blockwise random pivoting, Dong–Chen–Martinsson–Pearce, SIAM J. Matrix Anal. Appl. 46(3), 2025).
Given an RBRP skeleton `S0`, the refinement exchanges skeleton rows to reduce the amplification `A = ||W||_2`,
`W = X X_S^†`, under a hard budget `E(S) <= 1.05 E(S0)` on the relative Frobenius reconstruction error.
Compared methods: RBRP, QRCP, Vol(f=1) (unguarded maxvol-type exchange), VolGuard-500, Ours-T10, Hybrid-T10 (= VolGuard-500 → Ours-T10), Ours-full.

## Repository layout

```
RBRP_refinement/
├── README.md
├── requirements.txt
├── run_all.py              one entry point (no argument = tables + figures from results/); also: smoke | experiments | realdata | paper
├── code/                   everything the paper depends on (flat on purpose: scripts load each other via importlib)
│   ├── stage1.py                   RBRP core (blocked_rpid), matrix families, diagnostics, Stage-1 matched-gap study (Sec. 3)
│   ├── stage2_sorted_scan.py       the refinement algorithm (sorted scan), shared setup (B = 4, seeds, datasets); synthetic n=500 runs
│   ├── stage2_t_sweep.py           scan budget T and ordering ablation
│   ├── stage3_rank_sweep.py        rank sensitivity
│   ├── stage3_runtime_scaling.py   wall-clock and scaling in n
│   ├── stage3_realistic_baseline.py  Burgers / digits matrices, QRCP baseline, full scans (input of later scripts)
│   ├── stage3_volume_baseline.py   volume-exchange baselines (also imported as a module by the final/noise/real-data scripts)
│   ├── stage3_extra_ablations.py   error-guard ablation, refining the QRCP skeleton
│   ├── stage3_final_experiments.py all-method paired comparison (main comparison table, pooled statistics)
│   ├── stage3_qrcp_synthetic.py    QRCP baseline on the four synthetic datasets, paired with the final experiment (20 instances each)
│   ├── check_exact_rank_bound.py   numerical check of the exact-rank proposition (Sec. 4) with the real Vol/VolGuard code
│   ├── stage3_noise_downstream.py  downstream perturbation experiment (random / worst-case noise)
│   ├── stage3_realdata.py, realdata_loaders.py   additional real matrices: image patches, ECG Hankel, digits kernel
│   ├── stage1_within_rank_stats.py  within-rank spread of A and E for RBRP skeletons (Sec. 3.3, 'typical spread')
│   ├── bootstrap_ci.py             95% bootstrap intervals (seed = resampling unit, B = 5000, fixed RNG seed)
│   ├── gen_tables.py, gen_ci_tables.py, gen_real_tables.py   CSV -> paper/tables/*.tex
│   └── make_figures.py             results/ -> paper/figs/
├── results/                stored raw outputs (CSV/PNG/JSON), one folder per script; ~10 MB, kept under version control
├── paper/                  main.tex, main.pdf, tables/ (generated), figs/ (generated)
```

## Quick start

```bash
pip install -r requirements.txt
python run_all.py                 # regenerates all LaTeX tables and all figures from results/ (about 2 min, incl. bootstrap)
python run_all.py paper           # optional, needs latexmk
```

`python run_all.py smoke` runs every experiment script on a tiny problem (sanity check of the installation, ~10 min on one core).
`python run_all.py experiments` re-runs **everything** from scratch into `results/` (many hours); each script can also be run alone,
all have `--smoke`, `--only <matrix>` and `--summarize-only` (see the docstring of each script). Use one thread
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`) for the runtime experiment.

**Real-data matrices (image patches, ECG Hankel, digits kernel):** raw results are in `results/stage3_realdata_results/`; `run_all.py` regenerates their tables, bootstrap intervals and `scatter_real.png` like all other items.

## From paper to code

| Paper item | Script (defaults = paper settings) | Stored result in `results/` |
|---|---|---|
| Sec. 3, Table `stage1`, Fig. `stage1` (b = 4) | `stage1.py --block 4 --outdir results/stage1_b4_results` | `stage1_b4_results/` |
| Sec. 3.3 'typical spread' table | `stage1_within_rank_stats.py` | `stage1_b4_results/within_rank_stats.csv` |
| Sec. 3 text, "block size b = 8" numbers | `stage1.py --block 8 --outdir results/stage1_b8_results` | `stage1_b8_results/` |
| Table `pilot` (4 synthetic sets, 40 seeds) | `stage2_sorted_scan.py` | `stage2_sorted_scan_results/raw_results.csv` |
| Table `tsweep`, Fig. `tsweep` | `stage2_t_sweep.py` | `stage2_T_sweep_results/` |
| Table `rank`, Fig. `rank` | `stage3_rank_sweep.py` | `stage3_rank_sweep_results/` |
| Table `runtime`, Fig. `runtime` | `stage3_runtime_scaling.py` | `stage3_runtime_results/` |
| (input of the next three) Burgers/digits, QRCP, full scans | `stage3_realistic_baseline.py` | `stage3_baseline_results/` |
| Table `guard`, Fig. `guardA`, Table `qrcp` | `stage3_extra_ablations.py` | `stage3_ablation_results/` |
| Table `compare`, Table `pooled`, Fig. `scatter` | `stage3_final_experiments.py` (+ `bootstrap_ci.py`) | `stage3_final_results/`, `bootstrap_synthetic_and_baselines/` |
| Remark after Prop. `exact` (exact-rank bound check) | `check_exact_rank_bound.py` | `exact_rank_check/` |
| Table `qrcp_synthetic` (QRCP on the synthetic datasets) | `stage3_qrcp_synthetic.py` (needs `stage3_final_results`) | `stage3_qrcp_synthetic_results/` |
| Tables `noise`, `nustar`, Figs. `noise_random`, `noise_worst` | `stage3_noise_downstream.py` (+ `bootstrap_ci.py`) | `stage3_noise_results/`, `bootstrap_synthetic_and_baselines/` |
| Tables `real`, `real_qrcp`, `real_pooled`, Fig. `scatter_real` | `stage3_realdata.py` (+ `bootstrap_ci.py`) | `stage3_realdata_results/`, `bootstrap_real/` |

Order of dependence when re-running from scratch: `stage1` → `stage2_sorted_scan` → {`stage2_t_sweep`, `stage3_rank_sweep`, `stage3_runtime_scaling`}
and `stage3_realistic_baseline` → {`stage3_extra_ablations`, `stage3_final_experiments`} → `stage3_qrcp_synthetic`, `stage3_noise_downstream`, `stage3_realdata` → bootstrap → tables/figures.
(`stage3_final_experiments.py` reuses the full-scan results of `stage2_sorted_scan` and `stage3_realistic_baseline`; this is why `run_all.py experiments` keeps this order.)

## Verification status

Checked on Python 3.12.3, numpy 2.4, scipy 1.17, pandas 3.0, scikit-learn 1.8, scikit-image 0.26 (Linux, 1 core):

* All ten generated tables of the main text (`pilot, tsweep, rank, compare, guard, runtime, qrcp, pooled, noise, nustar`) are regenerated **byte-identically** from the stored CSVs; the six bootstrap outputs are byte-identical too.
* `stage1.py --block 4` re-run from scratch reproduces `stage1_b4_results` (`C_RBRP_all_families.csv`, `A_exact_counterexample.csv`, `B_noisy_fixed_rank_pairs.csv`) byte-identically; the hand-typed Stage-1 table and the b = 8 numbers quoted in Sec. 3 match the stored summaries.
* All experiment scripts pass `--smoke`.
* The three real-data tables and their bootstrap outputs are byte-identical as well.
* Figures: twelve are exact copies of script outputs; `tsweep.png` (paper version kept) can be re-drawn by `make_figures.py` in paper styling.

## Known caveats

* **Burgers snapshot matrix** is generated by a 1000-step FFT time stepping; re-runs agree with the stored results only up to a few percent across machines/library versions. Other matrices are deterministic given the seeds. Re-running timing experiments will of course give different wall-clock numbers.
* Block size: the paper's experiments use `b = 4` (`B = 4` in `stage2_sorted_scan.py`); `stage1.py` itself defaults to `--block 8`, so `--block 4` must be passed (`run_all.py` does this).
* `scipy.datasets.electrocardiogram` downloads one small file on first use (needs internet once).
* `olivetti` is implemented in `realdata_loaders.py` but is **not** used in the paper (needs a figshare download, untested).
