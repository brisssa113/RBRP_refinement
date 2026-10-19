"""Stage 3.2: runtime relative to RBRP, and scaling with the matrix size n.

For each dataset (representative k of Section 7) and each n in N_LIST (d = 60 fixed):
    - time RBRP (blocked_rpid) alone                      -> median of several repetitions
    - time the refinement, variant 'T10' and variant 'full' (ordered scan, Section 6)
    - record N_eval, rounds, rho_A, rho_E

The full scan is expensive (cost grows roughly like n^2), so it is only run for n <= --full-max-n,
and with fewer seeds for large n. RBRP and T10 are run for all n.
Timing is wall-clock (time.perf_counter); set OMP_NUM_THREADS=1 (and OPENBLAS/MKL) for clean numbers.

Usage (next to stage1.py, stage2_sorted_scan.py):
    python stage3_runtime_scaling.py --smoke --ref-csv raw_results.csv
    python stage3_runtime_scaling.py --ref-csv raw_results.csv
    python stage3_runtime_scaling.py --ref-csv raw_results.csv --only kernel_snapshots   # one dataset
    python stage3_runtime_scaling.py --summarize-only
"""
from __future__ import annotations
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # repository root (results/ lives here)
N_LIST = [500, 1000, 2000, 4000]
REPRESENTATIVE = {'nearly_collinear_clusters': 4, 'repeated_directions': 4, 'localized_svd': 14, 'kernel_snapshots': 16}
T_BUDGET = 10
RBRP_REPS = 5


def load_base():
    spec = importlib.util.spec_from_file_location('stage2_sorted_scan', HERE / 'stage2_sorted_scan.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def run_dataset(base, mod, ds, ns, seeds_small, seeds_large, full_max_n, out):
    di = list(base.DATASET_K).index(ds)
    k = REPRESENTATIVE[ds]
    builder = getattr(mod, base.BUILDERS[ds])
    rows = []
    t_ds = time.perf_counter()
    for n in ns:
        n_seeds = seeds_small if n <= 1000 else seeds_large
        for seed in range(n_seeds):
            X = builder(n=n, d=60, seed=seed)
            rbrp_seed = 1_000_000 + di * 100_000 + seed          # same formula as Section 7
            times = []
            for _ in range(RBRP_REPS):
                t = time.perf_counter()
                row_perm, *_ = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=rbrp_seed)
                times.append(time.perf_counter() - t)
            t_rbrp = float(np.median(times))
            S0 = np.asarray(row_perm[:k], dtype=int)
            t = time.perf_counter()
            d0 = mod.compute_id_diagnostics(X, S0)
            t_diag = time.perf_counter() - t
            if not d0['valid']:
                rows.append({'dataset': ds, 'n': n, 'seed': seed, 'variant': 'invalid_S0'})
                continue
            E0, A0 = float(d0['error_rel']), float(d0['amp2'])
            rows.append({'dataset': ds, 'n': n, 'k': k, 'seed': seed, 'variant': 'RBRP', 'E0': E0, 'A0': A0,
                         'time_sec': t_rbrp, 'one_eval_sec': t_diag})
            variants = [('T10', T_BUDGET)] + ([('full', None)] if n <= full_max_n else [])
            for name, T in variants:
                t = time.perf_counter()
                Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, max_scan=T)
                el = time.perf_counter() - t
                rows.append({'dataset': ds, 'n': n, 'k': k, 'seed': seed, 'variant': name, 'E0': E0, 'A0': A0,
                             'Ef': float(df['error_rel']), 'Af': float(df['amp2']),
                             'rho_A': float(df['amp2']) / A0, 'rho_E': float(df['error_rel']) / E0,
                             'n_eval': ne, 'rounds': sum(h['accepted'] for h in hist), 'time_sec': el,
                             'time_per_eval_ms': 1000 * el / max(ne, 1), 'stop_reason': why,
                             'rbrp_time_sec': t_rbrp, 'one_eval_sec': t_diag})
            print(f'  [{ds}] n={n} seed {seed + 1}/{n_seeds} done, elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
            pd.DataFrame(rows).to_csv(out / f'runtime_raw_{ds}.csv', index=False)      # checkpoint


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan


def summarize(out, ref_path):
    files = sorted(glob.glob(str(out / 'runtime_raw_*.csv')))
    if not files:
        print('no raw files'); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bad = df[df.variant == 'invalid_S0']
    if len(bad):
        print(f'WARNING {len(bad)} invalid S0 rows')
    df = df[df.variant != 'invalid_S0']
    if ref_path:
        ref = pd.read_csv(ref_path)
        for var, m in (('T10', 'sorted_T10'), ('full', 'sorted_full')):
            a = df[(df.n == 500) & (df.variant == var)][['dataset', 'seed', 'Af']]
            b = ref[ref.method == m][['dataset', 'seed', 'Af']]
            mm = a.merge(b, on=['dataset', 'seed'], suffixes=('_new', '_old'))
            if len(mm):
                print(f'reproducibility n=500, {var} vs Section 7: max rel |Af diff| = '
                      f'{np.max(np.abs(mm.Af_new - mm.Af_old) / mm.Af_old):.2e} over {len(mm)} runs')
    g = df.groupby(['dataset', 'n', 'variant'])
    s = g.agg(runs=('time_sec', 'size'), time_sec=('time_sec', 'mean'), n_eval=('n_eval', 'mean'),
              rounds=('rounds', 'mean'), ms_per_eval=('time_per_eval_ms', 'mean'), rho_A=('rho_A', 'mean'),
              rho_E=('rho_E', 'mean')).reset_index()
    s.to_csv(out / 'runtime_summary.csv', index=False)
    print('\n=== runtime / scaling summary (means over seeds) ===')
    print(s.round(4).to_string(index=False))

    # refinement cost relative to RBRP
    piv = s.pivot_table(index=['dataset', 'n'], columns='variant', values='time_sec')
    if 'RBRP' in piv.columns:
        r = pd.DataFrame(index=piv.index)
        r['RBRP_ms'] = 1000 * piv['RBRP']
        for v in ('T10', 'full'):
            if v in piv.columns:
                r[f'{v}_sec'] = piv[v]
                r[f'{v}/RBRP'] = piv[v] / piv['RBRP']
        print('\n=== refinement time relative to RBRP time ===')
        print(r.round(3).to_string())

    # log-log slopes of mean time versus n
    print('\n=== empirical scaling exponent: time ~ n^b (log-log fit over the n values available) ===')
    rows = []
    for ds in sorted(s.dataset.unique()):
        for v in ('RBRP', 'T10', 'full'):
            q = s[(s.dataset == ds) & (s.variant == v)].sort_values('n')
            if len(q) >= 2:
                rows.append({'dataset': ds, 'variant': v, 'n_values': list(q.n), 'exponent_time': slope(q.n, q.time_sec),
                             'exponent_n_eval': slope(q.n, q.n_eval) if v != 'RBRP' else np.nan,
                             'exponent_ms_per_eval': slope(q.n, q.ms_per_eval) if v != 'RBRP' else np.nan})
    e = pd.DataFrame(rows)
    e.to_csv(out / 'runtime_scaling_exponents.csv', index=False)
    print(e.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
        for ax, ds in zip(axes.ravel(), REPRESENTATIVE):
            for v, mk in (('RBRP', 'k^-'), ('T10', 'o--'), ('full', 's-')):
                q = s[(s.dataset == ds) & (s.variant == v)].sort_values('n')
                if len(q):
                    ax.plot(q.n, q.time_sec, mk, label=v if v != 'T10' else f'T={T_BUDGET}')
            ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=0.3, which='both')
            ax.set_title(f'{ds} (k={REPRESENTATIVE[ds]})', fontsize=9); ax.set_xlabel('n'); ax.set_ylabel('mean time (s)')
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / 'runtime_scaling.png', dpi=160)
    except Exception as ex:
        print('plot skipped:', ex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'stage3_runtime_results'))
    ap.add_argument('--ref-csv', default=str(ROOT / 'results' / 'stage2_sorted_scan_results' / 'raw_results.csv'), help='raw_results.csv from stage2_sorted_scan.py (n=500 reproducibility check)')
    ap.add_argument('--seeds', type=int, default=5, help='seeds for n <= 1000')
    ap.add_argument('--large-seeds', type=int, default=2, help='seeds for n >= 2000')
    ap.add_argument('--full-max-n', type=int, default=2000, help='run the full scan only for n <= this')
    ap.add_argument('--only', default=None)
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ns = N_LIST
    if a.smoke:
        ns, a.seeds, a.large_seeds, a.full_max_n = [500, 1000], 1, 1, 500
    if not a.summarize_only:
        base = load_base()
        mod = base.load_stage1(Path(a.stage1))
        t0 = time.perf_counter()
        for ds in REPRESENTATIVE:
            if a.only and ds != a.only:
                continue
            print(f'=== {ds} (k={REPRESENTATIVE[ds]}), n in {ns} ===', flush=True)
            run_dataset(base, mod, ds, ns, a.seeds, a.large_seeds, a.full_max_n, out)
        print(f'\nwall time {time.perf_counter() - t0:.0f}s')
    summarize(out, a.ref_csv)


if __name__ == '__main__':
    main()