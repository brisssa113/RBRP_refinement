"""Stage 3.1: rank sensitivity of the ordered-scan refinement.

For each dataset several skeleton sizes k are tested (they always include the representative k of
Section 7, so the earlier results can be reproduced). For every (dataset, k, seed):
    S0 = first k RBRP pivots, with the SAME RBRP seed formula as stage2_sorted_scan.py
    variant 'T10'  : ordered scan, at most 10 rows per round
    variant 'full' : ordered scan, all rows (single-exchange local optimum)

The full scan for the representative k is NOT rerun when --full-csv (raw_results.csv from
stage2_sorted_scan.py) is given; it is merged from there. The T10 run is always done, and for the
representative k it must reproduce the earlier T10 result (printed as a check).

Latent ranks of the generators: collinear clusters 12, repeated directions 10, localized SVD 18,
kernel snapshots has no finite rank (smooth family).

Usage (next to stage1.py, stage2_sorted_scan.py):
    python stage3_rank_sweep.py --smoke --full-csv raw_results.csv
    python stage3_rank_sweep.py --full-csv raw_results.csv                       # all datasets
    python stage3_rank_sweep.py --full-csv raw_results.csv --only kernel_snapshots   # one dataset
    python stage3_rank_sweep.py --summarize-only                                  # merge + tables + plot
Several --only runs can be started in parallel terminals (set OMP_NUM_THREADS=1 first).
"""
from __future__ import annotations
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RANKS = {
    'nearly_collinear_clusters': [4, 6, 8, 12],
    'repeated_directions': [4, 6, 8, 10],
    'localized_svd': [6, 10, 14, 18],
    'kernel_snapshots': [8, 12, 16, 20],
}
REPRESENTATIVE = {'nearly_collinear_clusters': 4, 'repeated_directions': 4, 'localized_svd': 14, 'kernel_snapshots': 16}
T_BUDGET = 10


def load_base():
    spec = importlib.util.spec_from_file_location('stage2_sorted_scan', HERE / 'stage2_sorted_scan.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def run_dataset(base, mod, ds, seeds, ref, out, verify):
    di = list(base.DATASET_K).index(ds)
    builder = getattr(mod, base.BUILDERS[ds])
    rows = []
    t_ds = time.perf_counter()
    for seed in range(seeds):
        X = builder(n=500, d=60, seed=seed)
        for k in RANKS[ds]:
            rbrp_seed = 1_000_000 + di * 100_000 + seed          # identical formula to Section 7
            try:
                row_perm, *_ = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=rbrp_seed)
                S0 = np.asarray(row_perm[:k], dtype=int)
                d0 = mod.compute_id_diagnostics(X, S0)
                if not d0['valid']:
                    rows.append({'dataset': ds, 'seed': seed, 'k': k, 'variant': 'invalid_S0'})
                    continue
                E0, A0 = float(d0['error_rel']), float(d0['amp2'])
                variants = [('T10', T_BUDGET)]
                skip_full = ref is not None and k == REPRESENTATIVE[ds]
                if not skip_full and not args_no_full:
                    variants.append(('full', None))
                for name, T in variants:
                    t = time.perf_counter()
                    Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, max_scan=T)
                    el = time.perf_counter() - t
                    r = {'dataset': ds, 'seed': seed, 'k': k, 'variant': name, 'E0': E0, 'A0': A0,
                         'Ef': float(df['error_rel']), 'Af': float(df['amp2']),
                         'rho_A': float(df['amp2']) / A0, 'rho_E': float(df['error_rel']) / E0,
                         'guard_ok': bool(df['error_rel'] <= base.ERROR_GUARD * E0 + 1e-14),
                         'n_eval': ne, 'time_sec': el, 'rounds': sum(h['accepted'] for h in hist),
                         'stop_reason': why}
                    if name == 'full' and seed < verify:
                        r['verified_admissible_left'] = base.verify_local_optimum(mod, X, Sf, E0)
                    rows.append(r)
            except Exception as e:                                # keep the sweep alive; record the failure
                rows.append({'dataset': ds, 'seed': seed, 'k': k, 'variant': 'error', 'stop_reason': repr(e)[:200]})
        print(f'  [{ds}] seed {seed + 1}/{seeds} done, elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out / f'rank_sweep_raw_{ds}.csv', index=False)   # checkpoint every seed
    return pd.DataFrame(rows)


def summarize(out, ref_path, seeds_cap):
    files = sorted(glob.glob(str(out / 'rank_sweep_raw_*.csv')))
    if not files:
        print('no raw files found'); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    bad = df[df.variant.isin(['invalid_S0', 'error'])]
    if len(bad):
        print(f'WARNING: {len(bad)} rows invalid/error:\n', bad.groupby(['dataset', 'k', 'variant']).size().to_string())
    df = df[~df.variant.isin(['invalid_S0', 'error'])].copy()
    if ref_path:
        ref = pd.read_csv(ref_path)
        rf = ref[(ref.method == 'sorted_full') & ref.dataset.isin(df.dataset.unique())].copy()
        rf = rf[rf.apply(lambda r: r.k == REPRESENTATIVE[r.dataset], axis=1)]
        rf = rf[rf.seed <= df.seed.max()]
        rf['variant'] = 'full'
        keep = ['dataset', 'seed', 'k', 'variant', 'E0', 'A0', 'Ef', 'Af', 'rho_A', 'rho_E', 'guard_ok', 'n_eval', 'time_sec']
        df = pd.concat([df, rf[keep]], ignore_index=True)
        # reproducibility check on the representative k
        old = ref[ref.method == 'sorted_T10'][['dataset', 'seed', 'Af']]
        new = df[(df.variant == 'T10')][['dataset', 'seed', 'k', 'Af']]
        new = new[new.apply(lambda r: r.k == REPRESENTATIVE[r.dataset], axis=1)]
        m = old.merge(new, on=['dataset', 'seed'], suffixes=('_old', '_new'))
        if len(m):
            print(f'\nreproducibility (representative k, T=10 vs Section 7): max rel |Af diff| = '
                  f'{np.max(np.abs(m.Af_old - m.Af_new) / m.Af_old):.2e} over {len(m)} runs')
    df['improved'] = df.rho_A < 1 - 1e-9
    g = df.groupby(['dataset', 'k', 'variant'])
    s = g.agg(n=('rho_A', 'size'), median_A0=('A0', 'median'), mean_rho_A=('rho_A', 'mean'), se=('rho_A', se),
              frac_improved=('improved', 'mean'), mean_rho_E=('rho_E', 'mean'), guard_ok=('guard_ok', 'mean'),
              mean_n_eval=('n_eval', 'mean'), mean_sec=('time_sec', 'mean')).reset_index()
    s.to_csv(out / 'rank_sweep_summary.csv', index=False)
    df.to_csv(out / 'rank_sweep_all.csv', index=False)
    print('\n=== Rank sensitivity: per dataset, k, variant ===')
    print(s.round(4).to_string(index=False))
    p = s.pivot_table(index=['dataset', 'k'], columns='variant', values='mean_rho_A')
    if {'T10', 'full'} <= set(p.columns):
        p['T10/full'] = p['T10'] / p['full']
        print('\n=== mean rho_A: T10 vs full (ratio >1 means T10 is worse) ===')
        print(p.round(3).to_string())
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        dss = list(RANKS)
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
        for ax, ds in zip(axes.ravel(), dss):
            for v, mk in (('T10', 'o--'), ('full', 's-')):
                q = s[(s.dataset == ds) & (s.variant == v)].sort_values('k')
                if len(q):
                    ax.errorbar(q.k, q.mean_rho_A, yerr=q.se, fmt=mk, capsize=2, label=v if v == 'full' else f'T={T_BUDGET}')
            ax.axhline(1.0, color='gray', lw=0.8, ls=':')
            ax.set_title(ds, fontsize=9); ax.set_xlabel('skeleton size k'); ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel('mean rho_A'); axes[1, 0].set_ylabel('mean rho_A'); axes[0, 0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / 'rank_sweep_rho_A_vs_k.png', dpi=160)
    except Exception as e:
        print('plot skipped:', e)


args_no_full = False


def main():
    global args_no_full
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(HERE / 'stage3_rank_sweep_results'))
    ap.add_argument('--full-csv', default=None, help='raw_results.csv from stage2_sorted_scan.py')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--only', default=None, help='run a single dataset')
    ap.add_argument('--no-full', action='store_true', help='skip all full scans (T10 only)')
    ap.add_argument('--verify', type=int, default=1, help='brute-force verify local optimum for first N seeds')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    args_no_full = a.no_full
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.seeds = 2
    if not a.summarize_only:
        base = load_base()
        mod = base.load_stage1(Path(a.stage1))
        ref = pd.read_csv(a.full_csv) if a.full_csv else None
        t0 = time.perf_counter()
        for ds in RANKS:
            if a.only and ds != a.only:
                continue
            print(f'=== {ds}: k in {RANKS[ds]} ===', flush=True)
            run_dataset(base, mod, ds, a.seeds, ref, out, a.verify)
        print(f'\nsweep wall time {time.perf_counter() - t0:.0f}s')
    summarize(out, a.full_csv, a.seeds)


if __name__ == '__main__':
    main()