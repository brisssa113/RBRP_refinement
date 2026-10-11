"""Stage 3.5: baseline "RBRP + volume-based exchange" versus the proposed amplification-aware refinement.

Why: strong RRQR (Gu-Eisenstat), maxvol (Goreinov et al.) and the Dominant algorithm (Osinsky) improve a skeleton by
one-for-one exchanges that increase the VOLUME of the selected submatrix. Reviewers will ask how that compares with
exchanges that directly reduce A = ||W||_2 under an error budget.

Volume baseline (rows version).  U = top-k left singular vectors of X (n x k, orthonormal).  For a skeleton S,
    A_hat = U U_S^{-1}.   Swapping row i into the place of skeleton row s_j multiplies the volume of U_S by |A_hat[i,j]|,
and A_hat is updated by a rank-one formula.  Algorithm: swap the pair (i,j) with the largest |A_hat[i,j]| while it exceeds f.
    f = 1 : locally maximal volume (maxvol / Dominant with c = 1 / strong RRQR with f = 1)
    f = 2 : cheaper, weaker guarantee (Gu-Eisenstat style, f = 2)
These swaps use only the orthonormal basis and know nothing about the error E.  After the swaps, the EXACT A and E of
the resulting skeleton on the actual matrix X are measured, with the same metrics as before.

Variants:
    Vol(f=1), Vol(f=2)     : unguarded volume exchange starting from the RBRP skeleton
    VolGuard(f=1)          : same, but a swap is accepted only if the exact error stays <= 1.05*E0
                             (candidates tried in decreasing |A_hat[i,j]|, at most 25 per iteration)
    Ours-T10, Ours-full    : the proposed refinement (T10 recomputed; the full scan is merged from earlier result files)

Matrices (same protocol as before): the 4 synthetic datasets at the representative k (data seeds 0..9, RBRP seed formula
of Section 7), Burgers snapshots and digits at k = 8,16,24 / 10,20,30 (RBRP seeds 0..9).

Usage (next to stage1.py, stage2_sorted_scan.py, stage3_realistic_baseline.py):
    python stage3_volume_baseline.py --smoke
    python stage3_volume_baseline.py --baseline-dir stage3_baseline_results --raw-csv stage2_sorted_scan_results/raw_results.csv
    python stage3_volume_baseline.py --summarize-only --baseline-dir ... --raw-csv ...
"""
from __future__ import annotations
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
GUARD = 1.05
T_BUDGET = 10
GUARD_CANDS = 25
METHODS = ['RBRP', 'QRCP', 'Vol(f=1)', 'Vol(f=2)', 'VolGuard(f=1)', 'Ours-T10', 'Ours-full']


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def volume_exchange(mod, X, U, S0, f, E0=None, guard=None, max_iter=3000):
    """Swap the pair with the largest |A_hat[i,j]| while it exceeds f.  If guard is given, a swap is accepted only
    if the exact error E(S') <= guard*E0 (up to GUARD_CANDS candidates per iteration, in decreasing |A_hat|)."""
    n, k = U.shape
    S = np.asarray(S0, dtype=int).copy()

    def build(S):
        return np.linalg.solve(U[S].T, U.T).T          # U @ inv(U_S)

    try:
        A = build(S)
    except np.linalg.LinAlgError:
        return S, 0, False
    mask = np.ones(n, bool); mask[S] = False
    swaps, converged = 0, False
    for it in range(max_iter):
        if it > 0 and it % 50 == 0:
            A = build(S)
        B = np.abs(A); B[~mask] = 0.0
        if guard is None:
            idx = int(np.argmax(B)); i, j = divmod(idx, k)
            if B[i, j] <= f + 1e-9:
                converged = True; break
        else:
            flat = B.ravel()
            top = np.argpartition(-flat, min(GUARD_CANDS, flat.size - 1))[:GUARD_CANDS]
            top = top[np.argsort(-flat[top])]
            chosen = None
            for idx in top:
                i, j = divmod(int(idx), k)
                if B[i, j] <= f + 1e-9:
                    break
                S_try = S.copy(); S_try[j] = i
                d = mod.compute_id_diagnostics(X, S_try)
                if d['valid'] and d['error_rel'] <= guard * E0:
                    chosen = (i, j); break
            if chosen is None:
                converged = True; break
            i, j = chosen
        piv = A[i, j]
        col = A[:, j].copy(); row = A[i, :].copy(); row[j] -= 1.0
        A -= np.outer(col, row) / piv
        mask[S[j]] = True; mask[i] = False
        S[j] = i; swaps += 1
    return S, swaps, converged


def run_matrix(base, mod, rb, name, ds_kind, ks, seeds, smoke, rows, out):
    """ds_kind: 'synthetic' (name = dataset key) or 'real' (name in rb.KS)."""
    t_ds = time.perf_counter()
    if ds_kind == 'real':
        X0 = rb.get_matrix(name, smoke)
        t = time.perf_counter(); Usvd = np.linalg.svd(X0, full_matrices=False)[0]; t_svd0 = time.perf_counter() - t
    for k in ks:
        if ds_kind == 'real':
            (Q, R, piv) = sla.qr(X0.T, mode='economic', pivoting=True)
            dq = mod.compute_id_diagnostics(X0, np.asarray(piv[:k], dtype=int))
            rows.append({'dataset': name, 'k': k, 'seed': -1, 'method': 'QRCP', 'A': float(dq['amp2']),
                         'E': float(dq['error_rel'])})
        for seed in range(seeds):
            if ds_kind == 'synthetic':
                di = list(base.DATASET_K).index(name)
                X = getattr(mod, base.BUILDERS[name])(n=500, d=60, seed=seed)
                rbrp_seed = 1_000_000 + di * 100_000 + seed
                t = time.perf_counter(); U_all = np.linalg.svd(X, full_matrices=False)[0]; t_svd = time.perf_counter() - t
            else:
                X, rbrp_seed, U_all, t_svd = X0, seed, Usvd, t_svd0
            U = U_all[:, :k]
            row_perm, *_ = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=rbrp_seed)
            S0 = np.asarray(row_perm[:k], dtype=int)
            d0 = mod.compute_id_diagnostics(X, S0)
            if not d0['valid']:
                continue
            E0, A0 = float(d0['error_rel']), float(d0['amp2'])
            rows.append({'dataset': name, 'k': k, 'seed': seed, 'method': 'RBRP', 'A': A0, 'E': E0, 'A0': A0, 'E0': E0})
            specs = [('Vol(f=1)', dict(f=1.0)), ('Vol(f=2)', dict(f=2.0)), ('VolGuard(f=1)', dict(f=1.0, guard=GUARD, E0=E0))]
            for mname, kw in specs:
                t = time.perf_counter()
                S, sw, conv = volume_exchange(mod, X, U, S0, kw['f'], E0=kw.get('E0'), guard=kw.get('guard'))
                el = time.perf_counter() - t
                d = mod.compute_id_diagnostics(X, S)
                rows.append({'dataset': name, 'k': k, 'seed': seed, 'method': mname, 'A': float(d['amp2']) if d['valid'] else np.nan,
                             'E': float(d['error_rel']) if d['valid'] else np.nan, 'A0': A0, 'E0': E0, 'swaps': sw,
                             'converged': conv, 'time_sec': el + t_svd})
            t = time.perf_counter()
            Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, max_scan=T_BUDGET)
            rows.append({'dataset': name, 'k': k, 'seed': seed, 'method': 'Ours-T10', 'A': float(df['amp2']),
                         'E': float(df['error_rel']), 'A0': A0, 'E0': E0, 'swaps': sum(h['accepted'] for h in hist),
                         'time_sec': time.perf_counter() - t})
        print(f'  [{name}] k={k} done, elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out / 'volume_baseline_raw.csv', index=False)


def merge_full(df, baseline_dir, raw_csv):
    """Add 'Ours-full' rows (and check S0 consistency) from earlier result files."""
    extra, chk = [], []
    if baseline_dir:
        for f in glob.glob(str(Path(baseline_dir) / 'baseline_raw_*.csv')):
            b = pd.read_csv(f)
            b = b[b.get('valid', True) == True]
            full = b[b.method == 'RBRP+full']
            for _, r in full.iterrows():
                extra.append({'dataset': r.dataset, 'k': int(r.k), 'seed': int(r.seed), 'method': 'Ours-full', 'A': r.A,
                              'E': r.E, 'time_sec': r.time_sec})
            for _, r in b[b.method == 'RBRP'].iterrows():
                chk.append((r.dataset, int(r.k), int(r.seed), r.A, r.E))
    if raw_csv and Path(raw_csv).exists():
        r0 = pd.read_csv(raw_csv)
        K = {'nearly_collinear_clusters': 4, 'repeated_directions': 4, 'localized_svd': 14, 'kernel_snapshots': 16}
        for _, r in r0[r0.method == 'sorted_full'].iterrows():
            extra.append({'dataset': r.dataset, 'k': K[r.dataset], 'seed': int(r.seed), 'method': 'Ours-full', 'A': r.Af,
                          'E': r.Ef, 'time_sec': r.time_sec})
            chk.append((r.dataset, K[r.dataset], int(r.seed), r.A0, r.E0))
    if extra:
        keys = set(zip(df.dataset, df.k, df.seed))
        extra = [e for e in extra if (e['dataset'], e['k'], e['seed']) in keys]
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    if chk:
        c = pd.DataFrame(chk, columns=['dataset', 'k', 'seed', 'A_old', 'E_old'])
        m = df[df.method == 'RBRP'].merge(c, on=['dataset', 'k', 'seed'])
        if len(m):
            print(f'S0 consistency with earlier results: max rel |A diff| = {np.max(np.abs(m.A - m.A_old) / m.A_old):.2e}, '
                  f'max rel |E diff| = {np.max(np.abs(m.E - m.E_old) / m.E_old):.2e} over {len(m)} runs')
    return df


def summarize(out, baseline_dir, raw_csv):
    df = pd.read_csv(out / 'volume_baseline_raw.csv')
    df = merge_full(df, baseline_dir, raw_csv)
    # attach RBRP E0/A0 to every run (needed for ratios)
    r0 = df[df.method == 'RBRP'][['dataset', 'k', 'seed', 'A', 'E']].rename(columns={'A': 'A0_', 'E': 'E0_'})
    df = df.merge(r0, on=['dataset', 'k', 'seed'], how='left')
    df['A_over_A0'] = df.A / df.A0_
    df['E_over_E0'] = df.E / df.E0_
    df['guard_ok'] = df.E_over_E0 <= GUARD + 1e-9
    g = df.groupby(['dataset', 'k', 'method'], sort=False)
    s = g.agg(runs=('A', 'size'), median_A=('A', 'median'), mean_rho_A=('A_over_A0', 'mean'), mean_rho_E=('E_over_E0', 'mean'),
              max_rho_E=('E_over_E0', 'max'), guard_ok=('guard_ok', 'mean'), median_swaps=('swaps', 'median'),
              mean_time_sec=('time_sec', 'mean')).reset_index()
    s['o'] = s.method.map({m: i for i, m in enumerate(METHODS)})
    s = s.sort_values(['dataset', 'k', 'o'], kind='stable').drop(columns='o')
    s.to_csv(out / 'volume_baseline_summary.csv', index=False)
    print('\n=== per matrix and k (A = ||W||_2, rho_A = A/A0, rho_E = E/E0; guard_ok = share with E <= 1.05 E0) ===')
    print(s.round(3).to_string(index=False))

    # paired comparison of Ours-T10 / Ours-full versus each volume variant
    p = df.pivot_table(index=['dataset', 'k', 'seed'], columns='method', values='A')
    pe = df.pivot_table(index=['dataset', 'k', 'seed'], columns='method', values='E')
    rows = []
    for ours in ('Ours-T10', 'Ours-full'):
        for vol in ('Vol(f=1)', 'Vol(f=2)', 'VolGuard(f=1)'):
            if ours in p and vol in p:
                q = pd.DataFrame({'a': p[ours] / p[vol], 'e': pe[ours] / pe[vol]}).dropna()
                for (ds, k), z in q.groupby(level=[0, 1], sort=False):
                    rows.append({'ours': ours, 'vs': vol, 'dataset': ds, 'k': k, 'pairs': len(z),
                                 'ours_A_lower': (z.a < 1 - 1e-9).mean(), 'median_A_ratio': z.a.median(),
                                 'median_E_ratio': z.e.median()})
    pr = pd.DataFrame(rows)
    pr.to_csv(out / 'volume_baseline_paired.csv', index=False)
    print('\n=== paired: ours versus volume exchange (ratio < 1 means ours has the smaller value) ===')
    print(pr.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = list(dict.fromkeys(zip(df.dataset, df.k)))
        for col, ylab, fn, ref in (('A', 'A = ||W||_2', 'volume_baseline_A.png', None), ('E_over_E0', 'E / E0', 'volume_baseline_E.png', GUARD)):
            fig, axes = plt.subplots(2, 5, figsize=(17, 6.5))
            for ax, (ds, k) in zip(axes.ravel(), keys):
                sub = df[(df.dataset == ds) & (df.k == k)]
                ms = [m for m in METHODS if m in set(sub.method) and not (col == 'E_over_E0' and m == 'QRCP')]
                ax.boxplot([sub[sub.method == m][col].dropna().values for m in ms])
                ax.set_xticklabels([m.replace('VolGuard', 'VG').replace('Ours-', 'O-').replace('Vol(', 'V(') for m in ms], rotation=45, fontsize=7)
                if ref: ax.axhline(ref, color='gray', ls='--', lw=1)
                if col == 'A': ax.set_yscale('log')
                ax.set_title(f'{ds}, k={k}', fontsize=8); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(out / fn, dpi=150)
    except Exception as e:
        print('plot skipped:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(HERE / 'stage3_volume_baseline_results'))
    ap.add_argument('--baseline-dir', default=str(HERE / 'stage3_baseline_results'))
    ap.add_argument('--raw-csv', default=str(HERE / 'stage2_sorted_scan_results' / 'raw_results.csv'))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.seeds = 2
    if not a.summarize_only:
        base = load_module('stage2_sorted_scan', 'stage2_sorted_scan.py')
        rb = load_module('stage3_realistic_baseline', 'stage3_realistic_baseline.py')
        mod = base.load_stage1(Path(a.stage1))
        rows, t0 = [], time.perf_counter()
        for ds, k in base.DATASET_K.items():
            print(f'=== {ds} (synthetic, k={k}) ===', flush=True)
            run_matrix(base, mod, rb, ds, 'synthetic', [k], a.seeds, a.smoke, rows, out)
        for name, ks in rb.KS.items():
            print(f'=== {name} (real/snapshot) ===', flush=True)
            run_matrix(base, mod, rb, name, 'real', ks[:1] if a.smoke else ks, a.seeds, a.smoke, rows, out)
        pd.DataFrame(rows).to_csv(out / 'volume_baseline_raw.csv', index=False)
        print(f'\nwall time {time.perf_counter() - t0:.0f}s')
    summarize(out, a.baseline_dir, a.raw_csv)


if __name__ == '__main__':
    main()