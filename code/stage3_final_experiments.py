"""Stage 3.6: the remaining experiments in one script, to settle what the contribution is.

All methods start from the SAME RBRP skeleton S0 and are measured with the exact A = ||W||_2 and E on the actual matrix.

  RBRP            the skeleton S0 itself
  QRCP            QR with column pivoting on X^T (real matrices only; deterministic)
  Vol(f=1)        locally maximal volume exchange (maxvol / Dominant / strong RRQR with f=1), NO error control
  VolGuard-500    same swaps, accepted only if exact E <= 1.05*E0; up to 500 candidates per iteration
                  (the earlier version used 25; this removes most of that handicap)
  Ours-T10        proposed refinement, guard E <= 1.05*E0, at most 10 rows per round
  Ours-T10-noG    proposed refinement WITHOUT error guard (isolates the objective: spectral norm vs volume)
  Hybrid-T10      VolGuard-500 first, then Ours-T10 (guard still relative to the RBRP error E0)
  Ours-full       proposed refinement, full scan: reused from earlier result files where available, computed for the
                  missing seeds on Burgers (all k) and digits (k=10 only); synthetic matrices are reused (no new compute)

Matrices: 4 synthetic datasets at their representative k (data seeds 0..seeds-1, RBRP seed formula of Section 7),
Burgers snapshots (k=8,16,24) and digits (k=10,20,30) (RBRP seeds 0..seeds-1).

Outputs (in --out): final_raw_<matrix>.csv, final_summary.csv, final_paired.csv, final_dominance.csv, final_scatter.png.

Usage (next to stage1.py, stage2_sorted_scan.py, stage3_realistic_baseline.py, stage3_volume_baseline.py):
    python stage3_final_experiments.py --smoke
    python stage3_final_experiments.py                       # everything (about 1 hour; use --only to parallelize)
    python stage3_final_experiments.py --only burgers_snapshots
    python stage3_final_experiments.py --summarize-only
"""
from __future__ import annotations
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # repository root (results/ lives here)
GUARD, T_BUDGET, CANDS = 1.05, 10, 500
METHODS = ['RBRP', 'QRCP', 'Vol(f=1)', 'VolGuard-500', 'Ours-T10', 'Ours-T10-noG', 'Hybrid-T10', 'Ours-full']
PAIRS = [('Ours-T10', 'Vol(f=1)'), ('Ours-T10', 'VolGuard-500'), ('Hybrid-T10', 'VolGuard-500'), ('Hybrid-T10', 'Ours-T10'),
         ('Hybrid-T10', 'Vol(f=1)'), ('Ours-T10-noG', 'Vol(f=1)'), ('Ours-full', 'VolGuard-500'), ('Ours-full', 'Hybrid-T10'),
         ('Ours-full', 'Vol(f=1)')]
FULL_KS = {'burgers_snapshots': None, 'digits': [10]}          # None = all ranks


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def load_old_full(baseline_dir, raw_csv):
    """Earlier full-scan results keyed by (dataset, k, seed) -> (A, E, time)."""
    old = {}
    for f in glob.glob(str(Path(baseline_dir) / 'baseline_raw_*.csv')):
        b = pd.read_csv(f)
        for _, r in b[b.method == 'RBRP+full'].iterrows():
            old[(r.dataset, int(r.k), int(r.seed))] = (r.A, r.E, r.time_sec)
    if Path(raw_csv).exists():
        K = {'nearly_collinear_clusters': 4, 'repeated_directions': 4, 'localized_svd': 14, 'kernel_snapshots': 16}
        r0 = pd.read_csv(raw_csv)
        for _, r in r0[r0.method == 'sorted_full'].iterrows():
            old[(r.dataset, K[r.dataset], int(r.seed))] = (r.Af, r.Ef, r.time_sec)
    return old


def run_matrix(base, mod, rb, vb, name, kind, ks, seeds, full_seeds, old_full, smoke, out):
    rows, t_ds = [], time.perf_counter()
    if kind == 'real':
        X0 = rb.get_matrix(name, smoke)
        t = time.perf_counter(); Usvd = np.linalg.svd(X0, full_matrices=False)[0]; t_svd0 = time.perf_counter() - t
    for seed in range(seeds):
        if kind == 'synthetic':
            di = list(base.DATASET_K).index(name)
            X = getattr(mod, base.BUILDERS[name])(n=500, d=60, seed=seed)
            rbrp_seed = 1_000_000 + di * 100_000 + seed
            t = time.perf_counter(); U_all = np.linalg.svd(X, full_matrices=False)[0]; t_svd = time.perf_counter() - t
        else:
            X, rbrp_seed, U_all, t_svd = X0, seed, Usvd, t_svd0
        for k in ks:
            if kind == 'real' and seed == 0:
                (Q, R, piv) = sla.qr(X0.T, mode='economic', pivoting=True)
                dq = mod.compute_id_diagnostics(X0, np.asarray(piv[:k], dtype=int))
                rows.append({'dataset': name, 'k': k, 'seed': -1, 'method': 'QRCP', 'A': float(dq['amp2']), 'E': float(dq['error_rel'])})
            U = U_all[:, :k]
            row_perm, *_ = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=rbrp_seed)
            S0 = np.asarray(row_perm[:k], dtype=int)
            d0 = mod.compute_id_diagnostics(X, S0)
            if not d0['valid']:
                continue
            E0, A0 = float(d0['error_rel']), float(d0['amp2'])

            def rec(method, S_or_d, t_el, extra=None):
                d = S_or_d if isinstance(S_or_d, dict) else mod.compute_id_diagnostics(X, S_or_d)
                r = {'dataset': name, 'k': k, 'seed': seed, 'method': method, 'A': float(d['amp2']) if d['valid'] else np.nan,
                     'E': float(d['error_rel']) if d['valid'] else np.nan, 'A0': A0, 'E0': E0, 'time_sec': t_el}
                if extra: r.update(extra)
                rows.append(r)

            rec('RBRP', d0, 0.0)
            t = time.perf_counter(); S, sw, conv = vb.volume_exchange(mod, X, U, S0, 1.0); t_v = time.perf_counter() - t
            rec('Vol(f=1)', S, t_v + t_svd, {'swaps': sw})
            vb.GUARD_CANDS = CANDS
            t = time.perf_counter(); Sg, sw, conv = vb.volume_exchange(mod, X, U, S0, 1.0, E0=E0, guard=GUARD); t_g = time.perf_counter() - t
            rec('VolGuard-500', Sg, t_g + t_svd, {'swaps': sw})
            t = time.perf_counter(); Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, max_scan=T_BUDGET); t_o = time.perf_counter() - t
            rec('Ours-T10', df, t_o, {'swaps': sum(h['accepted'] for h in hist), 'n_eval': ne})
            t = time.perf_counter(); Sn, dn, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, error_guard=np.inf, max_scan=T_BUDGET); t_n = time.perf_counter() - t
            rec('Ours-T10-noG', dn, t_n, {'swaps': sum(h['accepted'] for h in hist), 'n_eval': ne})
            t = time.perf_counter(); Sh, dh, hist, ne, why = base.sorted_scan_refinement(mod, X, Sg, E0, max_scan=T_BUDGET); t_h = time.perf_counter() - t
            rec('Hybrid-T10', dh, t_g + t_svd + t_h, {'swaps': sum(h['accepted'] for h in hist), 'n_eval': ne})
            # full scan: reuse where available, compute where the policy asks for it
            key = (name, k, seed)
            if key in old_full:
                a, e, tt = old_full[key]
                rows.append({'dataset': name, 'k': k, 'seed': seed, 'method': 'Ours-full', 'A': a, 'E': e, 'A0': A0, 'E0': E0, 'time_sec': tt, 'reused': True})
            elif kind == 'real' and seed < full_seeds and (FULL_KS[name] is None or k in FULL_KS[name]):
                t = time.perf_counter(); Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, max_scan=None); t_f = time.perf_counter() - t
                rec('Ours-full', df, t_f, {'n_eval': ne, 'stop_reason': why})
        print(f'  [{name}] seed {seed + 1}/{seeds} done, elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out / f'final_raw_{name}.csv', index=False)


def summarize(out):
    files = sorted(glob.glob(str(out / 'final_raw_*.csv')))
    if not files:
        print('no raw files'); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    r0 = df[df.method == 'RBRP'][['dataset', 'k', 'seed', 'A', 'E']].rename(columns={'A': 'A0_', 'E': 'E0_'})
    df = df.merge(r0, on=['dataset', 'k', 'seed'], how='left')
    df['rho_A'], df['rho_E'] = df.A / df.A0_, df.E / df.E0_
    df['guard_ok'] = df.rho_E <= GUARD + 1e-9
    s = df.groupby(['dataset', 'k', 'method'], sort=False).agg(
        runs=('A', 'size'), median_A=('A', 'median'), mean_rho_A=('rho_A', 'mean'), mean_rho_E=('rho_E', 'mean'),
        max_rho_E=('rho_E', 'max'), guard_ok=('guard_ok', 'mean'), mean_time_sec=('time_sec', 'mean')).reset_index()
    s['o'] = s.method.map({m: i for i, m in enumerate(METHODS)})
    s = s.sort_values(['dataset', 'k', 'o'], kind='stable').drop(columns='o')
    s.to_csv(out / 'final_summary.csv', index=False)
    print('\n=== per matrix and k: A = ||W||_2, rho_A = A/A0, rho_E = E/E0, guard_ok = share with E <= 1.05 E0 ===')
    print(s.round(3).to_string(index=False))

    pa = df.pivot_table(index=['dataset', 'k', 'seed'], columns='method', values='A')
    pe = df.pivot_table(index=['dataset', 'k', 'seed'], columns='method', values='E')
    rows, dom, pooled = [], [], []
    for x, y in PAIRS:
        if x not in pa or y not in pa:
            continue
        q = pd.DataFrame({'a': pa[x] / pa[y], 'e': pe[x] / pe[y]}).dropna()
        for (ds, k), z in q.groupby(level=[0, 1], sort=False):
            rows.append({'x': x, 'y': y, 'dataset': ds, 'k': k, 'pairs': len(z), 'x_A_lower': (z.a < 1 - 1e-9).mean(),
                         'median_A_ratio': z.a.median(), 'median_E_ratio': z.e.median()})
            x_dom = ((z.a <= 1 + 1e-9) & (z.e <= 1 + 1e-9) & ((z.a < 1 - 1e-9) | (z.e < 1 - 1e-9))).mean()
            y_dom = ((z.a >= 1 - 1e-9) & (z.e >= 1 - 1e-9) & ((z.a > 1 + 1e-9) | (z.e > 1 + 1e-9))).mean()
            dom.append({'x': x, 'y': y, 'dataset': ds, 'k': k, 'pairs': len(z), 'x_dominates': x_dom, 'y_dominates': y_dom})
        pooled.append({'x': x, 'y': y, 'pairs': len(q), 'x_A_lower': (q.a < 1 - 1e-9).mean(), 'median_A_ratio': q.a.median(),
                       'median_E_ratio': q.e.median(),
                       'x_dominates': ((q.a <= 1 + 1e-9) & (q.e <= 1 + 1e-9) & ((q.a < 1 - 1e-9) | (q.e < 1 - 1e-9))).mean(),
                       'y_dominates': ((q.a >= 1 - 1e-9) & (q.e >= 1 - 1e-9) & ((q.a > 1 + 1e-9) | (q.e > 1 + 1e-9))).mean()})
    pr, dm, pl = pd.DataFrame(rows), pd.DataFrame(dom), pd.DataFrame(pooled)
    pr.to_csv(out / 'final_paired.csv', index=False); dm.to_csv(out / 'final_dominance.csv', index=False); pl.to_csv(out / 'final_pooled.csv', index=False)
    print('\n=== paired, per matrix (ratio x/y < 1 favours x) ===')
    print(pr.round(3).to_string(index=False))
    print('\n=== POOLED over all matrices and seeds: A ratio, E ratio and Pareto dominance (x dominates y = no worse in both A and E, better in one) ===')
    print(pl.round(3).to_string(index=False))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = list(dict.fromkeys(zip(df.dataset, df.k)))
        show = ['Vol(f=1)', 'VolGuard-500', 'Ours-T10', 'Ours-T10-noG', 'Hybrid-T10', 'Ours-full']
        mk = dict(zip(show, ['s', 'D', 'o', 'x', '^', '*']))
        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        for ax, (ds, k) in zip(axes.ravel(), keys):
            sub = df[(df.dataset == ds) & (df.k == k)]
            for m in show:
                z = sub[sub.method == m]
                if len(z):
                    ax.scatter(z.rho_E, z.rho_A, s=18, marker=mk[m], alpha=.7, label=m)
            ax.axvline(GUARD, color='gray', ls='--', lw=1); ax.set_title(f'{ds}, k={k}', fontsize=8)
            ax.set_xlabel('E / E0'); ax.set_ylabel('A / A0'); ax.grid(alpha=.3)
        axes[0, 0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(out / 'final_scatter.png', dpi=150)
    except Exception as e:
        print('plot skipped:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'stage3_final_results'))
    ap.add_argument('--baseline-dir', default=str(ROOT / 'results' / 'stage3_baseline_results'))
    ap.add_argument('--raw-csv', default=str(ROOT / 'results' / 'stage2_sorted_scan_results' / 'raw_results.csv'))
    ap.add_argument('--seeds', type=int, default=20, help='seeds for all methods except the full scan')
    ap.add_argument('--full-seeds', type=int, default=10, help='full scan on real matrices for the first N seeds (reused when available)')
    ap.add_argument('--only', default=None)
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.seeds, a.full_seeds = 2, 1
    if not a.summarize_only:
        base = load_module('stage2_sorted_scan', 'stage2_sorted_scan.py')
        rb = load_module('stage3_realistic_baseline', 'stage3_realistic_baseline.py')
        vb = load_module('stage3_volume_baseline', 'stage3_volume_baseline.py')
        mod = base.load_stage1(Path(a.stage1))
        old = {} if a.smoke else load_old_full(a.baseline_dir, a.raw_csv)
        if a.smoke:                                            # smoke uses smaller matrices: reuse only the synthetic full scans
            old = {k: v for k, v in load_old_full(a.baseline_dir, a.raw_csv).items() if k[0] in base.DATASET_K}
        t0 = time.perf_counter()
        for ds, k in base.DATASET_K.items():
            if a.only and a.only != ds:
                continue
            print(f'=== {ds} (synthetic, k={k}) ===', flush=True)
            run_matrix(base, mod, rb, vb, ds, 'synthetic', [k], a.seeds, a.full_seeds, old, a.smoke, out)
        for name, ks in rb.KS.items():
            if a.only and a.only != name:
                continue
            print(f'=== {name} (real/snapshot) ===', flush=True)
            run_matrix(base, mod, rb, vb, name, 'real', ks[:1] if a.smoke else ks, a.seeds, a.full_seeds, old, a.smoke, out)
        print(f'\nwall time {time.perf_counter() - t0:.0f}s')
    summarize(out)


if __name__ == '__main__':
    main()