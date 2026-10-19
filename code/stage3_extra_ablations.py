"""Stage 3.4: two small ablations on the realistic matrices of stage3_realistic_baseline.py.

(a) Refine the QRCP skeleton.  Start from the first k pivots of QR with column pivoting (deterministic),
    guard E <= 1.05 * E(QRCP).  Variants: T=10 (always) and the full scan (only where it is affordable:
    Burgers all k, digits k=10; a single deterministic run each).
    Question: is the refinement skeleton-agnostic, and does it improve on QRCP itself?

(b) Error-guard ablation.  Refine every RBRP skeleton (same 10 RBRP seeds as the baseline script, T=10)
    with guard factor g in {1.00, 1.05, 1.20, inf}:  E(S) <= g * E0.   g = 1.05 is the proposed setting
    (it must reproduce RBRP+T10 of stage3_realistic_baseline.py), g = inf means no error constraint.
    Question: what does the hard constraint cost in amplification, and what does it protect in error?

Usage (next to stage1.py, stage2_sorted_scan.py, stage3_realistic_baseline.py):
    python stage3_extra_ablations.py --smoke
    python stage3_extra_ablations.py --baseline-dir stage3_baseline_results
    python stage3_extra_ablations.py --only digits
    python stage3_extra_ablations.py --summarize-only
"""
from __future__ import annotations
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # repository root (results/ lives here)
GUARDS = [1.00, 1.05, 1.20, np.inf]
T_BUDGET = 10


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def gname(g):
    return 'inf' if not np.isfinite(g) else f'{g:.2f}'


def run_dataset(base, rb, mod, name, ks, seeds, out, smoke):
    X = rb.get_matrix(name, smoke)
    n, d = X.shape
    print(f'=== {name}: X is {n} x {d}, k in {ks} ===', flush=True)
    rows = []
    t_ds = time.perf_counter()
    for k in ks:
        # ---- (a) QRCP skeleton and its refinement
        Q, R, piv = sla.qr(X.T, mode='economic', pivoting=True)
        S_qr = np.asarray(piv[:k], dtype=int)
        dq = mod.compute_id_diagnostics(X, S_qr)
        E_qr, A_qr = float(dq['error_rel']), float(dq['amp2'])
        rows.append({'dataset': name, 'k': k, 'part': 'a', 'method': 'QRCP', 'seed': -1, 'guard': 'n/a',
                     'E': E_qr, 'A': A_qr, 'E_qr': E_qr, 'A_qr': A_qr})
        do_full = (rb.FULL_KS[name] is None or k in rb.FULL_KS[name] or smoke)
        for vname, T in [('QRCP+T10', T_BUDGET)] + ([('QRCP+full', None)] if do_full else []):
            t = time.perf_counter()
            Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S_qr, E_qr, max_scan=T)
            rows.append({'dataset': name, 'k': k, 'part': 'a', 'method': vname, 'seed': -1, 'guard': '1.05',
                         'E': float(df['error_rel']), 'A': float(df['amp2']), 'E0': E_qr, 'A0': A_qr,
                         'E_qr': E_qr, 'A_qr': A_qr, 'n_eval': ne, 'rounds': sum(h['accepted'] for h in hist),
                         'time_sec': time.perf_counter() - t, 'stop_reason': why})
            print(f'  [{name}] k={k} {vname}: A {A_qr:.3f} -> {float(df["amp2"]):.3f}, '
                  f'E {E_qr:.4f} -> {float(df["error_rel"]):.4f}, elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
        # ---- (b) guard ablation on RBRP skeletons
        for seed in range(seeds):
            res = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=seed)
            S0 = np.asarray(res[0][:k], dtype=int)
            d0 = mod.compute_id_diagnostics(X, S0)
            if not d0['valid']:
                continue
            E0, A0 = float(d0['error_rel']), float(d0['amp2'])
            for g in GUARDS:
                t = time.perf_counter()
                Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, error_guard=g, max_scan=T_BUDGET)
                rows.append({'dataset': name, 'k': k, 'part': 'b', 'method': 'RBRP+T10', 'seed': seed,
                             'guard': gname(g), 'E': float(df['error_rel']), 'A': float(df['amp2']),
                             'E0': E0, 'A0': A0, 'E_qr': E_qr, 'A_qr': A_qr, 'n_eval': ne,
                             'rounds': sum(h['accepted'] for h in hist), 'time_sec': time.perf_counter() - t,
                             'stop_reason': why})
            print(f'  [{name}] k={k} guard ablation, RBRP seed {seed + 1}/{seeds} done, '
                  f'elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out / f'ablation_raw_{name}.csv', index=False)          # checkpoint per k


def summarize(out, baseline_dir):
    files = sorted(glob.glob(str(out / 'ablation_raw_*.csv')))
    if not files:
        print('no raw files'); return
    df = pd.concat([pd.read_csv(f, dtype={'guard': str}) for f in files], ignore_index=True)   # keep '1.00' as text
    df['rho_A'] = df.A / df.A0
    df['rho_E'] = df.E / df.E0
    df['A_over_qr'] = df.A / df.A_qr
    df['E_over_qr'] = df.E / df.E_qr

    # ---------- (a)
    a = df[df.part == 'a']
    ta = a.pivot_table(index=['dataset', 'k'], columns='method', values=['A', 'E'], aggfunc='first')
    rowsA = []
    for (ds, k), _ in ta.iterrows():
        r = {'dataset': ds, 'k': k, 'A_QRCP': ta.loc[(ds, k), ('A', 'QRCP')], 'E_QRCP': ta.loc[(ds, k), ('E', 'QRCP')]}
        for v in ('QRCP+T10', 'QRCP+full'):
            if ('A', v) in ta.columns and pd.notna(ta.loc[(ds, k), ('A', v)]):
                r[f'A_{v}'] = ta.loc[(ds, k), ('A', v)]
                r[f'rho_A_{v}'] = r[f'A_{v}'] / r['A_QRCP']
                r[f'rho_E_{v}'] = ta.loc[(ds, k), ('E', v)] / r['E_QRCP']
        rowsA.append(r)
    sa = pd.DataFrame(rowsA)
    sa.to_csv(out / 'ablation_a_qrcp_refine.csv', index=False)
    print('\n=== (a) refining the QRCP skeleton (single deterministic run per row; guard 1.05 * E_QRCP) ===')
    print(sa.round(4).to_string(index=False))

    # ---------- (b)
    b = df[df.part == 'b'].copy()
    order = ['1.00', '1.05', '1.20', 'inf']
    g = b.groupby(['dataset', 'k', 'guard'])
    sb = g.agg(runs=('A', 'size'), median_A=('A', 'median'), mean_rho_A=('rho_A', 'mean'), mean_rho_E=('rho_E', 'mean'),
               max_rho_E=('rho_E', 'max'), median_A_over_QRCP=('A_over_qr', 'median'),
               median_E_over_QRCP=('E_over_qr', 'median'), mean_n_eval=('n_eval', 'mean'),
               mean_time_sec=('time_sec', 'mean')).reset_index()
    sb['o'] = sb.guard.map({x: i for i, x in enumerate(order)})
    sb = sb.sort_values(['dataset', 'k', 'o']).drop(columns='o')
    sb.to_csv(out / 'ablation_b_guard.csv', index=False)
    print('\n=== (b) error-guard ablation: RBRP skeletons refined with T=10, E <= g * E0 ===')
    print(sb.round(4).to_string(index=False))

    # paired effect of the guard: g=inf and g=1.00 versus g=1.05, same RBRP seed
    p = b.pivot_table(index=['dataset', 'k', 'seed'], columns='guard', values='A')
    pe = b.pivot_table(index=['dataset', 'k', 'seed'], columns='guard', values='E')
    if {'1.05', 'inf'} <= set(p.columns):
        t = pd.DataFrame({'A_inf/A_1.05': (p['inf'] / p['1.05']).groupby(['dataset', 'k']).median(),
                          'E_inf/E_1.05': (pe['inf'] / pe['1.05']).groupby(['dataset', 'k']).median(),
                          'A_1.00/A_1.05': (p['1.00'] / p['1.05']).groupby(['dataset', 'k']).median(),
                          'E_1.00/E_1.05': (pe['1.00'] / pe['1.05']).groupby(['dataset', 'k']).median()})
        t.to_csv(out / 'ablation_b_paired.csv')
        print('\n=== (b) paired medians relative to the proposed guard 1.05 (same RBRP seed) ===')
        print(t.round(4).to_string())

    # reproducibility versus the baseline script
    if baseline_dir:
        bf = sorted(glob.glob(str(Path(baseline_dir) / 'baseline_raw_*.csv')))
        if bf:
            ref = pd.concat([pd.read_csv(f) for f in bf], ignore_index=True)
            ref = ref[ref.method == 'RBRP+T10'][['dataset', 'k', 'seed', 'A']]
            new = b[b.guard == '1.05'][['dataset', 'k', 'seed', 'A']]
            m = ref.merge(new, on=['dataset', 'k', 'seed'], suffixes=('_base', '_new'))
            if len(m):
                print(f'\nreproducibility (guard 1.05 vs RBRP+T10 of the baseline script): '
                      f'max rel |A diff| = {np.max(np.abs(m.A_base - m.A_new) / m.A_base):.2e} over {len(m)} runs')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = sorted(b[['dataset', 'k']].drop_duplicates().itertuples(index=False))
        for col, ylab, fname, ref_col in (('A', 'A = ||W||_2', 'ablation_guard_A.png', 'A_qr'),
                                          ('E', 'relative error E', 'ablation_guard_E.png', 'E_qr')):
            fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
            for ax, (ds, k) in zip(axes.ravel(), keys):
                sub = b[(b.dataset == ds) & (b.k == k)]
                ax.boxplot([sub[sub.guard == x][col].values for x in order])
                ax.set_xticklabels([f'g={x}' for x in order])
                ax.axhline(sub[ref_col].iloc[0], color='gray', ls='--', lw=1, label='QRCP')
                ax.set_yscale('log'); ax.grid(alpha=0.3, which='both'); ax.set_title(f'{ds}, k={k}', fontsize=9)
                ax.set_ylabel(ylab)
            axes[0, 0].legend(fontsize=8)
            fig.tight_layout(); fig.savefig(out / fname, dpi=160)
    except Exception as e:
        print('plot skipped:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'stage3_ablation_results'))
    ap.add_argument('--baseline-dir', default=str(ROOT / 'results' / 'stage3_baseline_results'))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--only', default=None)
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
        t0 = time.perf_counter()
        for name, ks in rb.KS.items():
            if a.only and name != a.only:
                continue
            run_dataset(base, rb, mod, name, ks[:1] if a.smoke else ks, a.seeds, out, a.smoke)
        print(f'\nwall time {time.perf_counter() - t0:.0f}s')
    summarize(out, a.baseline_dir)


if __name__ == '__main__':
    main()