"""Stage 2.11: scan-budget sweep and ordering ablation.

For each (dataset, seed) the SAME S0 as in stage2_sorted_scan.py is used (same RBRP seed formula),
so results can be merged with the full-scan results already computed there.

Runs, for T in T_LIST:
    order = 'norm'   : non-skeleton rows sorted by ||W(i,:)||_2, largest first   (proposed)
    order = 'random' : non-skeleton rows in a uniformly random order             (ablation)
and each round examines at most the first T rows of that order (see Section 6.5 of the handover).

Optionally merges the full-scan reference from raw_results.csv (--full-csv), so the full scan is not rerun.

Usage (next to stage1.py and stage2_sorted_scan.py):
    python stage2_T_sweep.py --smoke --full-csv raw_results.csv
    python stage2_T_sweep.py --full-csv raw_results.csv
"""
from __future__ import annotations
import argparse, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T_LIST = [1, 2, 5, 10, 20, 50]


def load_base():
    spec = importlib.util.spec_from_file_location('stage2_sorted_scan', HERE / 'stage2_sorted_scan.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def refine(base, mod, X, S0, E0, T, order, rng, max_iter=500):
    """One-for-one refinement; each round tries at most T rows in the given order."""
    S = np.asarray(S0, dtype=int).copy()
    n, k = X.shape[0], len(S)
    cur = mod.compute_id_diagnostics(X, S)
    n_eval, rounds = 0, 0
    for _ in range(max_iter):
        if order == 'norm':
            W = base.interp_matrix(mod, X, S)
            rn = np.linalg.norm(W, axis=1)
            rn[S] = -np.inf
            idx = np.argsort(-rn, kind='stable')[: n - k]
        else:
            idx = rng.permutation(np.setdiff1d(np.arange(n), S))
        accepted = None
        for add_idx in idx[:T]:
            best = None
            for pos in range(k):
                S_try = S.copy()
                S_try[pos] = int(add_idx)
                d = mod.compute_id_diagnostics(X, S_try)
                n_eval += 1
                if not d['valid'] or d['error_rel'] > base.ERROR_GUARD * E0:
                    continue
                if not d['amp2'] < cur['amp2'] * (1.0 - base.REL_TOL):
                    continue
                key = (d['amp2'], d['error_rel'])
                if best is None or key < best[0]:
                    best = (key, S_try, d)
            if best is not None:
                accepted = best
                break
        if accepted is None:
            break
        _, S, cur = accepted
        rounds += 1
    return S, cur, n_eval, rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(HERE / 'stage2_T_sweep_results'))
    ap.add_argument('--full-csv', default=None, help='raw_results.csv from stage2_sorted_scan.py')
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.seeds = 2
    base = load_base()
    mod = base.load_stage1(Path(args.stage1))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows, t_all = [], time.perf_counter()
    for di, (ds, k) in enumerate(base.DATASET_K.items()):
        builder = getattr(mod, base.BUILDERS[ds])
        print(f'=== {ds} (k={k}) ===', flush=True)
        for seed in range(args.seeds):
            X = builder(n=500, d=60, seed=seed)
            rbrp_seed = 1_000_000 + di * 100_000 + seed       # identical to stage2_sorted_scan.py
            row_perm, *_ = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=rbrp_seed)
            S0 = np.asarray(row_perm[:k], dtype=int)
            d0 = mod.compute_id_diagnostics(X, S0)
            E0, A0 = float(d0['error_rel']), float(d0['amp2'])
            for order in ('norm', 'random'):
                for ti, T in enumerate(T_LIST):
                    rng = np.random.default_rng(3_000_000 + di * 100_000 + seed * 10 + ti)
                    t = time.perf_counter()
                    Sf, df, ne, rd = refine(base, mod, X, S0, E0, T, order, rng)
                    rows.append({'dataset': ds, 'seed': seed, 'k': k, 'order': order, 'T': T,
                                 'A0': A0, 'E0': E0, 'Af': float(df['amp2']), 'Ef': float(df['error_rel']),
                                 'rho_A': float(df['amp2']) / A0, 'rho_E': float(df['error_rel']) / E0,
                                 'guard_ok': bool(df['error_rel'] <= base.ERROR_GUARD * E0 + 1e-14),
                                 'n_eval': ne, 'rounds': rd, 'time_sec': time.perf_counter() - t})
        print(f'  done, elapsed {time.perf_counter() - t_all:.1f}s', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / 'T_sweep_raw.csv', index=False)

    # merge full-scan reference (norm order, T = infinity)
    full = None
    if args.full_csv:
        ref = pd.read_csv(args.full_csv)
        full = ref[ref.method == 'sorted_full'][['dataset', 'seed', 'rho_A', 'rho_E', 'n_eval', 'time_sec', 'Af', 'A0']].copy()
        full = full[full.seed < args.seeds]
        # sanity: T=10 norm must reproduce sorted_T10 from the earlier script exactly
        t10 = ref[ref.method == 'sorted_T10'][['dataset', 'seed', 'Af']].merge(
            df[(df.order == 'norm') & (df['T'] == 10)][['dataset', 'seed', 'Af']], on=['dataset', 'seed'], suffixes=('_old', '_new'))
        if len(t10):
            print(f'\nreproducibility check (norm, T=10 vs earlier sorted_T10): '
                  f'max |Af diff| / Af = {np.max(np.abs(t10.Af_old - t10.Af_new) / t10.Af_old):.2e} over {len(t10)} runs')

    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

    summ = df.groupby(['order', 'T']).agg(n=('rho_A', 'size'), mean_rho_A=('rho_A', 'mean'), se=('rho_A', se),
                                           mean_rho_E=('rho_E', 'mean'), guard_ok=('guard_ok', 'mean'),
                                           mean_n_eval=('n_eval', 'mean'), mean_sec=('time_sec', 'mean')).reset_index()
    if full is not None:
        summ = pd.concat([summ, pd.DataFrame([{'order': 'norm', 'T': np.inf, 'n': len(full),
                                               'mean_rho_A': full.rho_A.mean(), 'se': se(full.rho_A),
                                               'mean_rho_E': full.rho_E.mean(), 'guard_ok': 1.0,
                                               'mean_n_eval': full.n_eval.mean(), 'mean_sec': full.time_sec.mean()}])],
                         ignore_index=True)
    # Pareto flag among the proposed (norm-order) points: non-dominated in (mean_n_eval, mean_rho_A)
    nm = summ[summ.order == 'norm']
    summ['pareto_norm'] = False
    for i, r in nm.iterrows():
        dom = ((nm.mean_n_eval <= r.mean_n_eval) & (nm.mean_rho_A <= r.mean_rho_A) &
               ((nm.mean_n_eval < r.mean_n_eval) | (nm.mean_rho_A < r.mean_rho_A))).any()
        summ.loc[i, 'pareto_norm'] = not dom
    summ.to_csv(out / 'T_sweep_summary.csv', index=False)

    print('\n=== T sweep: pooled over datasets ===')
    print(summ.round(4).to_string(index=False))

    # paired ablation: random - norm at equal T
    p = df.pivot_table(index=['dataset', 'seed', 'T'], columns='order', values='rho_A').reset_index()
    p['diff'] = p['random'] - p['norm']
    print('\n=== ablation: rho_A(random order) - rho_A(norm order), paired on S0 ===')
    print(f'{"T":>4}{"mean diff":>11}{"SE":>9}{"norm better":>13}{"random better":>15}')
    for T, g in p.groupby('T'):
        print(f'{T:>4}{g["diff"].mean():>11.4f}{se(g["diff"]):>9.4f}{(g["diff"] > 1e-9).mean():>13.1%}{(g["diff"] < -1e-9).mean():>15.1%}')

    print('\n=== by dataset: mean rho_A (norm | random) ===')
    byd = df.groupby(['dataset', 'order', 'T']).rho_A.mean().unstack('order').round(3)
    print(byd.to_string())

    # plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for o, mk in (('norm', 'o-'), ('random', 's--')):
            s = summ[(summ.order == o) & np.isfinite(summ['T'])]
            ax.errorbar(s.mean_n_eval, s.mean_rho_A, yerr=s.se, fmt=mk, capsize=2, label=f'{o} order')
            for _, r in s.iterrows():
                ax.annotate(f'T={int(r["T"])}', (r.mean_n_eval, r.mean_rho_A), fontsize=7, xytext=(3, 3), textcoords='offset points')
        f = summ[np.isinf(summ['T'])]
        if len(f):
            ax.errorbar(f.mean_n_eval, f.mean_rho_A, yerr=f.se, fmt='k*', ms=11, capsize=2, label='full scan (norm)')
        ax.set_xscale('log'); ax.set_xlabel('mean exact evaluations N_eval'); ax.set_ylabel('mean rho_A = A_f / A_0')
        ax.grid(alpha=0.3); ax.legend(); ax.set_title('Scan budget T: cost vs amplification reduction')
        fig.tight_layout(); fig.savefig(out / 'T_sweep_cost_performance.png', dpi=160)
    except Exception as e:
        print('plot skipped:', e)
    print(f'\ntotal wall time {time.perf_counter() - t_all:.1f}s; results in {out}')


if __name__ == '__main__':
    main()