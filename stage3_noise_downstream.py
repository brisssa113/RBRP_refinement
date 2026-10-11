"""Stage 3.7: downstream benefit of a lower amplification.

Use case: only the skeleton rows are observed, with perturbation N:   X_hat = W (X_S + N),   W = X X_S^+ .
Reconstruction error relative to the truth X:
    X_hat - X = (W X_S - X) + W N .
Two perturbation models (nu, eta are dimensionless):

(1) RANDOM noise: entries of N iid N(0, sigma^2), sigma = nu * ||X||_F / sqrt(n d)  (nu = noise relative to the rms entry of X).
    E||X_hat - X||_F^2 / ||X||_F^2 = E(S)^2 + nu^2 ||W||_F^2 / n         (exact; the cross term has zero mean)
    -> controlled by ||W||_F, NOT by ||W||_2  (note  sqrt(k) <= ||W||_F <= sqrt(k) ||W||_2).

(2) WORST-CASE perturbation of relative size eta:  ||N||_F = eta * ||X_S||_F, direction chosen adversarially.
    A direction exists with  ||X_hat - X||_F^2 >= ||W X_S - X||_F^2 + (||W||_2 ||N||_F)^2, hence
    worst-case error >= sqrt( E(S)^2 + (A(S) * eta * ||X_S||_F / ||X||_F)^2 )     (reported below; upper bound: sum instead of root-sum-square)
    -> controlled by ||W||_2.

For every configuration and RBRP seed we start from the same RBRP skeleton as in stage3_final_experiments.py and compare
RBRP, QRCP (real matrices only), Vol(f=1), VolGuard-500, Ours-T10 and Hybrid-T10.
Reported per method: A, ||W||_F, E, the predicted reconstruction errors at several nu / eta, their ratio to the RBRP skeleton
(paired by seed), and the crossover nu* = E0 sqrt(n) / ||W0||_F above which the noise term of the RBRP skeleton exceeds its bias.
A Monte-Carlo check of formula (1) is printed at the end.

Usage (next to stage1.py, stage2_sorted_scan.py, stage3_realistic_baseline.py, stage3_volume_baseline.py):
    python stage3_noise_downstream.py --smoke
    python stage3_noise_downstream.py
    python stage3_noise_downstream.py --only burgers_snapshots
    python stage3_noise_downstream.py --summarize-only
"""
from __future__ import annotations
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
GUARD, T_BUDGET, CANDS = 1.05, 10, 500
NUS = [1e-3, 1e-2, 1e-1, 1.0]
METHODS = ['RBRP', 'QRCP', 'Vol(f=1)', 'VolGuard-500', 'Ours-T10', 'Hybrid-T10']


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def metrics(base, mod, X, S):
    d = mod.compute_id_diagnostics(X, S)
    if not d['valid']:
        return None
    W = base.interp_matrix(mod, X, S)
    n, dd = X.shape
    xn = float(np.linalg.norm(X))
    r = {'A': float(d['amp2']), 'F': float(np.linalg.norm(W, 'fro')), 'E': float(d['error_rel']),
         'XS': float(np.linalg.norm(X[S])), 'xn': xn, 'n': n}
    for nu in NUS:
        r[f'rand_{nu:g}'] = float(np.sqrt(r['E'] ** 2 + nu ** 2 * r['F'] ** 2 / n))
        r[f'worst_{nu:g}'] = float(np.sqrt(r['E'] ** 2 + (r['A'] * nu * r['XS'] / xn) ** 2))
    return r, W


def run_matrix(base, mod, rb, vb, name, kind, ks, seeds, smoke, out, mc_rows):
    rows, t_ds = [], time.perf_counter()
    if kind == 'real':
        X0 = rb.get_matrix(name, smoke)
        Usvd = np.linalg.svd(X0, full_matrices=False)[0]
    for seed in range(seeds):
        if kind == 'synthetic':
            di = list(base.DATASET_K).index(name)
            X = getattr(mod, base.BUILDERS[name])(n=500, d=60, seed=seed)
            rbrp_seed = 1_000_000 + di * 100_000 + seed
            U_all = np.linalg.svd(X, full_matrices=False)[0]
        else:
            X, rbrp_seed, U_all = X0, seed, Usvd
        for k in ks:
            if kind == 'real' and seed == 0:
                (Q, R, piv) = sla.qr(X0.T, mode='economic', pivoting=True)
                m = metrics(base, mod, X0, np.asarray(piv[:k], dtype=int))
                if m:
                    rows.append({'dataset': name, 'k': k, 'seed': -1, 'method': 'QRCP', **m[0]})
            U = U_all[:, :k]
            row_perm, *_ = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=rbrp_seed)
            S0 = np.asarray(row_perm[:k], dtype=int)
            m0 = metrics(base, mod, X, S0)
            if m0 is None:
                continue
            E0 = m0[0]['E']
            sk = {'RBRP': S0}
            sk['Vol(f=1)'], _, _ = vb.volume_exchange(mod, X, U, S0, 1.0)
            vb.GUARD_CANDS = CANDS
            Sg, _, _ = vb.volume_exchange(mod, X, U, S0, 1.0, E0=E0, guard=GUARD)
            sk['VolGuard-500'] = Sg
            sk['Ours-T10'] = base.sorted_scan_refinement(mod, X, S0, E0, max_scan=T_BUDGET)[0]
            sk['Hybrid-T10'] = base.sorted_scan_refinement(mod, X, Sg, E0, max_scan=T_BUDGET)[0]
            for mname, S in sk.items():
                m = metrics(base, mod, X, np.asarray(S, dtype=int))
                if m:
                    rows.append({'dataset': name, 'k': k, 'seed': seed, 'method': mname, **m[0]})
                    if seed == 0 and mname in ('RBRP', 'Hybrid-T10'):       # Monte-Carlo check of formula (1)
                        S_ = np.asarray(S, dtype=int); W = m[1]; n, d = X.shape; xn = m[0]['xn']
                        rng = np.random.default_rng(7); nu = 0.1; sig = nu * xn / np.sqrt(n * d); acc = 0.0; draws = 100
                        for _ in range(draws):
                            Nz = sig * rng.standard_normal((len(S_), d))
                            acc += np.linalg.norm(W @ (X[S_] + Nz) - X) ** 2 / xn ** 2
                        pred = m[0]['E'] ** 2 + nu ** 2 * m[0]['F'] ** 2 / n
                        mc_rows.append({'dataset': name, 'k': k, 'method': mname, 'MC_mean_sq': acc / draws, 'formula': pred,
                                        'rel_dev': abs(acc / draws - pred) / pred})
        print(f'  [{name}] seed {seed + 1}/{seeds} done, elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out / f'noise_raw_{name}.csv', index=False)


def summarize(out):
    files = sorted(glob.glob(str(out / 'noise_raw_*.csv')))
    if not files:
        print('no raw files'); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    r0 = df[df.method == 'RBRP'].set_index(['dataset', 'k', 'seed'])
    df = df.join(r0[['A', 'F', 'E'] + [f'rand_{v:g}' for v in NUS] + [f'worst_{v:g}' for v in NUS]].add_suffix('_0'),
                 on=['dataset', 'k', 'seed'])
    df['rho_A'], df['rho_F'], df['rho_E'] = df.A / df.A_0, df.F / df.F_0, df.E / df.E_0
    for v in NUS:
        df[f'rr_{v:g}'] = df[f'rand_{v:g}'] / df[f'rand_{v:g}_0']
        df[f'rw_{v:g}'] = df[f'worst_{v:g}'] / df[f'worst_{v:g}_0']
    df['nu_star'] = df.E_0 * np.sqrt(df.n) / df.F_0
    agg = {'runs': ('A', 'size'), 'median_A': ('A', 'median'), 'median_F': ('F', 'median'), 'mean_rho_A': ('rho_A', 'mean'),
           'mean_rho_F': ('rho_F', 'mean'), 'mean_rho_E': ('rho_E', 'mean')}
    for v in NUS:
        agg[f'med_rand_ratio_{v:g}'] = (f'rr_{v:g}', 'median')
        agg[f'med_worst_ratio_{v:g}'] = (f'rw_{v:g}', 'median')
    s = df.groupby(['dataset', 'k', 'method'], sort=False).agg(**agg).reset_index()
    s['o'] = s.method.map({m: i for i, m in enumerate(METHODS)})
    s = s.sort_values(['dataset', 'k', 'o'], kind='stable').drop(columns='o')
    s.to_csv(out / 'noise_summary.csv', index=False)
    print('\n=== per configuration: amplification A, Frobenius norm F=||W||_F, reductions and noise-error ratios to the RBRP skeleton ===')
    print('    (median over seeds of err(method)/err(RBRP); "rand" uses ||W||_F, "worst" uses ||W||_2)')
    print(s.round(3).to_string(index=False))

    star = df[df.method == 'RBRP'].groupby(['dataset', 'k']).nu_star.median().rename('median nu* (noise level where noise term = bias, RBRP skeleton)')
    print('\n=== crossover noise level nu* of the RBRP skeletons ===')
    print(star.round(4).to_string())

    h = df[df.method.isin(['Hybrid-T10', 'Ours-T10', 'VolGuard-500', 'Vol(f=1)'])]
    pooled = h.groupby('method').agg(mean_rho_A=('rho_A', 'mean'), mean_rho_F=('rho_F', 'mean'), median_rho_A=('rho_A', 'median'),
                                     median_rho_F=('rho_F', 'median'), share_F_reduced=('rho_F', lambda x: (x < 1 - 1e-9).mean()),
                                     **{f'med_rand_{v:g}': (f'rr_{v:g}', 'median') for v in NUS},
                                     **{f'med_worst_{v:g}': (f'rw_{v:g}', 'median') for v in NUS}).reset_index()
    pooled.to_csv(out / 'noise_pooled.csv', index=False)
    print('\n=== POOLED over all configurations and seeds (ratio to the RBRP skeleton, median) ===')
    print(pooled.round(3).to_string(index=False))

    ok = df[df.method.isin(['RBRP'] + list(h.method.unique()))]
    c = np.corrcoef(np.log(ok.A), np.log(ok.F))[0, 1]
    print(f'\ncorrelation of log A and log ||W||_F over all skeletons: {c:.3f};   median of A/||W||_F = {np.median(ok.A / ok.F):.3f}; '
          f'||W||_F/sqrt(k) median = {np.median(ok.F / np.sqrt(ok.k)):.2f}')

    mcf = sorted(glob.glob(str(out / 'noise_mc.csv')))
    if mcf:
        mc = pd.read_csv(mcf[0])
        print(f'\nMonte-Carlo check of the formula (100 draws, nu=0.1): max relative deviation {mc.rel_dev.max():.3f}, median {mc.rel_dev.median():.3f} over {len(mc)} cases')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = list(dict.fromkeys(zip(df.dataset, df.k)))
        grid = np.logspace(-3, 0, 31)
        for kind, fn, ylab in (('rand', 'noise_random.png', 'rel. reconstruction error (random noise)'),
                               ('worst', 'noise_worst.png', 'rel. reconstruction error (worst-case, lower bound)')):
            fig, axes = plt.subplots(2, 5, figsize=(18, 7))
            for ax, (ds, k) in zip(axes.ravel(), keys):
                sub = df[(df.dataset == ds) & (df.k == k)]
                for m in ['RBRP', 'Vol(f=1)', 'VolGuard-500', 'Ours-T10', 'Hybrid-T10']:
                    z = sub[sub.method == m]
                    if not len(z):
                        continue
                    E, F, A, XS, xn, n = (z[c].median() for c in ('E', 'F', 'A', 'XS', 'xn', 'n'))
                    y = np.sqrt(E ** 2 + grid ** 2 * F ** 2 / n) if kind == 'rand' else np.sqrt(E ** 2 + (A * grid * XS / xn) ** 2)
                    ax.plot(grid, y, label=m)
                ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(alpha=.3, which='both'); ax.set_title(f'{ds}, k={k}', fontsize=8)
                ax.set_xlabel('nu' if kind == 'rand' else 'eta')
            axes[0, 0].set_ylabel(ylab, fontsize=7); axes[0, 0].legend(fontsize=6)
            fig.tight_layout(); fig.savefig(out / fn, dpi=150)
    except Exception as e:
        print('plot skipped:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(HERE / 'stage3_noise_results'))
    ap.add_argument('--seeds', type=int, default=20)
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
        vb = load_module('stage3_volume_baseline', 'stage3_volume_baseline.py')
        mod = base.load_stage1(Path(a.stage1))
        mc_rows, t0 = [], time.perf_counter()
        for ds, k in base.DATASET_K.items():
            if a.only and a.only != ds:
                continue
            print(f'=== {ds} (synthetic, k={k}) ===', flush=True)
            run_matrix(base, mod, rb, vb, ds, 'synthetic', [k], a.seeds, a.smoke, out, mc_rows)
        for name, ks in rb.KS.items():
            if a.only and a.only != name:
                continue
            print(f'=== {name} (real/snapshot) ===', flush=True)
            run_matrix(base, mod, rb, vb, name, 'real', ks[:1] if a.smoke else ks, a.seeds, a.smoke, out, mc_rows)
        if mc_rows:
            p = out / 'noise_mc.csv'
            old = pd.read_csv(p) if p.exists() else pd.DataFrame()
            pd.concat([old, pd.DataFrame(mc_rows)], ignore_index=True).to_csv(p, index=False)
        print(f'\nwall time {time.perf_counter() - t0:.0f}s')
    summarize(out)


if __name__ == '__main__':
    main()