"""Bootstrap confidence intervals for the comparisons of the paper.

Resampling unit: the RBRP seed.  Within every configuration (matrix, k) the seeds are resampled with replacement (stratified
bootstrap); the paired structure is kept (all methods of one seed are resampled together).  Pooled statistics are recomputed
for every resample.  95% percentile intervals, B = 5000.

Inputs : final_raw_*.csv (from stage3_final_experiments.py) and noise_raw_*.csv (from stage3_noise_downstream.py)
Outputs: ci_pooled.csv, ci_per_config.csv, ci_noise.csv, ci_noise_pair.csv, nu_star.csv, ci_misc.txt, and LaTeX tables in tex/
Usage  : python bootstrap_ci.py --indir <folder with the raw csv files> --out bootstrap
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

B = 5000
TOL = 1e-9
PAIRS = [('Ours-T10', 'Vol(f=1)'), ('Ours-T10', 'VolGuard-500'), ('Hybrid-T10', 'VolGuard-500'), ('Hybrid-T10', 'Ours-T10'),
         ('Hybrid-T10', 'Vol(f=1)'), ('Ours-T10-noG', 'Vol(f=1)'), ('Ours-full', 'VolGuard-500'), ('Ours-full', 'Hybrid-T10'),
         ('Ours-full', 'Vol(f=1)')]
NM = {'nearly_collinear_clusters': 'Collinear clusters', 'repeated_directions': 'Repeated directions', 'localized_svd': 'Localized SVD',
      'kernel_snapshots': 'Kernel snapshots', 'burgers_snapshots': 'Burgers', 'digits': 'Digits',
      'image_patches': 'Image patches', 'ecg_hankel': 'ECG Hankel', 'digits_kernel': 'Digits kernel', 'olivetti': 'Olivetti'}
rng = np.random.default_rng(20261014)


def ci(v):
    return np.percentile(v, [2.5, 97.5])


def pooled_stat(groups, fn):
    """groups: list of per-config arrays (n_i x m); fn maps the concatenated array to a vector of statistics."""
    return fn(np.concatenate(groups, axis=0))


def boot_pooled(groups, fn):
    est = fn(np.concatenate(groups, axis=0))
    res = []
    for _ in range(B):
        res.append(fn(np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups], axis=0)))
    res = np.array(res)
    lo, hi = np.percentile(res, [2.5, 97.5], axis=0)
    return est, lo, hi


def pair_fn(z):                       # columns: a = A_x/A_y, e = E_x/E_y
    a, e = z[:, 0], z[:, 1]
    xd = ((a <= 1 + TOL) & (e <= 1 + TOL) & ((a < 1 - TOL) | (e < 1 - TOL))).mean()
    yd = ((a >= 1 - TOL) & (e >= 1 - TOL) & ((a > 1 + TOL) | (e > 1 + TOL))).mean()
    return np.array([np.median(a), np.median(e), (a < 1 - TOL).mean(), xd, yd])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default='.')
    ap.add_argument('--out', default='bootstrap')
    a = ap.parse_args()
    out = Path(a.out); (out / 'tex').mkdir(parents=True, exist_ok=True)
    fr = pd.concat([pd.read_csv(x) for x in glob.glob(str(Path(a.indir) / '*final_raw_*.csv'))], ignore_index=True)
    nfiles = glob.glob(str(Path(a.indir) / '*noise_raw_*.csv'))
    nr = pd.concat([pd.read_csv(x) for x in nfiles], ignore_index=True) if nfiles else pd.DataFrame(columns=['seed'])
    fr = fr[fr.seed >= 0]
    pa, pe = fr.pivot_table(index=['dataset', 'k', 'seed'], columns='method', values='A'), fr.pivot_table(index=['dataset', 'k', 'seed'], columns='method', values='E')
    misc = []

    # ---------------- pooled paired comparisons
    rows, pc = [], []
    for x, y in PAIRS:
        if x not in pa.columns or y not in pa.columns:
            continue
        q = pd.DataFrame({'a': pa[x] / pa[y], 'e': pe[x] / pe[y]}).dropna()
        groups = [g.values for _, g in q.groupby(level=[0, 1])]
        est, lo, hi = boot_pooled(groups, pair_fn)
        rows.append({'x': x, 'y': y, 'pairs': len(q), 'A_ratio': est[0], 'A_lo': lo[0], 'A_hi': hi[0], 'E_ratio': est[1], 'E_lo': lo[1], 'E_hi': hi[1],
                     'x_smaller_A': est[2], 'sm_lo': lo[2], 'sm_hi': hi[2], 'x_dom': est[3], 'xd_lo': lo[3], 'xd_hi': hi[3],
                     'y_dom': est[4], 'yd_lo': lo[4], 'yd_hi': hi[4]})
        # per-config
        for (ds, k), g in q.groupby(level=[0, 1]):
            v = g.a.values
            bm = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(B)])
            lg = np.log(v); nz = np.abs(lg) > 1e-9
            w = stats.wilcoxon(lg[nz]).pvalue if nz.sum() >= 6 else np.nan
            pc.append({'x': x, 'y': y, 'dataset': ds, 'k': k, 'pairs': len(v), 'A_ratio': np.median(v), 'lo': ci(bm)[0], 'hi': ci(bm)[1],
                       'share_x_smaller': (v < 1 - TOL).mean(), 'share_tied': (np.abs(v - 1) <= TOL).mean(), 'wilcoxon_p': w})
    P, C = pd.DataFrame(rows), pd.DataFrame(pc)
    P.to_csv(out / 'ci_pooled.csv', index=False); C.to_csv(out / 'ci_per_config.csv', index=False)
    print('=== POOLED paired comparisons with 95% bootstrap intervals (seeds resampled within configuration) ===')
    for _, r in P.iterrows():
        print(f"{r.x:>13s} vs {r.y:<13s} n={r.pairs:3d} | A ratio {r.A_ratio:.3f} [{r.A_lo:.3f},{r.A_hi:.3f}] | E ratio {r.E_ratio:.3f} [{r.E_lo:.3f},{r.E_hi:.3f}] | "
              f"x smaller {100*r.x_smaller_A:.0f}% [{100*r.sm_lo:.0f},{100*r.sm_hi:.0f}] | x dom {100*r.x_dom:.1f}% [{100*r.xd_lo:.1f},{100*r.xd_hi:.1f}] | y dom {100*r.y_dom:.1f}% [{100*r.yd_lo:.1f},{100*r.yd_hi:.1f}]")
    print('\n=== PER CONFIGURATION: Hybrid vs VolGuard and Ours-T10 vs VolGuard (median A ratio, 95% CI, Wilcoxon p on non-tied pairs) ===')
    for xy in [('Hybrid-T10', 'VolGuard-500'), ('Ours-T10', 'VolGuard-500'), ('Ours-full', 'VolGuard-500')]:
        print('--', xy)
        for _, r in C[(C.x == xy[0]) & (C.y == xy[1])].iterrows():
            print(f"   {NM[r.dataset]:20s} k={r.k:2d} n={r.pairs:2d}  {r.A_ratio:.3f} [{r.lo:.3f},{r.hi:.3f}]  smaller {100*r.share_x_smaller:.0f}%  tied {100*r.share_tied:.0f}%  p={r.wilcoxon_p:.3g}")

    # ---------------- feasibility of Vol(f=1) and amplification reduction per method
    f = fr.merge(fr[fr.method == 'RBRP'][['dataset', 'k', 'seed', 'A', 'E']].rename(columns={'A': 'A_r', 'E': 'E_r'}), on=['dataset', 'k', 'seed'])
    f['rho_A'], f['rho_E'] = f.A / f.A_r, f.E / f.E_r
    v = f[f.method == 'Vol(f=1)']; viol = (v.rho_E > 1.05 + 1e-9).values
    lo_w, hi_w = stats.binomtest(int(viol.sum()), len(viol)).proportion_ci(0.95, method='wilson')
    misc.append(f'Vol(f=1) violates E<=1.05E0 in {viol.sum()} of {len(viol)} runs ({100*viol.mean():.1f}%), Wilson 95% CI [{100*lo_w:.1f}%, {100*hi_w:.1f}%]; max rho_E {v.rho_E.max():.2f}')
    for m in ['VolGuard-500', 'Ours-T10', 'Hybrid-T10']:
        z = f[f.method == m]; misc.append(f'{m}: feasible in {100*(z.rho_E <= 1.05 + 1e-9).mean():.0f}% of runs; max rho_E {z.rho_E.max():.4f}')
    lines = []
    for m in ['Hybrid-T10', 'Ours-full']:
        z = f[f.method == m]
        if z.empty:
            continue
        gs = [g.rho_A.values.reshape(-1, 1) for _, g in z.groupby(['dataset', 'k'])]
        est, lo, hi = boot_pooled(gs, lambda t: np.array([np.median(t[:, 0]), np.mean(t[:, 0])]))
        lines.append(f'{m}: pooled median rho_A {est[0]:.3f} [{lo[0]:.3f},{hi[0]:.3f}], mean {est[1]:.3f} [{lo[1]:.3f},{hi[1]:.3f}]')
        per = z.groupby(['dataset', 'k']).rho_A.apply(lambda s: (s.mean(), *np.percentile([s.values[rng.integers(0, len(s), len(s))].mean() for _ in range(B)], [2.5, 97.5])))
        lines.append('   per config mean rho_A [95% CI]: ' + '; '.join(f"{NM[d]} k={k}: {a:.3f} [{l:.3f},{h:.3f}]" for (d, k), (a, l, h) in per.items()))
    misc += lines
    # median A ratio RBRP / Hybrid
    q = (pa['RBRP'] / pa['Hybrid-T10']).dropna()
    est, lo, hi = boot_pooled([g.values.reshape(-1, 1) for _, g in q.groupby(level=[0, 1])], lambda t: np.array([np.median(t[:, 0])]))
    misc.append(f'median of A(RBRP)/A(Hybrid) over all runs: {est[0]:.2f} [{lo[0]:.2f},{hi[0]:.2f}]')

    # ---------------- downstream noise
    if nr.empty:
        print('\n(no noise_raw files: downstream part skipped)'); print('\n=== misc ==='); print('\n'.join(misc)); open(out / 'ci_misc.txt', 'w').write('\n'.join(misc)); return
    nr = nr[nr.seed >= 0]
    r0 = nr[nr.method == 'RBRP'].set_index(['dataset', 'k', 'seed'])
    levels = ['0.001', '0.01', '0.1', '1']
    nrows, prow = [], []
    for m in ['Vol(f=1)', 'VolGuard-500', 'Ours-T10', 'Hybrid-T10']:
        z = nr[nr.method == m].set_index(['dataset', 'k', 'seed'])
        for kind in ('rand', 'worst'):
            for lv in levels:
                rat = (z[f'{kind}_{lv}'] / r0[f'{kind}_{lv}']).dropna()
                est, lo, hi = boot_pooled([g.values.reshape(-1, 1) for _, g in rat.groupby(level=[0, 1])], lambda t: np.array([np.median(t[:, 0])]))
                nrows.append({'method': m, 'kind': kind, 'level': float(lv), 'ratio_to_RBRP': est[0], 'lo': lo[0], 'hi': hi[0]})
        for col in ('A', 'F'):
            rat = (z[col] / r0[col]).dropna()
            est, lo, hi = boot_pooled([g.values.reshape(-1, 1) for _, g in rat.groupby(level=[0, 1])], lambda t: np.array([np.median(t[:, 0])]))
            nrows.append({'method': m, 'kind': 'rho_' + col, 'level': np.nan, 'ratio_to_RBRP': est[0], 'lo': lo[0], 'hi': hi[0]})
    N = pd.DataFrame(nrows); N.to_csv(out / 'ci_noise.csv', index=False)
    # hybrid vs VolGuard directly
    zh, zv = nr[nr.method == 'Hybrid-T10'].set_index(['dataset', 'k', 'seed']), nr[nr.method == 'VolGuard-500'].set_index(['dataset', 'k', 'seed'])
    for kind in ('rand', 'worst'):
        for lv in levels:
            rat = (zh[f'{kind}_{lv}'] / zv[f'{kind}_{lv}']).dropna()
            est, lo, hi = boot_pooled([g.values.reshape(-1, 1) for _, g in rat.groupby(level=[0, 1])], lambda t: np.array([np.median(t[:, 0])]))
            prow.append({'kind': kind, 'level': float(lv), 'hybrid_over_volguard': est[0], 'lo': lo[0], 'hi': hi[0]})
    PN = pd.DataFrame(prow); PN.to_csv(out / 'ci_noise_pair.csv', index=False)
    print('\n=== DOWNSTREAM: median error ratio to the RBRP skeleton, 95% CI ===')
    for m in N.method.unique():
        s = N[N.method == m]
        print(m, '| ' + ' | '.join(f"{r.kind} {r.level:g}: {r.ratio_to_RBRP:.3f} [{r.lo:.3f},{r.hi:.3f}]" if r.kind in ('rand', 'worst') else f"{r.kind}: {r.ratio_to_RBRP:.3f} [{r.lo:.3f},{r.hi:.3f}]" for r in s.itertuples()))
    print('\n=== Hybrid / VolGuard error ratio (paired), 95% CI ===')
    print(PN.round(3).to_string(index=False))

    # ---------------- nu*, correlations (the numbers that were TODO in the paper)
    r = nr[nr.method == 'RBRP'].copy(); r['nu_star'] = r.E * np.sqrt(r.n) / r.F
    ns = r.groupby(['dataset', 'k']).nu_star.agg(['median', lambda s: np.percentile(s, 25), lambda s: np.percentile(s, 75)]); ns.columns = ['median', 'q25', 'q75']
    ns.to_csv(out / 'nu_star.csv'); print('\n=== nu* of the RBRP skeletons (median, quartiles over seeds) ==='); print(ns.round(4).to_string())
    ok = nr[nr.method.isin(['RBRP', 'Vol(f=1)', 'VolGuard-500', 'Ours-T10', 'Hybrid-T10'])]
    misc.append(f'corr(log A, log ||W||_F) over {len(ok)} skeletons: {np.corrcoef(np.log(ok.A), np.log(ok.F))[0, 1]:.3f}; median A/||W||_F = {np.median(ok.A / ok.F):.3f}')
    print('\n=== misc ==='); print('\n'.join(misc)); open(out / 'ci_misc.txt', 'w').write('\n'.join(misc))


if __name__ == '__main__':
    main()
