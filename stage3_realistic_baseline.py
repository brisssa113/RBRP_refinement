"""Stage 3.3: more realistic matrices and a simple baseline.

Matrices (rows = points to interpolate, skeleton = selected rows):
  burgers_snapshots : rows = snapshots (initial phase a, viscosity nu, time t) of the 1D viscous Burgers
                      equation u_t + (u^2/2)_x = nu u_xx on the periodic interval [0,1), N = 256 grid points,
                      pseudo-spectral with 2/3 dealiasing and a first-order IMEX Euler step (dt = 1e-3),
                      u0 = sin(2 pi (x - a)).  25 phases x 8 viscosities in [0.01, 0.05] x 5 times in
                      [0.1, 1.0] -> 1000 x 256.  The phase makes the family translation-like, so the singular
                      values decay gradually (optimal relative error 7e-2, 2e-2, 2e-3 at k = 8, 16, 24).
                      (A test matrix of snapshot type, not a high-accuracy PDE solution.)
  digits            : scikit-learn handwritten digits, 1797 x 64 (bundled with scikit-learn, no download).

Methods, for each (matrix, k):
  RBRP        : first k RBRP pivots, 10 RBRP seeds (RBRP is the only random ingredient)
  QRCP        : first k pivots of QR with column pivoting applied to X^T (deterministic baseline)
  RBRP+T10    : ordered-scan refinement of each RBRP skeleton, at most 10 rows per round
  RBRP+full   : full ordered scan (only for the first --full-seeds RBRP seeds, it is expensive)
All use the same metrics as before: E = ||X - W X_S||_F/||X||_F, A = ||W||_2, with the guard E <= 1.05 E0
where E0 is the error of the RBRP skeleton being refined.

Usage (next to stage1.py, stage2_sorted_scan.py):
    python stage3_realistic_baseline.py --smoke
    python stage3_realistic_baseline.py
    python stage3_realistic_baseline.py --only digits
    python stage3_realistic_baseline.py --summarize-only
"""
from __future__ import annotations
import argparse, glob, importlib.util, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
KS = {'burgers_snapshots': [8, 16, 24], 'digits': [10, 20, 30]}
T_BUDGET = 10
# The full scan is expensive on these matrices (digits, k=10: about 10^5 evaluations per run), so it is
# restricted: Burgers all ranks, digits only the smallest rank, and at most this many RBRP seeds each.
FULL_KS = {'burgers_snapshots': None, 'digits': [10]}          # None = all ranks
FULL_MAX_SEEDS = {'burgers_snapshots': 3, 'digits': 2}


def load_base():
    spec = importlib.util.spec_from_file_location('stage2_sorted_scan', HERE / 'stage2_sorted_scan.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def make_burgers(n_a=25, n_nu=8, n_t=5, N=256, dt=1e-3, t_min=0.1, t_max=1.0, nu_min=0.01, nu_max=0.05):
    x = np.arange(N) / N
    kk = 2 * np.pi * np.fft.fftfreq(N, d=1.0 / N)
    kidx = np.fft.fftfreq(N, d=1.0 / N)
    mask = (np.abs(kidx) < N / 3).astype(float)                  # 2/3 dealiasing
    a = np.arange(n_a) / n_a
    nus = np.linspace(nu_min, nu_max, n_nu)
    A, NU = np.meshgrid(a, nus, indexing='ij')
    A, NU = A.ravel(), NU.ravel()[:, None]
    u = np.sin(2 * np.pi * (x[None, :] - A[:, None]))
    uh = np.fft.fft(u, axis=1)
    snap_steps = np.round(np.linspace(t_min, t_max, n_t) / dt).astype(int)
    out = np.zeros((len(A), n_t, N))
    denom = 1.0 + dt * NU * kk[None, :] ** 2
    step, j = 0, 0
    while j < n_t:
        u = np.fft.ifft(uh, axis=1).real
        nl = 1j * kk[None, :] * np.fft.fft(0.5 * u * u, axis=1) * mask[None, :]
        uh = (uh - dt * nl) / denom
        step += 1
        if step == snap_steps[j]:
            out[:, j, :] = np.fft.ifft(uh, axis=1).real
            j += 1
    return out.reshape(len(A) * n_t, N)


def make_digits():
    from sklearn.datasets import load_digits
    return load_digits().data.astype(float)


def get_matrix(name, smoke):
    if name == 'burgers_snapshots':
        return make_burgers(n_a=10, n_nu=6, n_t=5) if smoke else make_burgers()
    return make_digits()[:500] if smoke else make_digits()


def timed_median(fn, reps):
    ts, res = [], None
    for _ in range(reps):
        t = time.perf_counter(); res = fn(); ts.append(time.perf_counter() - t)
    return res, float(np.median(ts))


def run_dataset(base, mod, name, ks, seeds, full_seeds, verify, out, smoke):
    X = get_matrix(name, smoke)
    n, d = X.shape
    print(f'=== {name}: X is {n} x {d}, k in {ks}; fro norm {np.linalg.norm(X):.3g}; '
          f'rank-tolerance rank {np.linalg.matrix_rank(X)} ===', flush=True)
    rows = []
    t_ds = time.perf_counter()
    for k in ks:
        # deterministic baseline: QR with column pivoting on X^T
        (Q, R, piv), t_qr = timed_median(lambda: sla.qr(X.T, mode='economic', pivoting=True), 3)
        S_qr = np.asarray(piv[:k], dtype=int)
        dq = mod.compute_id_diagnostics(X, S_qr)
        rows.append({'dataset': name, 'k': k, 'method': 'QRCP', 'seed': -1, 'E': float(dq['error_rel']),
                     'A': float(dq['amp2']), 'time_sec': t_qr, 'valid': bool(dq['valid'])})
        for seed in range(seeds):
            (res, t_rbrp) = timed_median(lambda: mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=seed), 3)
            S0 = np.asarray(res[0][:k], dtype=int)
            d0 = mod.compute_id_diagnostics(X, S0)
            if not d0['valid']:
                rows.append({'dataset': name, 'k': k, 'method': 'RBRP', 'seed': seed, 'valid': False}); continue
            E0, A0 = float(d0['error_rel']), float(d0['amp2'])
            rows.append({'dataset': name, 'k': k, 'method': 'RBRP', 'seed': seed, 'E': E0, 'A': A0,
                         'time_sec': t_rbrp, 'valid': True})
            do_full = (FULL_KS[name] is None or k in FULL_KS[name] or smoke) and seed < min(full_seeds, FULL_MAX_SEEDS[name])
            variants = [('RBRP+T10', T_BUDGET)] + ([('RBRP+full', None)] if do_full else [])
            for vname, T in variants:
                t = time.perf_counter()
                Sf, df, hist, ne, why = base.sorted_scan_refinement(mod, X, S0, E0, max_scan=T)
                el = time.perf_counter() - t
                r = {'dataset': name, 'k': k, 'method': vname, 'seed': seed, 'E': float(df['error_rel']),
                     'A': float(df['amp2']), 'E0': E0, 'A0': A0, 'time_sec': el, 'rbrp_time_sec': t_rbrp,
                     'n_eval': ne, 'rounds': sum(h['accepted'] for h in hist), 'stop_reason': why, 'valid': True}
                if vname == 'RBRP+full' and seed < verify:
                    r['verified_admissible_left'] = base.verify_local_optimum(mod, X, Sf, E0)
                rows.append(r)
            print(f'  [{name}] k={k} RBRP seed {seed + 1}/{seeds} done, elapsed {time.perf_counter() - t_ds:.0f}s', flush=True)
            pd.DataFrame(rows).to_csv(out / f'baseline_raw_{name}.csv', index=False)


def summarize(out):
    files = sorted(glob.glob(str(out / 'baseline_raw_*.csv')))
    if not files:
        print('no raw files'); return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.valid == True].copy()
    order = ['RBRP', 'QRCP', 'RBRP+T10', 'RBRP+full']
    q = df[df.method == 'QRCP'].set_index(['dataset', 'k'])[['E', 'A']].rename(columns={'E': 'E_qr', 'A': 'A_qr'})
    df = df.join(q, on=['dataset', 'k'])
    df['A_over_qr'] = df.A / df.A_qr
    df['E_over_qr'] = df.E / df.E_qr
    g = df.groupby(['dataset', 'k', 'method'])
    s = g.agg(runs=('A', 'size'), median_A=('A', 'median'), q25_A=('A', lambda x: x.quantile(.25)),
              q75_A=('A', lambda x: x.quantile(.75)), median_E=('E', 'median'),
              median_A_over_QRCP=('A_over_qr', 'median'), median_E_over_QRCP=('E_over_qr', 'median'),
              frac_A_below_QRCP=('A_over_qr', lambda x: (x < 1).mean()), mean_time_sec=('time_sec', 'mean'),
              mean_n_eval=('n_eval', 'mean')).reset_index()
    s['m'] = s.method.map({m: i for i, m in enumerate(order)})
    s = s.sort_values(['dataset', 'k', 'm']).drop(columns='m')
    s.to_csv(out / 'baseline_summary.csv', index=False)
    print('\n=== Baseline comparison (A = ||W||_2, E = relative Frobenius error) ===')
    print(s.round(4).to_string(index=False))
    # refinement vs its own RBRP start, paired by seed
    ref = df[df.method.isin(['RBRP+T10', 'RBRP+full'])].copy()
    ref['rho_A'] = ref.A / ref.A0
    ref['rho_E'] = ref.E / ref.E0
    r = ref.groupby(['dataset', 'k', 'method']).agg(runs=('rho_A', 'size'), mean_rho_A=('rho_A', 'mean'),
                                                     mean_rho_E=('rho_E', 'mean'), max_rho_E=('rho_E', 'max')).reset_index()
    r.to_csv(out / 'baseline_refinement_vs_rbrp.csv', index=False)
    print('\n=== Refinement relative to its own RBRP skeleton ===')
    print(r.round(4).to_string(index=False))
    v = df.get('verified_admissible_left')
    if v is not None and v.notna().any():
        print(f'\nbrute-force local-optimum check: {int(v.notna().sum())} runs verified, admissible swaps left: {sorted(v.dropna().unique())}')
    tot = df[df.method.isin(['RBRP', 'RBRP+T10'])].groupby(['dataset', 'k', 'method']).time_sec.mean().unstack()
    if {'RBRP', 'RBRP+T10'} <= set(tot.columns):
        tot['T10 / RBRP time'] = tot['RBRP+T10'] / tot['RBRP']
        print('\n=== time: RBRP vs T10 refinement (refinement time only, seconds) ===')
        print(tot.round(4).to_string())
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys = sorted(df[['dataset', 'k']].drop_duplicates().itertuples(index=False))
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
        for ax, (ds, k) in zip(axes.ravel(), keys):
            sub = df[(df.dataset == ds) & (df.k == k)]
            data = [sub[sub.method == m].A.values for m in order]
            ax.boxplot([d_ if len(d_) else [np.nan] for d_ in data])
            ax.set_xticklabels(['RBRP', 'QRCP', '+T10', '+full'])
            ax.set_yscale('log'); ax.set_title(f'{ds}, k={k}', fontsize=9); ax.grid(alpha=0.3, which='both')
            ax.set_ylabel('A = ||W||_2')
        fig.tight_layout(); fig.savefig(out / 'baseline_amplification.png', dpi=160)
    except Exception as e:
        print('plot skipped:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(HERE / 'stage3_baseline_results'))
    ap.add_argument('--seeds', type=int, default=10, help='RBRP seeds per (matrix, k)')
    ap.add_argument('--full-seeds', type=int, default=3, help='full scan only for the first N RBRP seeds (capped per matrix, see FULL_MAX_SEEDS)')
    ap.add_argument('--verify', type=int, default=1, help='brute-force verify local optimum for first N full-scan seeds')
    ap.add_argument('--only', default=None)
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.seeds, a.full_seeds, a.verify = 2, 1, 1
    if not a.summarize_only:
        base = load_base()
        mod = base.load_stage1(Path(a.stage1))
        t0 = time.perf_counter()
        for name, ks in KS.items():
            if a.only and name != a.only:
                continue
            run_dataset(base, mod, name, ks[:1] if a.smoke else ks, a.seeds, a.full_seeds, a.verify, out, a.smoke)
        print(f'\nwall time {time.perf_counter() - t0:.0f}s')
    summarize(out)


if __name__ == '__main__':
    main()