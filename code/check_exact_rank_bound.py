"""Numerical check of the exact-rank proposition (Sec. 4): for rank X = k, Vol(f=1) ends with |W_ij| <= 1 off the skeleton and
A(S1) <= sqrt(1+k(n-k)); VolGuard coincides with Vol in exact arithmetic, but in floating point its error guard (E0 ~ 1e-16) can block exchanges.
Random exact-rank matrices X = U_true B with strongly non-uniform row norms; RBRP skeleton S0 (b = 4).
Usage: python code/check_exact_rank_bound.py  -> results/exact_rank_check/exact_rank_check.csv (+ summary)
"""
import importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
def lm(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
base = lm('stage2_sorted_scan', 'stage2_sorted_scan.py'); vb = lm('stage3_volume_baseline', 'stage3_volume_baseline.py'); mod = base.load_stage1(HERE / 'stage1.py')
rng = np.random.default_rng(1); rows = []
for n, k, d in [(200, 5, 30), (300, 8, 40), (500, 12, 60)]:
    for t in range(6):
        Ut = np.linalg.qr(rng.standard_normal((n, k)) * np.exp(1.5 * rng.standard_normal((n, 1))))[0]
        X = Ut @ (rng.standard_normal((k, d)) * np.exp(rng.standard_normal((k, 1))))
        U = np.linalg.svd(X, full_matrices=False)[0][:, :k]
        row_perm, *_ = mod.blocked_rpid(X, k=k, b=4, mode='rp', ortho=1, seed=t); S0 = np.asarray(row_perm[:k], dtype=int)
        d0 = mod.compute_id_diagnostics(X, S0); E0, A0 = float(d0['error_rel']), float(d0['amp2'])
        for tag, kw in (('Vol(f=1)', {}), ('VolGuard-500', dict(E0=E0, guard=1.05))):
            S1, sw, conv = vb.volume_exchange(mod, X, U, S0, 1.0, **kw)
            W = U @ np.linalg.inv(U[S1]); off = np.setdiff1d(np.arange(n), S1); d1 = mod.compute_id_diagnostics(X, S1)
            f_eff = max(1.0, float(np.abs(W[off]).max())); rows.append(dict(n=n, k=k, trial=t, method=tag, E0=E0, E1=float(d1['error_rel']), A0=A0, A1=float(d1['amp2']),
                bound_f1=np.sqrt(1 + k * (n - k)), bound_feff=np.sqrt(1 + f_eff**2 * k * (n - k)), max_off=float(np.abs(W[off]).max()), swaps=sw, converged=bool(conv)))
R = pd.DataFrame(rows); (ROOT / 'results' / 'exact_rank_check').mkdir(parents=True, exist_ok=True); R.to_csv(ROOT / 'results' / 'exact_rank_check' / 'exact_rank_check.csv', index=False)
R['A1<=bound(f_eff)'] = R.A1 <= R.bound_feff * (1 + 1e-9); R['reached_f1'] = R.max_off <= 1 + 1e-6
print(R.groupby(['n', 'k', 'method']).agg(runs=('trial', 'size'), A0_med=('A0', 'median'), A1_med=('A1', 'median'), bound=('bound_f1', 'first'),
      reached_f1=('reached_f1', 'sum'), bound_holds=('A1<=bound(f_eff)', 'sum')).to_string(float_format=lambda x: '%.3g' % x))
print('overall: bound with the attained f_eff holds in %d/%d runs; f=1 reached in %d/%d (Vol) and %d/%d (VolGuard)' % (R['A1<=bound(f_eff)'].sum(), len(R),
      R[R.method == 'Vol(f=1)'].reached_f1.sum(), (R.method == 'Vol(f=1)').sum(), R[R.method == 'VolGuard-500'].reached_f1.sum(), (R.method == 'VolGuard-500').sum()))
print('max A1/bound_f1 = %.3f' % (R.A1 / R.bound_f1).max())
