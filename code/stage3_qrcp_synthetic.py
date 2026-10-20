"""QRCP baseline on the four synthetic datasets (closes the gap that QRCP was only run on Burgers/digits/real matrices).

Protocol = that of stage3_final_experiments.py: dataset = builder(n=500, d=60, seed) with data seeds 0..seeds-1 (default 20, the seeds of the
final experiment), representative rank k of each dataset. QRCP = first k pivots of QR with column pivoting on X^T (deterministic, one run per
matrix instance). A0/E0 of the RBRP skeleton are recomputed with the same RBRP seed formula and checked against results/stage3_final_results.
Outputs (results/stage3_qrcp_synthetic_results/): qrcp_synthetic_raw.csv, qrcp_synthetic_summary.csv.
Usage: python code/stage3_qrcp_synthetic.py [--seeds 20] [--smoke]
"""
import argparse, importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.linalg as sla

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
FINAL = ROOT / 'results' / 'stage3_final_results'

def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m; spec.loader.exec_module(m); return m

def boot_ci(x, B=5000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); n = len(x)
    med = np.median(x[rng.integers(0, n, (B, n))], axis=1); return np.percentile(med, [2.5, 97.5])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results' / 'stage3_qrcp_synthetic_results'))
    ap.add_argument('--seeds', type=int, default=20); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); a.seeds = 2 if a.smoke else a.seeds
    base = load_module('stage2_sorted_scan', 'stage2_sorted_scan.py'); mod = base.load_stage1(HERE / 'stage1.py')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows = []
    for di, (ds, k) in enumerate(base.DATASET_K.items()):
        for seed in range(a.seeds):
            X = getattr(mod, base.BUILDERS[ds])(n=500, d=60, seed=seed)
            rbrp_seed = 1_000_000 + di * 100_000 + seed
            row_perm, *_ = mod.blocked_rpid(X, k=k, b=base.B, mode='rp', ortho=1, seed=rbrp_seed)
            d0 = mod.compute_id_diagnostics(X, np.asarray(row_perm[:k], dtype=int))
            t = time.perf_counter(); (_, _, piv) = sla.qr(X.T, mode='economic', pivoting=True)
            dq = mod.compute_id_diagnostics(X, np.asarray(piv[:k], dtype=int)); t = time.perf_counter() - t
            rows.append(dict(dataset=ds, k=k, seed=seed, A=float(dq['amp2']), E=float(dq['error_rel']), A0=float(d0['amp2']),
                             E0=float(d0['error_rel']), time_sec=t))
    raw = pd.DataFrame(rows); raw.to_csv(out / 'qrcp_synthetic_raw.csv', index=False)
    if a.smoke: print('smoke ok'); return
    fin = pd.concat([pd.read_csv(FINAL / f'final_raw_{d}.csv') for d in base.DATASET_K]); fin = fin[fin.seed >= 0]
    chk = fin[fin.method == 'RBRP'].merge(raw, on=['dataset', 'seed'], suffixes=('', '_q'))
    assert len(chk) == len(raw) and np.allclose(chk.A0, chk.A0_q, rtol=1e-9) and np.allclose(chk.E0, chk.E0_q, rtol=1e-9), 'RBRP baseline mismatch'
    print('RBRP A0/E0 of all %d matrices reproduce the stored final experiment' % len(raw))
    S = []
    for ds in base.DATASET_K:
        q = raw[raw.dataset == ds].set_index('seed'); r = dict(dataset=ds, k=int(q.k.iloc[0]), runs=len(q))
        r['median_A_RBRP'] = float(q.A0.median()); r['median_A_QRCP'] = float(q.A.median()); r['median_E_QRCP_over_E0'] = float((q.E / q.E0).median())
        r['frac_QRCP_within_budget'] = float((q.E <= 1.05 * q.E0 + 1e-14).mean())
        for m in ['Ours-T10', 'Hybrid-T10']:
            f = fin[(fin.dataset == ds) & (fin.method == m)].set_index('seed')
            r[f'median_A_{m}'] = float(f.A.median()); ratio = (f.A / q.A).dropna()
            r[f'median_A_{m}_over_QRCP'] = float(ratio.median()); lo, hi = boot_ci(ratio.values)
            r[f'ci_lo_{m}'], r[f'ci_hi_{m}'] = float(lo), float(hi); r[f'frac_{m}_below_QRCP'] = float((ratio < 1).mean())
        S.append(r)
    S = pd.DataFrame(S); S.to_csv(out / 'qrcp_synthetic_summary.csv', index=False); print(S.round(3).T.to_string())

if __name__ == '__main__':
    main()
