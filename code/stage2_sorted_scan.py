"""Stage 2.10: deterministic sorted-scan refinement (no candidate pool, no gamma/b/c).

Refinement rule (replaces Section 6 of the handover; Section 5 is unchanged):
  each round
    1. W = X X_S^dagger; rank non-skeleton rows by ||W(i,:)||_2, largest first.
    2. take the next row i in that order, try swapping it with each of the k skeleton points.
    3. keep swaps with  E(S') <= 1.05*E0  and  A(S') < A(S)  (exact checks).
    4. if any survive -> accept the one with smallest A(S'), start a new round.
       otherwise      -> move on to the next row in the order.
    5. stop when every non-skeleton row has been tried without an admissible swap
       (= true single-swap local optimum), or when the optional budget max_scan is exhausted.

The same file also re-implements the OLD random-pool refinement (copied from
stage2_global_pilot.py, unchanged logic) so both methods can be run from the SAME S0.

Usage (put this file next to stage1.py):
    python stage2_sorted_scan.py --smoke          # 2 seeds per dataset, quick check
    python stage2_sorted_scan.py                  # 4 datasets x 40 seeds
"""
from __future__ import annotations
import argparse, csv, importlib.util, math, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # repository root (results/ lives here)

DATASET_K = {
    'nearly_collinear_clusters': 4,
    'repeated_directions': 4,
    'localized_svd': 14,
    'kernel_snapshots': 16,
}
BUILDERS = {
    'nearly_collinear_clusters': 'make_collinear_clusters',
    'repeated_directions': 'make_repeated_directions',
    'localized_svd': 'make_localized_svd',
    'kernel_snapshots': 'make_kernel_snapshots',
}
B = 4
GAMMAS = [0.5, 1.0, 1.5, 2.0, 3.0]
ERROR_GUARD = 1.05
REL_TOL = 1e-12          # "strictly smaller A" is measured with this relative margin


def load_stage1(path: Path):
    spec = importlib.util.spec_from_file_location('stage1', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------
# NEW: sorted-scan refinement
# ----------------------------------------------------------------------------
def interp_matrix(mod, X, S):
    """W = X X_S^dagger  (same computation as stage1.compute_id_diagnostics)."""
    Xs = X[S]
    Q, R = mod.spla.qr(Xs.T, mode='economic')
    return mod.spla.solve_triangular(R, (X @ Q).T, trans='N', lower=False).T


def sorted_scan_refinement(mod, X, S0, initial_error, error_guard=ERROR_GUARD,
                           max_scan=None, max_iter=500):
    """Deterministic refinement. max_scan=None -> full scan (true local optimum)."""
    X = np.asarray(X, dtype=float)
    S = np.asarray(S0, dtype=int).copy()
    n, k = X.shape[0], len(S)
    cur = mod.compute_id_diagnostics(X, S)
    if not cur['valid']:
        raise RuntimeError('Initial skeleton is invalid.')
    n_eval = 0
    hist = []
    stop_reason = 'max_iter'
    for it in range(max_iter):
        W = interp_matrix(mod, X, S)
        rn = np.linalg.norm(W, axis=1)
        rn[S] = -np.inf                                   # skeleton rows are never candidates
        order = np.argsort(-rn, kind='stable')[: n - k]   # largest ||W(i,:)|| first
        if max_scan is not None:
            order = order[:max_scan]
        accepted = None
        scanned = 0
        for add_idx in order:
            scanned += 1
            best = None
            for pos in range(k):
                S_try = S.copy()
                S_try[pos] = int(add_idx)
                d = mod.compute_id_diagnostics(X, S_try)
                n_eval += 1
                if not d['valid']:
                    continue
                if d['error_rel'] > error_guard * initial_error:
                    continue
                if not d['amp2'] < cur['amp2'] * (1.0 - REL_TOL):
                    continue
                key = (d['amp2'], d['error_rel'])
                if best is None or key < best[0]:
                    best = (key, S_try, d, pos)
            if best is not None:
                accepted = best
                break
        hist.append({'iteration': it, 'rows_scanned': scanned, 'accepted': accepted is not None,
                     'n_eval_cum': n_eval, 'amp_before': float(cur['amp2'])})
        if accepted is None:
            stop_reason = 'local_optimum' if max_scan is None else 'scan_budget_exhausted'
            break
        _, S, cur, _ = accepted
    return S, cur, hist, n_eval, stop_reason


def verify_local_optimum(mod, X, S, initial_error, error_guard=ERROR_GUARD):
    """Independent brute-force check over ALL k(n-k) swaps. Returns number of admissible swaps."""
    cur = mod.compute_id_diagnostics(X, S)
    cnt = 0
    for pos in range(len(S)):
        for i in np.setdiff1d(np.arange(X.shape[0]), S):
            S_try = S.copy(); S_try[pos] = i
            d = mod.compute_id_diagnostics(X, S_try)
            if d['valid'] and d['error_rel'] <= error_guard * initial_error \
                    and d['amp2'] < cur['amp2'] * (1.0 - REL_TOL):
                cnt += 1
    return cnt


# ----------------------------------------------------------------------------
# OLD: random residual-pool refinement (logic copied from stage2_global_pilot.py)
# ----------------------------------------------------------------------------
def old_pool_refinement(mod, X, S0, gamma, initial_error, seed, error_guard=ERROR_GUARD, max_iter=20):
    X = np.asarray(X, dtype=float)
    S = np.asarray(S0, dtype=int).copy()
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    c = int(math.ceil(gamma * B))
    cur = mod.compute_id_diagnostics(X, S)
    n_eval = 0
    for _ in range(max_iter):
        W = interp_matrix(mod, X, S)
        res = X - W @ X[S]
        e = np.sum(res * res, axis=1)
        e[S] = 0.0
        avail = np.setdiff1d(np.arange(n), S)
        tot = float(e.sum())
        if avail.size == 0 or tot <= mod.EPS:
            break
        cands = rng.choice(avail, size=min(c, avail.size), replace=False, p=e[avail] / tot)
        adm = []
        for pos in range(len(S)):
            for a in cands:
                S_try = S.copy(); S_try[pos] = int(a)
                d = mod.compute_id_diagnostics(X, S_try)
                n_eval += 1
                if not d['valid'] or d['error_rel'] > error_guard * initial_error \
                        or d['amp2'] >= cur['amp2']:
                    continue
                adm.append((d['amp2'], d['error_rel'], S_try, d))
        if not adm:
            break
        adm.sort(key=lambda z: (z[0], z[1]))
        _, _, S, cur = adm[0]
    return S, cur, n_eval


# ----------------------------------------------------------------------------
# Experiment: every method starts from the SAME S0 for a given (dataset, seed)
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'stage2_sorted_scan_results'))
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--budget-T', type=int, default=10, help='extra run: scan at most T rows per round')
    ap.add_argument('--verify', type=int, default=2, help='brute-force verify local optimum for first N seeds')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.seeds = 2
    mod = load_stage1(Path(args.stage1))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    t_all = time.perf_counter()
    for di, (ds, k) in enumerate(DATASET_K.items()):
        builder = getattr(mod, BUILDERS[ds])
        print(f'\n=== {ds} (k={k}) ===', flush=True)
        for seed in range(args.seeds):
            X = builder(n=500, d=60, seed=seed)
            rbrp_seed = 1_000_000 + di * 100_000 + seed          # independent of any method/gamma
            row_perm, *_ = mod.blocked_rpid(X, k=k, b=B, mode='rp', ortho=1, seed=rbrp_seed)
            S0 = np.asarray(row_perm[:k], dtype=int)
            d0 = mod.compute_id_diagnostics(X, S0)
            E0, A0 = float(d0['error_rel']), float(d0['amp2'])

            def record(method, Sf, df, n_eval, t, extra=None):
                r = {'dataset': ds, 'seed': seed, 'k': k, 'method': method,
                     'E0': E0, 'A0': A0, 'Ef': float(df['error_rel']), 'Af': float(df['amp2']),
                     'rho_A': float(df['amp2']) / A0, 'rho_E': float(df['error_rel']) / E0,
                     'guard_ok': bool(df['error_rel'] <= ERROR_GUARD * E0 + 1e-14),
                     'n_eval': n_eval, 'time_sec': t, 'stop_reason': '', 'rounds': np.nan}
                if extra: r.update(extra)
                rows.append(r)

            # old random-pool method, 5 gammas, same S0
            for gi, g in enumerate(GAMMAS):
                t = time.perf_counter()
                Sf, df, ne = old_pool_refinement(mod, X, S0, g, E0, seed=2_000_000 + di * 100_000 + gi * 1000 + seed)
                record(f'pool_g{g}', Sf, df, ne, time.perf_counter() - t)

            # new deterministic methods
            for name, T in [(f'sorted_T{args.budget_T}', args.budget_T), ('sorted_full', None)]:
                t = time.perf_counter()
                Sf, df, hist, ne, why = sorted_scan_refinement(mod, X, S0, E0, max_scan=T)
                record(name, Sf, df, ne, time.perf_counter() - t,
                       {'stop_reason': why, 'rounds': sum(h['accepted'] for h in hist)})
                if name == 'sorted_full' and seed < args.verify:
                    nbad = verify_local_optimum(mod, X, Sf, E0)
                    print(f'  seed {seed}: verify local optimum -> admissible swaps left = {nbad}', flush=True)
                    rows[-1]['verified_admissible_left'] = nbad
        print(f'  done, elapsed {time.perf_counter() - t_all:.1f}s', flush=True)

    fields = sorted(set().union(*(r.keys() for r in rows)))
    with open(out / 'raw_results.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=''); w.writeheader(); w.writerows(rows)

    methods = [f'pool_g{g}' for g in GAMMAS] + [f'sorted_T{args.budget_T}', 'sorted_full']
    print('\n=== OVERALL (all datasets pooled, same S0 for every method) ===')
    print(f'{"method":<14}{"mean rho_A":>11}{"median":>9}{"mean rho_E":>11}{"guard_ok":>9}{"mean n_eval":>12}{"mean sec":>9}')
    for m in methods:
        sub = [r for r in rows if r['method'] == m]
        print(f'{m:<14}{np.mean([r["rho_A"] for r in sub]):>11.4f}{np.median([r["rho_A"] for r in sub]):>9.4f}'
              f'{np.mean([r["rho_E"] for r in sub]):>11.4f}{np.mean([r["guard_ok"] for r in sub]):>9.3f}'
              f'{np.mean([r["n_eval"] for r in sub]):>12.1f}{np.mean([r["time_sec"] for r in sub]):>9.3f}')
    print('\n=== BY DATASET: mean rho_A ===')
    print(f'{"dataset":<28}' + ''.join(f'{m:>13}' for m in methods))
    for ds in DATASET_K:
        print(f'{ds:<28}' + ''.join(
            f'{np.mean([r["rho_A"] for r in rows if r["dataset"] == ds and r["method"] == m]):>13.4f}' for m in methods))
    # paired: sorted_full vs pool_g3.0
    a = {(r['dataset'], r['seed']): r['Af'] for r in rows if r['method'] == 'sorted_full'}
    b = {(r['dataset'], r['seed']): r['Af'] for r in rows if r['method'] == 'pool_g3.0'}
    keys = list(a)
    win = np.mean([a[x] < b[x] * (1 - 1e-9) for x in keys]); tie = np.mean([abs(a[x] - b[x]) <= 1e-9 * b[x] for x in keys])
    print(f'\nsorted_full vs pool_g3.0 (paired on S0): strictly better {win:.1%}, equal {tie:.1%}, worse {1 - win - tie:.1%}')
    print(f'sorted_full stop reasons: { {s: sum(r["stop_reason"] == s for r in rows if r["method"] == "sorted_full") for s in ("local_optimum", "max_iter")} }')
    print(f'total wall time {time.perf_counter() - t_all:.1f}s; results in {out}')


if __name__ == '__main__':
    main()