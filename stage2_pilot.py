"""Stage 2.8 pilot: RBRP + amplification-aware refinement.
Runs 4 datasets x 5 gamma values x 40 seeds = 800 experiments.
The Stage 1 implementation is imported but not modified.
Main metrics:
    rho_A = A_final / A_initial
    rho_E = E_final / E_initial
Acceptance rule inside refinement:
    E_new <= 1.05 * E_current
    A_new < A_current
Candidate pool:
    c = ceil(gamma * b)
    sampled without replacement using residual‑squared probabilities.
For reproducibility, each (dataset, gamma, seed) gets an independent RNG seed
constructed deterministically from the loop indices.
"""
from __future__ import annotations
import csv
import importlib
import sys
import math
import time
import platform
from pathlib import Path
import numpy as np
import importlib.util


STAGE1 = Path(r'D:\\fyp\\stage1.py')
OUT_DIR = Path(r'D:\\fyp\\stage2_pilot_results')
OUT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('stage1', STAGE1)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

blocked_rpid = mod.blocked_rpid
compute_id_diagnostics = mod.compute_id_diagnostics
make_collinear_clusters = mod.make_collinear_clusters
make_repeated_directions = mod.make_repeated_directions
make_localized_svd = mod.make_localized_svd
make_kernel_snapshots = mod.make_kernel_snapshots
spla = mod.spla
EPS = mod.EPS

DATASETS = {
    'nearly_collinear_clusters': {
        'k': 4,
        'builder': lambda seed: make_collinear_clusters(
            n=500, d=60, seed=seed
        ),
    },
    'repeated_directions': {
        'k': 4,
        'builder': lambda seed: make_repeated_directions(
            n=500, d=60, seed=seed
        ),
    },
    'localized_svd': {
        'k': 14,
        'builder': lambda seed: make_localized_svd(
            n=500, d=60, seed=seed
        ),
    },
    'kernel_snapshots': {
        'k': 16,
        'builder': lambda seed: make_kernel_snapshots(
            n=500, d=60, seed=seed
        ),
    },
}
B = 4
GAMMAS = [0.5, 1.0, 1.5, 2.0, 3.0]
N_SEEDS = 40
ERROR_GUARD = 1.05
MAX_ITER = 20
RUN_TIMEOUT_SEC = 10.0
IS_UNIX = platform.system() in ("Linux", "Darwin")

# ---------------- refinement ----------------
def amplification_aware_refinement(
    X: np.ndarray,
    S0: np.ndarray,
    b: int,
    gamma: float,
    error_guard: float,
    max_iter: int,
    seed: int,
):
    X = np.asarray(X, dtype=float)
    S = np.asarray(S0, dtype=int).copy()
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    c = int(math.ceil(gamma * b))
    history = []

    def diagnostics(S_):
        return compute_id_diagnostics(X, S_)

    current = diagnostics(S)
    if not current['valid']:
        raise RuntimeError('Initial skeleton set is not a valid strict ID.')

    for it in range(max_iter):
        Xs = X[S]
        Q, R = spla.qr(Xs.T, mode='economic')
        XQ = X @ Q
        W = spla.solve_triangular(
            R, XQ.T, trans='N', lower=False
        ).T
        residual = X - W @ Xs
        row_energy = np.sum(residual * residual, axis=1)
        row_energy[S] = 0.0
        available = np.setdiff1d(np.arange(n), S, assume_unique=False)
        total = float(np.sum(row_energy))

        if available.size == 0 or total <= EPS:
            history.append({
                'iteration': it,
                'accepted': False,
                'reason': 'no_available_candidate_or_zero_residual',
                'candidate_count': 0,
                'exchanges_total': 0,
                'exchanges_rank_invalid': 0,
                'exchanges_error_fail': 0,
                'exchanges_amp_fail': 0,
                'exchanges_admissible': 0,
                'error_before': float(current['error_rel']),
                'amp_before': float(current['amp2']),
                'error_after': np.nan,
                'amp_after': np.nan,
                'remove_index': np.nan,
                'add_index': np.nan,
            })
            break

        probs = row_energy[available] / total
        c_eff = min(c, available.size)
        candidates = rng.choice(
            available, size=c_eff, replace=False, p=probs
        )

        counts = {
            'total': 0,
            'rank_invalid': 0,
            'error_fail': 0,
            'amp_fail': 0,
            'admissible': 0,
        }
        admissible = []

        for remove_pos, remove_idx in enumerate(S):
            for add_idx in candidates:
                counts['total'] += 1
                S_try = S.copy()
                S_try[remove_pos] = int(add_idx)
                d = diagnostics(S_try)
                if not d['valid']:
                    counts['rank_invalid'] += 1
                    continue
                if d['error_rel'] > error_guard * current['error_rel']:
                    counts['error_fail'] += 1
                    continue
                if d['amp2'] >= current['amp2']:
                    counts['amp_fail'] += 1
                    continue
                counts['admissible'] += 1
                admissible.append(
                    (
                        float(d['amp2']),
                        float(d['error_rel']),
                        int(remove_pos),
                        int(remove_idx),
                        int(add_idx),
                        S_try,
                        d,
                    )
                )

        rec = {
            'iteration': it,
            'accepted': False,
            'reason': '',
            'candidate_count': int(c_eff),
            'exchanges_total': counts['total'],
            'exchanges_rank_invalid': counts['rank_invalid'],
            'exchanges_error_fail': counts['error_fail'],
            'exchanges_amp_fail': counts['amp_fail'],
            'exchanges_admissible': counts['admissible'],
            'error_before': float(current['error_rel']),
            'amp_before': float(current['amp2']),
            'error_after': np.nan,
            'amp_after': np.nan,
            'remove_index': np.nan,
            'add_index': np.nan,
        }

        if not admissible:
            rec['reason'] = 'no_admissible_improving_exchange'
            history.append(rec)
            break

        admissible.sort(key=lambda z: (z[0], z[1]))
        best_amp, best_err, remove_pos, remove_idx, add_idx, S_new, d_new = admissible[0]
        rec.update({
            'accepted': True,
            'reason': 'accepted_best_admissible',
            'error_after': best_err,
            'amp_after': best_amp,
            'remove_index': remove_idx,
            'add_index': add_idx,
        })
        history.append(rec)
        S = S_new
        current = d_new

    return S, current, history


class RunTimeout(Exception):
    pass


# -------- cross‑platform timeout wrapper --------
def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """Unix: signal timeout; Windows: wall‑clock soft check (no hard kill)."""
    if IS_UNIX:
        import signal
        def _timeout_handler(signum, frame):
            raise RunTimeout('single run exceeded wall‑time limit')
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        try:
            res = func(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)
        return res
    else:
        # Windows no SIGALRM: soft check only; cannot interrupt compute
        t_start = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = time.perf_counter() - t_start
        if elapsed > timeout_sec:
            raise RunTimeout(f"wall‑time exceed {timeout_sec}s (soft check)")
        return res


def run_one(dataset_name: str, X: np.ndarray, k: int, gamma: float, seed: int):
    t0 = time.perf_counter()
    row_perm, _, _, _, ranks = blocked_rpid(
        X,
        k=k,
        b=B,
        mode='rp',
        ortho=1,
        seed=seed,
    )
    S0 = np.asarray(row_perm[:k], dtype=int)
    d0 = compute_id_diagnostics(X, S0)
    if not d0['valid']:
        raise RuntimeError('RBRP initial skeleton is invalid.')

    Sf, df, hist = amplification_aware_refinement(
        X, S0, b=B, gamma=gamma,
        error_guard=ERROR_GUARD,
        max_iter=MAX_ITER,
        seed=seed,
    )
    rho_A = float(df['amp2'] / d0['amp2']) if d0['amp2'] > 0 else np.nan
    rho_E = float(df['error_rel'] / d0['error_rel']) if d0['error_rel'] > 0 else np.nan
    accepted = sum(int(h['accepted']) for h in hist)
    guard_ok = bool(df['error_rel'] <= ERROR_GUARD * d0['error_rel'] + 1e-14)
    amp_improved = bool(df['amp2'] < d0['amp2'] - 1e-14)
    success = bool(guard_ok and amp_improved)
    row = {
        'dataset': dataset_name,
        'seed': seed,
        'k': k,
        'b': B,
        'gamma': gamma,
        'candidate_count_c': int(math.ceil(gamma * B)),
        'initial_E': float(d0['error_rel']),
        'final_E': float(df['error_rel']),
        'rho_E': rho_E,
        'initial_A': float(d0['amp2']),
        'final_A': float(df['amp2']),
        'rho_A': rho_A,
        'initial_max_row_l1': float(d0['amp_inf1']),
        'final_max_row_l1': float(df['amp_inf1']),
        'initial_cond_Xs': float(d0['cond_Xs']),
        'final_cond_Xs': float(df['cond_Xs']),
        'initial_identity_error': float(d0['interp_identity_error']),
        'final_identity_error': float(df['interp_identity_error']),
        'accepted_exchanges': accepted,
        'iterations': len(hist),
        'guard_ok': guard_ok,
        'amp_improved': amp_improved,
        'success': success,
        'runtime_sec': float(time.perf_counter() - t0),
        'rbrp_ranks': str(ranks.tolist()),
        'S0': str(S0.tolist()),
        'Sf': str(Sf.tolist()),
        'status': 'ok',
        'error_message': '',
    }
    return row, hist


# Pre‑define fixed CSV headers to avoid IndexError when rows list empty
MAIN_CSV_FIELDS = [
    'dataset', 'seed', 'k', 'b', 'gamma', 'candidate_count_c',
    'initial_E', 'final_E', 'rho_E', 'initial_A', 'final_A', 'rho_A',
    'initial_max_row_l1', 'final_max_row_l1', 'initial_cond_Xs', 'final_cond_Xs',
    'initial_identity_error', 'final_identity_error', 'accepted_exchanges', 'iterations',
    'guard_ok', 'amp_improved', 'success', 'runtime_sec', 'rbrp_ranks',
    'S0', 'Sf', 'status', 'error_message', 'data_seed', 'refinement_seed'
]

HIST_FIELDS = [
    'dataset','data_seed','refinement_seed','gamma','iteration','accepted','reason',
    'candidate_count','exchanges_total','exchanges_rank_invalid','exchanges_error_fail',
    'exchanges_amp_fail','exchanges_admissible','error_before','amp_before',
    'error_after','amp_after','remove_index','add_index'
]

SUMMARY_FIELDS = [
    'dataset','gamma','c','n','n_total','n_timeout','n_error',
    'mean_rho_A','median_rho_A','q25_rho_A','q75_rho_A',
    'mean_rho_E','median_rho_E','success_rate','amp_improved_rate',
    'guard_ok_rate','mean_accepted_exchanges','mean_runtime_sec'
]

AGGREGATE_FIELDS = [
    'scope','gamma','c','n','n_total','n_timeout','n_error',
    'mean_rho_A','median_rho_A','mean_rho_E','success_rate',
    'amp_improved_rate','guard_ok_rate'
]


def safe_stats(arr):
    """Return dict with stats, handle empty array gracefully."""
    if len(arr) == 0:
        return {'mean':np.nan,'median':np.nan,'q25':np.nan,'q75':np.nan}
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.quantile(arr,0.25)),
        'q75': float(np.quantile(arr,0.75))
    }


def main():
    import os
    rows = []
    history_rows = []
    run_id = 0
    wall0 = time.perf_counter()
    total_runs = len(DATASETS) * len(GAMMAS) * N_SEEDS
    print(f'Stage 2.8 pilot: {total_runs} runs')
    print(f'b={B}, gammas={GAMMAS}, seeds={N_SEEDS}, error_guard={ERROR_GUARD}')
    print(f'Platform={platform.system()}, hard‑timeout_supported={IS_UNIX}')

    dataset_filter = os.environ.get('DATASET_FILTER')
    datasets_to_run = {k:v for k,v in DATASETS.items() if (not dataset_filter or k == dataset_filter)}

    for dataset_index, (dataset_name, spec_ds) in enumerate(datasets_to_run.items()):
        k = spec_ds['k']
        print(f'\n=== DATASET: {dataset_name} | k={k} ===')
        for gamma_index, gamma in enumerate(GAMMAS):
            gamma_rows = []
            for seed in range(N_SEEDS):
                X = spec_ds['builder'](seed)
                refinement_seed = 1_000_000 + dataset_index * 100_000 + gamma_index * 1_000 + seed
                try:
                    row, hist = run_with_timeout(
                        run_one, RUN_TIMEOUT_SEC,
                        dataset_name, X, k, gamma, refinement_seed
                    )
                    row['status'] = 'ok'
                    row['data_seed'] = seed
                    row['refinement_seed'] = refinement_seed
                except RunTimeout as exc:
                    row = {f:np.nan for f in MAIN_CSV_FIELDS}
                    row.update({
                        'dataset': dataset_name,
                        'seed': refinement_seed,
                        'data_seed': seed,
                        'refinement_seed': refinement_seed,
                        'k': k, 'b': B, 'gamma': gamma,
                        'candidate_count_c': int(math.ceil(gamma*B)),
                        'runtime_sec': RUN_TIMEOUT_SEC,
                        'status': 'timeout',
                        'error_message': str(exc)
                    })
                    hist = []
                except Exception as exc:
                    row = {f:np.nan for f in MAIN_CSV_FIELDS}
                    row.update({
                        'dataset': dataset_name,
                        'seed': refinement_seed,
                        'data_seed': seed,
                        'refinement_seed': refinement_seed,
                        'k': k, 'b': B, 'gamma': gamma,
                        'candidate_count_c': int(math.ceil(gamma*B)),
                        'runtime_sec': np.nan,
                        'status': 'error',
                        'error_message': repr(exc)
                    })
                    hist = []

                rows.append(row)
                gamma_rows.append(row)
                for h in hist:
                    history_rows.append({
                        'dataset': dataset_name,
                        'data_seed': seed,
                        'refinement_seed': refinement_seed,
                        'gamma': gamma,
                        **h,
                    })
                run_id +=1
                if run_id % 20 ==0 or run_id == total_runs:
                    print(
                        f'progress {run_id:4d}/{total_runs} | '
                        f'elapsed={time.perf_counter()-wall0:.1f}s'
                    )

            valid_gamma_rows = [r for r in gamma_rows if r.get('status') == 'ok' and np.isfinite(r['rho_A'])]
            arr_A = np.array([r['rho_A'] for r in valid_gamma_rows], dtype=float)
            arr_E = np.array([r['rho_E'] for r in valid_gamma_rows], dtype=float)
            statsA = safe_stats(arr_A)
            statsE = safe_stats(arr_E)
            amp_arr = np.array([r['amp_improved'] for r in valid_gamma_rows], dtype=bool)
            guard_arr = np.array([r['guard_ok'] for r in valid_gamma_rows], dtype=bool)
            succ_arr = np.array([r['success'] for r in valid_gamma_rows], dtype=bool)
            mean_rho_A, med_rho_A = statsA['mean'], statsA['median']
            mean_rho_E = statsE['mean']
            amp_rate = float(np.mean(amp_arr)) if len(amp_arr)>0 else np.nan
            guard_rate = float(np.mean(guard_arr)) if len(guard_arr)>0 else np.nan
            success_rate = float(np.mean(succ_arr)) if len(succ_arr)>0 else np.nan

            print(
                f'gamma={gamma:<3} c={math.ceil(gamma*B):<2} | '
                f'mean rho_A={mean_rho_A:.4f} | median rho_A={med_rho_A:.4f} | '
                f'mean rho_E={mean_rho_E:.4f} | '
                f'amp_improved={amp_rate:.3f} | guard_ok={guard_rate:.3f} | '
                f'success={success_rate:.3f} | '
                f'timeouts={sum(r.get("status")=="timeout" for r in gamma_rows)} | '
                f'errors={sum(r.get("status")=="error" for r in gamma_rows)}'
            )

    # Write main result CSV
    with open(OUT_DIR / 'stage2_step2_8_raw_results.csv', 'w', newline='', encoding='utf‑8') as f:
        w = csv.DictWriter(f, fieldnames=MAIN_CSV_FIELDS, restval='')
        w.writeheader()
        w.writerows(rows)

    # Write refinement history CSV
    with open(OUT_DIR / 'stage2_step2_8_refinement_history.csv', 'w', newline='', encoding='utf‑8') as f:
        w = csv.DictWriter(f, fieldnames=HIST_FIELDS, restval='')
        w.writeheader()
        w.writerows(history_rows)

    # Build summary
    summary = []
    for dataset_name in datasets_to_run:
        for gamma in GAMMAS:
            sub = [r for r in rows if r['dataset'] == dataset_name and r['gamma'] == gamma]
            valid_sub = [r for r in sub if r.get('status') == 'ok' and np.isfinite(r['rho_A'])]
            arr_A = np.array([r['rho_A'] for r in valid_sub], dtype=float)
            arr_E = np.array([r['rho_E'] for r in valid_sub], dtype=float)
            statsA = safe_stats(arr_A)
            statsE = safe_stats(arr_E)
            success = np.array([r['success'] for r in valid_sub], dtype=bool)
            amp_improved = np.array([r['amp_improved'] for r in valid_sub], dtype=bool)
            guard_ok = np.array([r['guard_ok'] for r in valid_sub], dtype=bool)
            mean_acc_ex = float(np.mean([r['accepted_exchanges'] for r in valid_sub])) if valid_sub else np.nan
            mean_rt = float(np.mean([r['runtime_sec'] for r in valid_sub])) if valid_sub else np.nan
            summary.append({
                'dataset': dataset_name,
                'gamma': gamma,
                'c': math.ceil(gamma * B),
                'n': len(valid_sub),
                'n_total': len(sub),
                'n_timeout': sum(r.get('status') == 'timeout' for r in sub),
                'n_error': sum(r.get('status') == 'error' for r in sub),
                'mean_rho_A': statsA['mean'],
                'median_rho_A': statsA['median'],
                'q25_rho_A': statsA['q25'],
                'q75_rho_A': statsA['q75'],
                'mean_rho_E': statsE['mean'],
                'median_rho_E': statsE['median'],
                'success_rate': float(np.mean(success)) if len(success)>0 else np.nan,
                'amp_improved_rate': float(np.mean(amp_improved)) if len(amp_improved)>0 else np.nan,
                'guard_ok_rate': float(np.mean(guard_ok)) if len(guard_ok)>0 else np.nan,
                'mean_accepted_exchanges': mean_acc_ex,
                'mean_runtime_sec': mean_rt,
            })
    with open(OUT_DIR / 'stage2_step2_8_summary_by_dataset_gamma.csv', 'w', newline='', encoding='utf‑8') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval='')
        w.writeheader()
        w.writerows(summary)

    # Build aggregate
    aggregate = []
    for gamma in GAMMAS:
        sub = [r for r in rows if r['gamma'] == gamma]
        valid_sub = [r for r in sub if r.get('status') == 'ok' and np.isfinite(r['rho_A'])]
        arr_A = np.array([r['rho_A'] for r in valid_sub], dtype=float)
        arr_E = np.array([r['rho_E'] for r in valid_sub], dtype=float)
        statsA = safe_stats(arr_A)
        statsE = safe_stats(arr_E)
        succ_arr = np.array([r['success'] for r in valid_sub], dtype=bool)
        amp_arr = np.array([r['amp_improved'] for r in valid_sub], dtype=bool)
        guard_arr = np.array([r['guard_ok'] for r in valid_sub], dtype=bool)
        aggregate.append({
            'scope': 'all_datasets',
            'gamma': gamma,
            'c': math.ceil(gamma * B),
            'n': len(valid_sub),
            'n_total': len(sub),
            'n_timeout': sum(r.get('status') == 'timeout' for r in sub),
            'n_error': sum(r.get('status') == 'error' for r in sub),
            'mean_rho_A': statsA['mean'],
            'median_rho_A': statsA['median'],
            'mean_rho_E': statsE['mean'],
            'success_rate': float(np.mean(succ_arr)) if len(succ_arr)>0 else np.nan,
            'amp_improved_rate': float(np.mean(amp_arr)) if len(amp_arr)>0 else np.nan,
            'guard_ok_rate': float(np.mean(guard_arr)) if len(guard_arr)>0 else np.nan,
        })
    # all runs summary
    valid_all = [r for r in rows if r.get('status') == 'ok' and np.isfinite(r['rho_A'])]
    arrA_all = np.array([r['rho_A'] for r in valid_all], dtype=float)
    arrE_all = np.array([r['rho_E'] for r in valid_all], dtype=float)
    statsA_all = safe_stats(arrA_all)
    statsE_all = safe_stats(arrE_all)
    succ_all = np.array([r['success'] for r in valid_all], dtype=bool)
    amp_all = np.array([r['amp_improved'] for r in valid_all], dtype=bool)
    guard_all = np.array([r['guard_ok'] for r in valid_all], dtype=bool)
    aggregate.append({
        'scope': 'all_800_runs',
        'gamma': np.nan,
        'c': np.nan,
        'n': len(rows),
        'n_total': len(rows),
        'n_timeout': sum(r.get('status') == 'timeout' for r in rows),
        'n_error': sum(r.get('status') == 'error' for r in rows),
        'mean_rho_A': statsA_all['mean'],
        'median_rho_A': statsA_all['median'],
        'mean_rho_E': statsE_all['mean'],
        'success_rate': float(np.mean(succ_all)) if len(succ_all)>0 else np.nan,
        'amp_improved_rate': float(np.mean(amp_all)) if len(amp_all)>0 else np.nan,
        'guard_ok_rate': float(np.mean(guard_all)) if len(guard_all)>0 else np.nan,
    })
    with open(OUT_DIR / 'stage2_step2_8_aggregate.csv', 'w', newline='', encoding='utf‑8') as f:
        w = csv.DictWriter(f, fieldnames=AGGREGATE_FIELDS, restval='')
        w.writeheader()
        w.writerows(aggregate)

    # Text report
    valid_rows = [r for r in rows if r.get('status') == 'ok' and np.isfinite(r['rho_A'])]
    succ_all_rate = float(np.mean([r['success'] for r in valid_rows])) if valid_rows else np.nan
    amp_all_rate = float(np.mean([r['amp_improved'] for r in valid_rows])) if valid_rows else np.nan
    guard_all_rate = float(np.mean([r['guard_ok'] for r in valid_rows])) if valid_rows else np.nan
    arrA = np.array([r['rho_A'] for r in valid_rows], dtype=float)
    arrE = np.array([r['rho_E'] for r in valid_rows], dtype=float)
    sA = safe_stats(arrA)
    sE = safe_stats(arrE)

    with open(OUT_DIR / 'stage2_step2_8_report.txt', 'w', encoding='utf‑8') as f:
        f.write('Stage 2.8 Pilot Report\n')
        f.write('======================\n\n')
        f.write(f'Total runs: {len(rows)}\n')
        f.write(f'Dataset filter: {dataset_filter or "ALL"}\n')
        f.write(f'b: {B}\n')
        f.write(f'gammas: {GAMMAS}\n')
        f.write(f'seeds per dataset/gamma: {N_SEEDS}\n')
        f.write(f'error guard: {ERROR_GUARD}\n\n')
        f.write('Overall\n')
        f.write(f'  success rate (rho_A < 1 and rho_E <= 1.05): {succ_all_rate:.6f}\n')
        f.write(f'  amplification improvement rate: {amp_all_rate:.6f}\n')
        f.write(f'  error‑guard satisfaction rate: {guard_all_rate:.6f}\n')
        f.write(f'  mean rho_A: {sA["mean"]:.6f}\n')
        f.write(f'  median rho_A: {sA["median"]:.6f}\n')
        f.write(f'  mean rho_E: {sE["mean"]:.6f}\n')
        f.write(f'  median rho_E: {sE["median"]:.6f}\n')
        f.write(f'Valid runs: {len(valid_rows)}\n')
        f.write(f'Timeouts: {sum(r.get("status") == "timeout" for r in rows)}\n')
        f.write(f'Errors: {sum(r.get("status") == "error" for r in rows)}\n')
        f.write(f'Wall time sec: {time.perf_counter()-wall0:.3f}\n')
        f.write('\nSee CSV files for complete per‑run results.\n')

    print('\n=== FINAL OVERALL ===')
    print(f'runs={len(rows)} | valid={len(valid_rows)} | '
          f'timeouts={sum(r.get("status")=="timeout" for r in rows)} | '
          f'errors={sum(r.get("status")=="error" for r in rows)}')
    print(f'success_rate={succ_all_rate:.6f}')
    print(f'amplification_improvement_rate={amp_all_rate:.6f}')
    print(f'error_guard_ok_rate={guard_all_rate:.6f}')
    print(f'mean_rho_A={sA["mean"]:.6f}')
    print(f'median_rho_A={sA["median"]:.6f}')
    print(f'mean_rho_E={sE["mean"]:.6f}')
    print(f'median_rho_E={sE["median"]:.6f}')
    print(f'wall_time_sec={time.perf_counter()-wall0:.3f}')
    print(f'results_dir={OUT_DIR}')


if __name__ == '__main__':
    import importlib.util
    main()
