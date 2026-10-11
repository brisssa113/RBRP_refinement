"""Stage 2.7 sanity check: amplification-aware postprocessing of one RBRP case.

Does not modify Stage 1 core. Imports blocked_rpid and compute_id_diagnostics
from stage1_final_validation.py and adds an independent refinement routine.
"""
from __future__ import annotations

import importlib.util
import sys
import math
from pathlib import Path
import numpy as np

STAGE1 = Path('D:\\fyp\\stage1.py')
spec = importlib.util.spec_from_file_location('stage1_final_validation', STAGE1)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

blocked_rpid = mod.blocked_rpid
compute_id_diagnostics = mod.compute_id_diagnostics
make_collinear_clusters = mod.make_collinear_clusters


def amplification_aware_refinement(
    X: np.ndarray,
    S0: np.ndarray,
    b: int = 4,
    gamma: float = 2.0,
    error_guard: float = 1.05,
    max_iter: int = 100,
    seed: int = 0,
):
    """Conservative one-for-one skeleton exchange.

    At each iteration:
      1) form the exact strict-ID residual for the current skeleton set;
      2) sample c=ceil(gamma*b) non-skeleton candidates using residual^2 weights;
      3) exhaustively test k*c one-for-one exchanges;
      4) accept the admissible exchange minimizing A=||W||_2;
      5) recompute the residual after the accepted exchange.

    Acceptance: E_new <= error_guard * E_current and A_new < A_current.
    """
    X = np.asarray(X, dtype=float)
    S = np.asarray(S0, dtype=int).copy()
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    c = int(math.ceil(gamma * b))
    history = []

    def diagnostics(S):
        return compute_id_diagnostics(X, S)

    current = diagnostics(S)
    if not current['valid']:
        raise RuntimeError('Initial skeleton set is not a valid strict ID.')

    for it in range(max_iter):
        Xs = X[S]
        Q, R = mod.spla.qr(Xs.T, mode='economic')
        XQ = X @ Q
        W = mod.spla.solve_triangular(
            R, XQ.T, trans='N', lower=False
        ).T

        residual = X - W @ Xs

        row_energy = np.sum(residual**2, axis=1)
        row_energy[S] = 0.0

        total = float(np.sum(row_energy))

        available = np.setdiff1d(
            np.arange(n),
            S,
            assume_unique=False
        )

        if available.size == 0 or total <= mod.EPS:
            break

        probs = row_energy[available] / total

        c_eff = min(c, available.size)

        candidates = rng.choice(
            available,
            size=c_eff,
            replace=False,
            p=probs
        )

        admissible = []

        counts = {
            'total': 0,
            'rank_invalid': 0,
            'error_fail': 0,
            'amp_fail': 0,
            'admissible': 0
        }

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
                        d['amp2'],
                        d['error_rel'],
                        remove_pos,
                        int(remove_idx),
                        int(add_idx),
                        S_try,
                        d
                    )
                )

        rec = {
            'iteration': it,
            'error_before': current['error_rel'],
            'amp_before': current['amp2'],
            'candidate_count': int(c_eff),
            'candidate_indices': [int(x) for x in candidates],
            **{
                f'exchanges_{k}': v
                for k, v in counts.items()
            },
        }

        if not admissible:

            rec['accepted'] = False
            rec['reason'] = 'no admissible improving exchange'

            history.append(rec)
            break

        admissible.sort(
            key=lambda z: (z[0], z[1])
        )

        (
            best_amp,
            best_err,
            remove_pos,
            remove_idx,
            add_idx,
            S_new,
            d_new
        ) = admissible[0]

        rec.update({
            'accepted': True,
            'remove_position': int(remove_pos),
            'remove_index': int(remove_idx),
            'add_index': int(add_idx),
            'error_after': float(best_err),
            'amp_after': float(best_amp),
            'identity_error_after':
                float(d_new['interp_identity_error']),
        })

        history.append(rec)

        S = S_new
        current = d_new

    return S, current, history


def main():

    X = make_collinear_clusters(
        n=500,
        d=60,
        seed=0
    )

    k, b, gamma, seed = 4, 4, 2.0, 0

    c = math.ceil(gamma * b)

    row_perm, _, _, _, ranks = blocked_rpid(
        X,
        k=k,
        b=b,
        mode='rp',
        ortho=1,
        seed=seed
    )

    S0 = np.asarray(
        row_perm[:k],
        dtype=int
    )

    d0 = compute_id_diagnostics(
        X,
        S0
    )

    print(
        '=== Stage 2.7 sanity check ==='
    )

    print(
        f'dataset=nearly-collinear clusters '
        f'| n={X.shape[0]} d={X.shape[1]}'
    )

    print(
        f'k={k}, b={b}, gamma={gamma}, '
        f'c={c}, seed={seed}'
    )

    print(
        f'RBRP ranks={ranks.tolist()}'
    )

    print(
        f'S0={S0.tolist()}'
    )

    print(
        f'E0={d0["error_rel"]:.12e}'
    )

    print(
        f'A0={d0["amp2"]:.12e}'
    )

    print(
        f'max_row_l1_0={d0["amp_inf1"]:.12e}'
    )

    print(
        f'cond_Xs_0={d0["cond_Xs"]:.12e}'
    )

    print(
        f'identity_error_0='
        f'{d0["interp_identity_error"]:.12e}'
    )

    Sf, df, hist = amplification_aware_refinement(
        X,
        S0,
        b=b,
        gamma=gamma,
        error_guard=1.05,
        seed=seed
    )

    print('\n--- refinement history ---')

    for h in hist:

        print(
            f"it={h['iteration']} | "
            f"E {h['error_before']:.6e} "
            f"-> {h.get('error_after', np.nan):.6e} "
            f"| A {h['amp_before']:.6e} "
            f"-> {h.get('amp_after', np.nan):.6e} "
            f"| candidates={h['candidate_count']} "
            f"| exchanges={h['exchanges_total']} "
            f"(rank-invalid="
            f"{h['exchanges_rank_invalid']}, "
            f"error-fail="
            f"{h['exchanges_error_fail']}, "
            f"amp-fail="
            f"{h['exchanges_amp_fail']}, "
            f"admissible="
            f"{h['exchanges_admissible']}) "
            f"| accepted={h['accepted']}"
            +
            (
                f" | remove={h['remove_index']} "
                f"add={h['add_index']}"
                if h['accepted']
                else ''
            )
        )

    print('\n--- final ---')

    print(
        f'Sf={Sf.tolist()}'
    )

    print(
        f'Ef={df["error_rel"]:.12e}'
    )

    print(
        f'Af={df["amp2"]:.12e}'
    )

    print(
        f'rho_E=Ef/E0='
        f'{df["error_rel"] / d0["error_rel"]:.12e}'
    )

    print(
        f'rho_A=Af/A0='
        f'{df["amp2"] / d0["amp2"]:.12e}'
    )

    print(
        f'max_row_l1_f='
        f'{df["amp_inf1"]:.12e}'
    )

    print(
        f'cond_Xs_f='
        f'{df["cond_Xs"]:.12e}'
    )

    print(
        f'identity_error_f='
        f'{df["interp_identity_error"]:.12e}'
    )

    print(
        f'accepted_exchanges='
        f'{sum(int(h["accepted"]) for h in hist)}'
    )

    print(
        f'iterations={len(hist)}'
    )

    assert len(S0) == k
    assert len(set(S0.tolist())) == k

    assert len(Sf) == k
    assert len(set(Sf.tolist())) == k

    assert d0['valid']
    assert df['valid']

    assert d0['interp_identity_error'] < 1e-8
    assert df['interp_identity_error'] < 1e-8

    assert (
        df['error_rel']
        <= 1.05 * d0['error_rel'] + 1e-14
    )

    assert (
        df['amp2']
        <= d0['amp2'] + 1e-14
    )

    print('\nSANITY_CHECK: PASS')


if __name__ == '__main__':
    main()