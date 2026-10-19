"""
Stage 1 final validation for amplification-aware RBRP research.

Research question
-----------------
Does small / similar strict-ID reconstruction error imply small / similar
interpolation amplification for skeletons produced by RBRP?

This script intentionally DOES NOT modify the RBRP selection rule. It contains:
  A. deterministic exact-rank counterexample;
  B. noisy controlled exhaustive fixed-rank skeleton study;
  C. multi-seed RBRP experiments on four controlled matrix families;
  D. predeclared matched-rank + matched-error gap statistics;
  E. figures and CSV/JSON outputs for reproducibility.

The RBRP core follows the original GitHub notebook implementation:
residual-weighted sampling + local CPQR filtering, with only the RNG seed
made explicit for reproducible multi-trial experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as spla
import matplotlib.pyplot as plt

EPS = np.finfo(float).eps


# -----------------------------------------------------------------------------
# 1. Original RBRP core (selection rule unchanged; explicit seed only)
# -----------------------------------------------------------------------------

def _block_size_local_clip(R: np.ndarray, tol: float) -> int:
    """Same local clipping rule as the original notebook."""
    rncs = np.flip(np.flip(spla.norm(R, ord=2, axis=1) ** 2).cumsum())
    if rncs[0] <= 0:
        return 0
    relative_rncs = rncs / rncs[0]
    return R.shape[0] if relative_rncs[-1] >= tol else int(np.argmax(relative_rncs < tol))


def blocked_rpid(
    X: np.ndarray,
    k: int,
    b: int = 10,
    tol: float | None = None,
    mode: str = "rp",
    ortho: int = 1,
    tol_loc: float | None = None,
    seed: int = 0,
):
    """RBRP copied structurally from the author's notebook.

    For Stage 1 we use mode='rp', ortho=1.  The only research-facing change is
    replacing the notebook-global SEED by an explicit seed argument.
    """
    X = np.asarray(X, dtype=np.double)
    n, d = X.shape
    if not (1 <= k <= min(n, d)):
        raise ValueError("k must satisfy 1 <= k <= min(n, d)")
    if b < 1:
        raise ValueError("b must be >= 1")

    row_perm = np.arange(n)
    rng = np.random.default_rng(seed=seed)
    tol_loc = 1 / b if tol_loc is None else float(tol_loc)

    t0 = time.time()
    diags = spla.norm(X, ord=2, axis=1).astype(np.double) ** 2
    residual = float(np.sum(diags))
    residual0 = residual
    time_pivot = [time.time() - t0]

    L = np.zeros((n, k), dtype=np.double)
    Q = np.zeros((d, k), dtype=np.double)
    residuals = [residual]
    ranks = []

    i = 0
    while i < k:
        t0 = time.time()
        remaining = row_perm[i:]
        nz = int(np.count_nonzero(diags[remaining] > 0))
        block_size = min(k - i, b, nz)
        if block_size <= 0:
            break

        weights = diags[remaining]
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            break

        if mode == "rp":
            js = i + rng.choice(
                n - i,
                size=block_size,
                p=weights / weight_sum,
                replace=False,
            )
        elif mode == "gp":
            js = i + np.argpartition(weights, -block_size)[-block_size:]
        else:
            raise RuntimeError(f"Algorithm '{mode}' not recognized")

        if ortho > 0:
            if i == 0:
                V = X[row_perm[js]].T
            else:
                V = X[row_perm[js]].T - Q[:, :i] @ (Q[:, :i].T @ X[row_perm[js]].T)

            QV, RV, pV = spla.qr(V, mode="economic", pivoting=True)
            block_size = _block_size_local_clip(RV, tol_loc)
            if block_size <= 0:
                break
            js = js[pV[:block_size]]

            # Preserve the original notebook's safe swap form.
            aux = row_perm[i : i + block_size].copy()
            row_perm[i : i + block_size] = row_perm[js]
            row_perm[js] = aux
            Q[:, i : i + block_size] = QV[:, :block_size]

            if ortho == 1:
                L[row_perm[i:], i : i + block_size] = X[row_perm[i:]] @ Q[:, i : i + block_size]
            else:
                raise RuntimeError("Stage 1 final script validates the published ortho=1 RBRP path only")
        else:
            raise RuntimeError("Stage 1 final script validates ortho=1 only")

        tail = row_perm[i + block_size :]
        if tail.size:
            diags[tail] -= np.sum(L[tail, i : i + block_size] ** 2, axis=1)
        diags = np.clip(diags, 0.0, None).astype(np.double)

        i += block_size
        residual = float(np.sum(diags[row_perm[i:]]))
        time_pivot.append(time.time() - t0 + time_pivot[-1])
        residuals.append(residual)
        ranks.append(i)

        if tol is not None and residual < residual0 * tol:
            break
        if i >= n or np.count_nonzero(diags[row_perm[i:]] > 0) < 1:
            break

    L = L[:, :i]
    residuals = np.asarray(residuals[1:], dtype=float) / max(residual0, EPS)
    ranks = np.asarray(ranks, dtype=int)
    time_pivot = np.asarray(time_pivot[1:], dtype=float)
    return row_perm, L, residuals, time_pivot, ranks


# -----------------------------------------------------------------------------
# 2. Strict-ID diagnostics
# -----------------------------------------------------------------------------

def compute_id_diagnostics(X: np.ndarray, skeleton: np.ndarray, rank_tol: float = 1e-12):
    """Compute strict-ID W = X X_S^dagger and stability diagnostics.

    A rank-deficient skeleton is declared invalid rather than silently regularized.
    """
    X = np.asarray(X, dtype=float)
    skeleton = np.asarray(skeleton, dtype=int)
    Xs = X[skeleton]
    k = len(skeleton)

    s = spla.svdvals(Xs)
    if len(s) == 0 or s[-1] <= rank_tol * max(s[0], 1.0):
        return {
            "valid": False,
            "rank": k,
            "error_rel": np.nan,
            "amp2": np.nan,
            "amp_inf1": np.nan,
            "cond_Xs": np.inf,
            "interp_identity_error": np.nan,
        }

    # Economy QR of X_S^T.  For full row rank X_S, X_S^dagger = Q R^{-T}.
    Q, R = spla.qr(Xs.T, mode="economic")
    XQ = X @ Q
    W = spla.solve_triangular(R, XQ.T, trans="N", lower=False).T

    denom = spla.norm(X, ord="fro")
    err = spla.norm(X - W @ Xs, ord="fro") / max(denom, EPS)
    amp2 = spla.norm(W, ord=2)
    amp_inf1 = float(np.max(np.sum(np.abs(W), axis=1)))
    cond_Xs = float(s[0] / s[-1])
    identity_error = spla.norm(W[skeleton] - np.eye(k), ord=np.inf)

    return {
        "valid": True,
        "rank": k,
        "error_rel": float(err),
        "amp2": float(amp2),
        "amp_inf1": amp_inf1,
        "cond_Xs": cond_Xs,
        "interp_identity_error": float(identity_error),
    }


# -----------------------------------------------------------------------------
# 3. Controlled matrix families
# -----------------------------------------------------------------------------

def make_exact_counterexample(n: int = 200, eps: float = 1e-4):
    """Exact rank-2 example with equally accurate but very differently amplified IDs."""
    X = np.empty((n, 2), dtype=float)
    X[0] = [1.0, 0.0]
    X[1] = [0.0, 1.0]
    X[2] = [1.0, eps]
    X[3] = [1.0, 2 * eps]
    # Deterministic interior rows, all in the same exact 2D row space.
    t = np.linspace(-2.0, 2.0, n - 4)
    X[4:, 0] = 1.0 + 0.2 * np.cos(3 * t)
    X[4:, 1] = 0.3 + 0.7 * np.sin(t)
    return X, np.array([0, 1]), np.array([0, 2])


def make_noisy_pair_matrix(n: int = 160, d: int = 20, noise: float = 2e-3, seed: int = 123):
    """Approximate rank-2 matrix with many nearly collinear rows and ambient noise."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    angles = np.concatenate([
        rng.normal(0.0, 2e-3, int(0.65 * n)),
        rng.uniform(0.08, np.pi / 2, n - int(0.65 * n)),
    ])
    scales = rng.uniform(0.7, 1.4, n)
    coeff = scales[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])
    X = coeff @ q.T + noise * rng.normal(size=(n, d))
    return X


def make_collinear_clusters(n: int = 500, d: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = 12
    B, _ = np.linalg.qr(rng.normal(size=(d, r)))
    centers = np.eye(r)
    probs = np.array([0.28, 0.18, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015])
    probs = probs / probs.sum()
    labels = rng.choice(r, size=n, p=probs)
    Z = centers[labels] + 0.025 * rng.normal(size=(n, r))
    row_scale = np.exp(rng.normal(0.0, 0.9, size=n))
    X = (Z * row_scale[:, None]) @ B.T
    X += 7e-4 * rng.normal(size=(n, d))
    return X


def make_localized_svd(n: int = 500, d: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = 18
    # Deliberately heterogeneous row leverage before orthonormalization.
    G = rng.normal(size=(n, r))
    G[: max(2 * r, n // 12)] *= 5.0
    U, _ = np.linalg.qr(G)
    V, _ = np.linalg.qr(rng.normal(size=(d, r)))
    s = np.geomspace(1.0, 2e-3, r)
    X = (U * s) @ V.T
    X += 2e-5 * rng.normal(size=(n, d))
    return X


def make_repeated_directions(n: int = 500, d: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    r = 10
    B, _ = np.linalg.qr(rng.normal(size=(d, r)))
    base = rng.normal(size=(r, r))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    labels = rng.integers(0, r, size=n)
    Z = base[labels].copy()
    exact_mask = rng.random(n) < 0.45
    Z[~exact_mask] += 5e-3 * rng.normal(size=(np.sum(~exact_mask), r))
    scales = rng.lognormal(mean=0.0, sigma=0.5, size=n)
    X = (Z * scales[:, None]) @ B.T
    X += 3e-4 * rng.normal(size=(n, d))
    return X


def make_kernel_snapshots(n: int = 500, d: int = 60, seed: int = 0):
    """Parameterized smooth snapshot family (kernel/PDE-like, fully synthetic)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, d)
    mu = rng.uniform(0.02, 0.98, size=n)
    width = rng.uniform(0.035, 0.15, size=n)
    amp = rng.lognormal(mean=0.0, sigma=0.35, size=n)
    X = np.empty((n, d), dtype=float)
    for i in range(n):
        g = np.exp(-0.5 * ((x - mu[i]) / width[i]) ** 2)
        wave = 0.10 * np.sin(2 * np.pi * (1.0 + 2.0 * mu[i]) * x)
        X[i] = amp[i] * (g + wave)
    X += 3e-4 * rng.normal(size=X.shape)
    return X


FAMILIES = {
    "collinear_clusters": make_collinear_clusters,
    "localized_svd": make_localized_svd,
    "repeated_directions": make_repeated_directions,
    "kernel_snapshots": make_kernel_snapshots,
}


# -----------------------------------------------------------------------------
# 4. Predeclared gap statistics (no post-hoc hand selection)
# -----------------------------------------------------------------------------

@dataclass
class GapResult:
    found: bool
    amp_ratio: float = np.nan
    min_amp: float = np.nan
    max_amp: float = np.nan
    min_error: float = np.nan
    max_error: float = np.nan
    rank: int | None = None
    n_points: int = 0


def largest_matched_error_gap(errors, amps, rel_width: float = 0.05, min_count: int = 8):
    """Largest amp ratio among points whose errors differ by <= rel_width relatively.

    This rule is declared before inspecting the result, so the highlighted gap is
    reproducible and not manually cherry-picked.
    """
    errors = np.asarray(errors, dtype=float)
    amps = np.asarray(amps, dtype=float)
    ok = np.isfinite(errors) & np.isfinite(amps) & (amps > 0)
    errors, amps = errors[ok], amps[ok]
    if len(errors) < min_count:
        return GapResult(False)

    order = np.argsort(errors)
    errors, amps = errors[order], amps[order]
    best = GapResult(False)
    j = 0
    for i in range(len(errors)):
        lo = errors[i]
        hi = max(lo * (1.0 + rel_width), lo + 100 * EPS)
        j = max(j, i)
        while j < len(errors) and errors[j] <= hi:
            j += 1
        if j - i < min_count:
            continue
        a = amps[i:j]
        ratio = float(np.max(a) / np.min(a))
        if (not best.found) or ratio > best.amp_ratio:
            best = GapResult(
                True,
                amp_ratio=ratio,
                min_amp=float(np.min(a)),
                max_amp=float(np.max(a)),
                min_error=float(errors[i]),
                max_error=float(errors[j - 1]),
                n_points=int(j - i),
            )
    return best


def matched_rank_error_gap(df: pd.DataFrame, rel_width: float = 0.10, min_count: int = 5):
    """Same-rank AND similar-error gap, controlling two obvious confounders."""
    best = GapResult(False)
    if df.empty:
        return best
    for rank, g in df.groupby("rank"):
        res = largest_matched_error_gap(g["error_rel"], g["amp2"], rel_width, min_count)
        if res.found and ((not best.found) or res.amp_ratio > best.amp_ratio):
            res.rank = int(rank)
            best = res
    return best


# -----------------------------------------------------------------------------
# 5. Experiments
# -----------------------------------------------------------------------------

def experiment_exact_counterexample(outdir: Path):
    X, good, bad = make_exact_counterexample()
    dg = compute_id_diagnostics(X, good)
    db = compute_id_diagnostics(X, bad)
    rows = [
        {"case": "good", "skeleton": good.tolist(), **dg},
        {"case": "bad", "skeleton": bad.tolist(), **db},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "A_exact_counterexample.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.scatter([max(dg["error_rel"], 1e-18)], [dg["amp2"]], s=90, label="well-separated skeleton")
    ax.scatter([max(db["error_rel"], 1e-18)], [db["amp2"]], s=110, marker="*", label="nearly-collinear skeleton")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Relative strict-ID error $\|X-WX_S\|_F/\|X\|_F$")
    ax.set_ylabel(r"Interpolation amplification $\|W\|_2$")
    ax.set_title("A. Exact-rank counterexample")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "A_exact_counterexample.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df


def experiment_noisy_pairs(outdir: Path, max_pairs: int = 9000):
    X = make_noisy_pair_matrix()
    n = X.shape[0]
    pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    rng = np.random.default_rng(20260911)
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), size=max_pairs, replace=False)]

    rows = []
    for i, j in pairs:
        d = compute_id_diagnostics(X, np.array([i, j]))
        if d["valid"]:
            rows.append({"i": int(i), "j": int(j), **d})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "B_noisy_fixed_rank_pairs.csv", index=False)

    gap = largest_matched_error_gap(df["error_rel"], df["amp2"], rel_width=0.05, min_count=20)
    with open(outdir / "B_noisy_gap_stat.json", "w", encoding="utf-8") as f:
        json.dump(asdict(gap), f, indent=2)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.scatter(df["error_rel"], df["amp2"], s=12, alpha=0.32)
    if gap.found:
        mask = (df["error_rel"] >= gap.min_error) & (df["error_rel"] <= gap.max_error)
        ax.scatter(df.loc[mask, "error_rel"], df.loc[mask, "amp2"], s=22, alpha=0.75, label="predeclared 5% error window")
        ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Relative strict-ID error $\|X-WX_S\|_F/\|X\|_F$")
    ax.set_ylabel(r"Interpolation amplification $\|W\|_2$")
    ax.set_title("B. Noisy fixed-rank skeletons: error vs amplification")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "B_noisy_fixed_rank_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return df, gap


def run_rbrp_family(
    family: str,
    outdir: Path,
    matrix_seed: int = 314159,
    trial_seeds=range(40),
    b: int = 8,
    k_max: int = 48,
    tol_loc: float | None = None,
):
    X = FAMILIES[family](seed=matrix_seed)
    rows = []

    for seed in trial_seeds:
        row_perm, _, residuals, _, ranks = blocked_rpid(
            X, k=min(k_max, min(X.shape)), b=b, tol=None,
            mode="rp", ortho=1, tol_loc=tol_loc, seed=int(seed),
        )
        for endpoint, rank in enumerate(ranks):
            S = row_perm[:rank].copy()
            d = compute_id_diagnostics(X, S)
            if d["valid"]:
                rows.append({
                    "family": family,
                    "matrix_seed": matrix_seed,
                    "trial_seed": int(seed),
                    "rank": int(rank),
                    "residual_proxy": float(residuals[endpoint]) if endpoint < len(residuals) else np.nan,
                    **d,
                })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"C_RBRP_{family}.csv", index=False)
    gap = matched_rank_error_gap(df, rel_width=0.10, min_count=5)
    return X, df, gap


def plot_rbrp_scatter(df: pd.DataFrame, family: str, gap: GapResult, outdir: Path):
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    ax.scatter(df["error_rel"], df["amp2"], s=18, alpha=0.42)
    if gap.found:
        mask = (
            (df["rank"] == gap.rank)
            & (df["error_rel"] >= gap.min_error)
            & (df["error_rel"] <= gap.max_error)
        )
        ax.scatter(
            df.loc[mask, "error_rel"], df.loc[mask, "amp2"],
            s=42, alpha=0.85,
            label=f"same rank k={gap.rank}, within 10% error",
        )
        ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Relative strict-ID error $\|X-WX_S\|_F/\|X\|_F$")
    ax.set_ylabel(r"Interpolation amplification $\|W\|_2$")
    ax.set_title(f"C. RBRP multi-seed gap: {family}")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"C_RBRP_{family}_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rbrp_by_rank(df: pd.DataFrame, family: str, outdir: Path):
    # Median and interquartile range at exact accepted ranks.
    g = df.groupby("rank")
    ranks = np.array(sorted(g.groups.keys()), dtype=int)
    med = np.array([g.get_group(k)["amp2"].median() for k in ranks])
    q25 = np.array([g.get_group(k)["amp2"].quantile(0.25) for k in ranks])
    q75 = np.array([g.get_group(k)["amp2"].quantile(0.75) for k in ranks])

    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    ax.plot(ranks, med, marker="o", label="median")
    ax.fill_between(ranks, q25, q75, alpha=0.25, label="25%-75%")
    ax.set_yscale("log")
    ax.set_xlabel("Accepted skeleton rank")
    ax.set_ylabel(r"Interpolation amplification $\|W\|_2$")
    ax.set_title(f"RBRP amplification across random seeds: {family}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"C_RBRP_{family}_by_rank.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_family(df: pd.DataFrame, gap: GapResult):
    return {
        "n_points": int(len(df)),
        "n_trials": int(df["trial_seed"].nunique()) if len(df) else 0,
        "rank_min": int(df["rank"].min()) if len(df) else None,
        "rank_max": int(df["rank"].max()) if len(df) else None,
        "error_min": float(df["error_rel"].min()) if len(df) else np.nan,
        "error_max": float(df["error_rel"].max()) if len(df) else np.nan,
        "amp_min": float(df["amp2"].min()) if len(df) else np.nan,
        "amp_max": float(df["amp2"].max()) if len(df) else np.nan,
        "matched_rank_error_gap": asdict(gap),
    }


# -----------------------------------------------------------------------------
# 6. Automatic Stage-1 decision
# -----------------------------------------------------------------------------

def stage1_decision(exact_df, noisy_gap: GapResult, family_summaries: dict):
    good = exact_df.loc[exact_df["case"] == "good"].iloc[0]
    bad = exact_df.loc[exact_df["case"] == "bad"].iloc[0]
    exact_ratio = float(bad["amp2"] / good["amp2"])
    exact_both_small = max(float(good["error_rel"]), float(bad["error_rel"])) < 1e-8

    # Conservative empirical criterion: at least two independent controlled RBRP
    # families exhibit >= 2x amplification spread at the same rank and within 10% error.
    rbrp_pass = []
    for family, s in family_summaries.items():
        g = s["matched_rank_error_gap"]
        if g["found"] and g["amp_ratio"] >= 2.0:
            rbrp_pass.append(family)

    pass_stage1 = (
        exact_both_small
        and exact_ratio >= 1e3
        and noisy_gap.found
        and noisy_gap.amp_ratio >= 2.0
        and len(rbrp_pass) >= 2
    )

    return {
        "stage1_pass": bool(pass_stage1),
        "exact_counterexample": {
            "both_errors_below_1e-8": bool(exact_both_small),
            "amplification_ratio_bad_over_good": exact_ratio,
        },
        "noisy_fixed_rank": asdict(noisy_gap),
        "rbrp_families_with_same_rank_similar_error_gap_ge_2x": rbrp_pass,
        "criterion": (
            "PASS iff: exact counterexample has both errors <1e-8 and amp ratio >=1e3; "
            "noisy fixed-rank 5%-error window has amp ratio >=2; and at least two RBRP "
            "families have amp ratio >=2 at the same rank within a 10%-error window."
        ),
        "interpretation": (
            "PASS supports only the Stage-1 claim that approximation error does not by itself "
            "control interpolation amplification, including for unmodified RBRP outputs. It does "
            "not prove a theorem about all matrices, nor that the proposed Stage-2 refinement will help."
        ),
    }


def write_summary_text(outdir: Path, decision: dict, family_summaries: dict):
    lines = []
    lines.append("STAGE 1 FINAL VALIDATION SUMMARY")
    lines.append("=" * 72)
    lines.append(f"Decision: {'PASS' if decision['stage1_pass'] else 'NOT YET PASS'}")
    lines.append("")
    lines.append(decision["criterion"])
    lines.append("")
    lines.append("Exact counterexample:")
    lines.append(json.dumps(decision["exact_counterexample"], indent=2))
    lines.append("")
    lines.append("Noisy fixed-rank matched-error gap:")
    lines.append(json.dumps(decision["noisy_fixed_rank"], indent=2))
    lines.append("")
    lines.append("RBRP families:")
    for fam, s in family_summaries.items():
        g = s["matched_rank_error_gap"]
        lines.append(
            f"- {fam}: points={s['n_points']}, trials={s['n_trials']}, "
            f"error=[{s['error_min']:.3e},{s['error_max']:.3e}], "
            f"amp=[{s['amp_min']:.3e},{s['amp_max']:.3e}], "
            + (f"same-rank/similar-error gap={g['amp_ratio']:.3g}x at k={g['rank']}" if g["found"] else "no qualifying window")
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append(decision["interpretation"])
    (outdir / "STAGE1_FINAL_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="stage1_final_results")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--kmax", type=int, default=48)
    parser.add_argument("--block", type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = {
        "trials": args.trials,
        "trial_seeds": list(range(args.trials)),
        "matrix_seed": 314159,
        "noisy_pair_seed": 123,
        "noisy_pair_subsample_seed": 20260911,
        "block_size": args.block,
        "k_max": args.kmax,
        "tol_loc": f"1/b = {1/args.block:g}",
        "strict_id_rank_tol": 1e-12,
        "noisy_matched_error_window": "5% relative width; min 20 points",
        "RBRP_matched_rank_error_window": "same exact accepted rank + 10% relative error width; min 5 points",
    }
    with open(outdir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("[A] Exact counterexample")
    exact_df = experiment_exact_counterexample(outdir)
    print(exact_df[["case", "error_rel", "amp2", "cond_Xs"]].to_string(index=False))

    print("\n[B] Noisy fixed-rank study")
    noisy_df, noisy_gap = experiment_noisy_pairs(outdir)
    print(f"valid skeleton pairs: {len(noisy_df)}")
    print(f"predeclared 5%-error-window amp gap: {noisy_gap.amp_ratio:.3g}x" if noisy_gap.found else "no qualifying window")

    print("\n[C] RBRP multi-seed studies")
    family_summaries = {}
    all_frames = []
    for family in FAMILIES:
        _, df, gap = run_rbrp_family(
            family, outdir,
            matrix_seed=314159,
            trial_seeds=range(args.trials),
            b=args.block,
            k_max=args.kmax,
            tol_loc=None,
        )
        plot_rbrp_scatter(df, family, gap, outdir)
        plot_rbrp_by_rank(df, family, outdir)
        summary = summarize_family(df, gap)
        family_summaries[family] = summary
        all_frames.append(df)
        if gap.found:
            print(f"{family:22s}: same-rank/similar-error gap = {gap.amp_ratio:8.3g}x at k={gap.rank}")
        else:
            print(f"{family:22s}: no qualifying same-rank/similar-error window")

    pd.concat(all_frames, ignore_index=True).to_csv(outdir / "C_RBRP_all_families.csv", index=False)
    with open(outdir / "C_RBRP_family_summaries.json", "w", encoding="utf-8") as f:
        json.dump(family_summaries, f, indent=2)

    decision = stage1_decision(exact_df, noisy_gap, family_summaries)
    with open(outdir / "STAGE1_DECISION.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    write_summary_text(outdir, decision, family_summaries)

    print("\n[D] Final Stage-1 decision")
    print("PASS" if decision["stage1_pass"] else "NOT YET PASS")
    print(decision["interpretation"])
    print(f"\nAll results written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
