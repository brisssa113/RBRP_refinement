"""Assemble paper/figs/ from results/.

Eleven of the thirteen figures are plain copies of PNGs written by the experiment scripts; `tsweep.png` is re-drawn here
(same data as results/stage2_T_sweep_results/T_sweep_summary.csv, paper styling: no title, 'proposed' legend label).
`scatter_real.png` is the scatter written by stage3_realdata.py (results/stage3_realdata_results/final_scatter.png).
Usage: python code/make_figures.py
"""
import shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent; RES = ROOT / 'results'; FIG = ROOT / 'paper' / 'figs'; FIG.mkdir(parents=True, exist_ok=True)
COPY = {
    's1_collinear_clusters.png': 'stage1_b4_results/C_RBRP_collinear_clusters_gap.png',
    's1_localized_svd.png': 'stage1_b4_results/C_RBRP_localized_svd_gap.png',
    's1_repeated_directions.png': 'stage1_b4_results/C_RBRP_repeated_directions_gap.png',
    's1_kernel_snapshots.png': 'stage1_b4_results/C_RBRP_kernel_snapshots_gap.png',
    's1_noisy.png': 'stage1_b4_results/B_noisy_fixed_rank_gap.png',
    'rank.png': 'stage3_rank_sweep_results/rank_sweep_rho_A_vs_k.png',
    'scatter.png': 'stage3_final_results/final_scatter.png',
    'guardA.png': 'stage3_ablation_results/ablation_guard_A.png',
    'runtime.png': 'stage3_runtime_results/runtime_scaling.png',
    'noise_random.png': 'stage3_noise_results/noise_random.png',
    'noise_worst.png': 'stage3_noise_results/noise_worst.png',
    'scatter_real.png': 'stage3_realdata_results/final_scatter.png',
}
for dst, src in COPY.items():
    if (RES / src).exists(): shutil.copy(RES / src, FIG / dst); print('copied', dst)
    else: print('MISSING', src, '-> run the corresponding experiment first')

s = pd.read_csv(RES / 'stage2_T_sweep_results' / 'T_sweep_summary.csv')
fig, ax = plt.subplots(figsize=(6.4, 4.1))
for o, mk, lab in (('norm', 'o-', 'norm order (proposed)'), ('random', 's--', 'random order')):
    q = s[(s.order == o) & np.isfinite(s['T'])]
    ax.errorbar(q.mean_n_eval, q.mean_rho_A, yerr=q.se, fmt=mk, capsize=2, label=lab)
    for _, r in q.iterrows():
        ax.annotate(f'T={int(r["T"])}', (r.mean_n_eval, r.mean_rho_A), fontsize=7, xytext=(3, 3), textcoords='offset points')
f = s[np.isinf(s['T'])]
ax.errorbar(f.mean_n_eval, f.mean_rho_A, yerr=f.se, fmt='k*', ms=11, capsize=2, label='full scan')
ax.set_xscale('log'); ax.set_xlabel('mean exact evaluations'); ax.set_ylabel('mean rho_A'); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(FIG / 'tsweep.png', dpi=150); print('drew tsweep.png')
