"""Within-rank dispersion of RBRP skeletons (supports Sec. 3.3 'typical spread').
For every family and every rank k with >= 10 recorded RBRP skeletons: P90/P10 of A and of E, and corr(log E, log A).
Usage: python code/stage1_within_rank_stats.py   -> results/stage1_b4_results/within_rank_stats.csv (+ summary printed)
"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / 'results' / 'stage1_b4_results'
d = pd.read_csv(D / 'C_RBRP_all_families.csv'); rows = []
for (f, k), h in d.groupby(['family', 'rank']):
    if len(h) < 10: continue
    a = h['amp2'].quantile([.1, .9]).values; e = h['error_rel'].quantile([.1, .9]).values
    c = np.corrcoef(np.log(h['error_rel']), np.log(h['amp2']))[0, 1] if h['error_rel'].std() > 0 else np.nan
    rows.append(dict(family=f, rank=k, n=len(h), A_p90_p10=a[1]/a[0], E_p90_p10=e[1]/e[0], corr_logE_logA=c, A_max_min=h['amp2'].max()/h['amp2'].min()))
R = pd.DataFrame(rows); R.to_csv(D / 'within_rank_stats.csv', index=False)
S = R.groupby('family').agg(ranks=('rank', 'size'), A_p90_p10_median=('A_p90_p10', 'median'), A_p90_p10_max=('A_p90_p10', 'max'),
                            E_p90_p10_median=('E_p90_p10', 'median'), corr_median=('corr_logE_logA', 'median'),
                            share_ratio_gt2=('A_p90_p10', lambda x: (x > 2).mean()))
print(S.round(2).to_string()); print('all ranks pooled: median A P90/P10 = %.2f, share >2 = %.2f' % (R.A_p90_p10.median(), (R.A_p90_p10 > 2).mean()))
