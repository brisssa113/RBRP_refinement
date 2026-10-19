from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent.parent; RES=ROOT/'results'; OUT=ROOT/'paper'/'tables'; OUT.mkdir(parents=True,exist_ok=True)
def rd(p,**k):
    hits=[h for h in RES.glob('*/'+p) if h.parent.name!='stage3_realdata_results']; assert len(hits)==1, (p,hits)   # exact file name, unique outside the real-data folder
    return pd.read_csv(hits[0],**k)
NM={'nearly_collinear_clusters':'Collinear clusters','repeated_directions':'Repeated directions','localized_svd':'Localized SVD','kernel_snapshots':'Kernel snapshots','burgers_snapshots':'Burgers','digits':'Digits'}
def w(name,header,rows,cols,caption,label,small=True):
    t=('\\begin{table}[t]\n\\centering'+('\\small' if small else '')+'\n\\caption{'+caption+'}\\label{'+label+'}\n'
       '\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+header+' \\\\\n\\midrule\n'
       +' \\\\\n'.join(rows)+' \\\\\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n')
    open(OUT/f'{name}.tex','w').write(t)
def f(x,d=3): return '--' if pd.isna(x) else f'{x:.{d}f}'
K={'nearly_collinear_clusters':4,'repeated_directions':4,'localized_svd':14,'kernel_snapshots':16}
# 1 pilot
d=rd('raw_results.csv'); d=d[d.method.isin(['sorted_full','sorted_T10'])]; rows=[]
for ds in list(K)+['ALL']:
    s=d if ds=='ALL' else d[d.dataset==ds]; a,b=s[s.method=='sorted_full'],s[s.method=='sorted_T10']
    rows.append(' & '.join([('All (160 runs)' if ds=='ALL' else f'{NM[ds]} ($k={K[ds]}$)'),f(a.rho_A.mean()),f(b.rho_A.mean()),f(a.rho_E.mean()),f'{a.n_eval.mean():,.0f}',f'{b.n_eval.mean():,.0f}',f'{a.time_sec.mean():.1f}',f'{b.time_sec.mean():.2f}']))
w('tab_pilot','Dataset & full $\\rho_A$ & $T{=}10$ $\\rho_A$ & full $\\rho_E$ & evals (full) & evals ($T{=}10$) & time (s) full & time (s) $T{=}10$',rows,'lrrrrrrr','Main experiment on the four synthetic datasets ($n=500$, $d=60$, 40 RBRP seeds each). $\\rho_A=A_f/A_0$ and $\\rho_E=E_f/E_0$ are averaged over seeds.','tab:pilot')
# 2 T sweep
t=rd('T_sweep_summary.csv'); nm,rn=t[t.order=='norm'],t[t.order=='random']; rows=[]
for _,r in nm.iterrows():
    q=rn[rn['T']==r['T']]; Tl='$\\infty$' if np.isinf(r['T']) else f"{int(r['T'])}"
    rows.append(' & '.join([Tl,f"{r.mean_rho_A:.3f} ({r.se:.3f})",f'{r.mean_n_eval:,.0f}',(f"{q.mean_rho_A.iloc[0]:.3f} ({q.se.iloc[0]:.3f})" if len(q) else '--'),(f'{q.mean_n_eval.iloc[0]:,.0f}' if len(q) else '--')]))
w('tab_tsweep','$T$ & row-norm order $\\rho_A$ (SE) & evals & random order $\\rho_A$ (SE) & evals',rows,'lrrrr','Scan budget $T$ and ordering ablation (160 runs, same $S_0$ for both orders).','tab:tsweep')
# 3 rank
r=rd('rank_sweep_summary.csv'); rows=[]
for (ds,k),g in r.groupby(['dataset','k'],sort=False):
    a,b=g[g.variant=='full'].iloc[0],g[g.variant=='T10'].iloc[0]
    rows.append(' & '.join([NM[ds],str(k),f'{a.median_A0:.1f}',f'{a.mean_rho_A:.3f} ({a.se:.3f})',f'{b.mean_rho_A:.3f} ({b.se:.3f})']))
w('tab_rank','Dataset & $k$ & median $A_0$ & full $\\rho_A$ (SE) & $T{=}10$ $\\rho_A$ (SE)',rows,'lrrrr','Rank sensitivity (20 seeds per configuration).','tab:rank')
# 4 compare
s=rd('final_summary.csv'); order=list(dict.fromkeys(zip(s.dataset,s.k))); rows=[]
def g(ds,k,m,c):
    q=s[(s.dataset==ds)&(s.k==k)&(s.method==m)]; return q[c].iloc[0] if len(q) else np.nan
for ds,k in order:
    rows.append(' & '.join([NM[ds],str(k)]+[f(g(ds,k,m,'median_A'),1) for m in ['RBRP','Vol(f=1)','VolGuard-500','Ours-T10','Hybrid-T10','Ours-full']]+[('--' if pd.isna(g(ds,k,'Vol(f=1)','guard_ok')) else f"{100*g(ds,k,'Vol(f=1)','guard_ok'):.0f}\\%")]))
w('tab_compare','Matrix & $k$ & RBRP & Vol(1) & VolGuard & Ours-$T$10 & Hybrid & Ours-full & Vol(1) feasible',rows,'lrrrrrrrr','Median amplification $A$ after each method (20 RBRP seeds; the full scan uses 10 seeds on Burgers and digits $k{=}10$ and none on digits $k{=}20,30$). The last column is the share of Vol(1) runs satisfying $E\\le1.05E_0$.','tab:compare')
# 6 guard
gg=rd('ablation_b_guard.csv',dtype={'guard':str}); rows=[]
for (ds,k),q in gg.groupby(['dataset','k'],sort=False):
    q=q.set_index('guard'); rows.append(' & '.join([NM[ds],str(k)]+[f"{q.loc[x,'median_A']:.2f} / {q.loc[x,'mean_rho_E']:.3f}" for x in ['1.00','1.05','1.20','inf']]))
w('tab_guard','Matrix & $k$ & $g{=}1.00$ & $g{=}1.05$ & $g{=}1.20$ & $g{=}\\infty$',rows,'lrrrrr','Error-budget ablation (RBRP skeletons, $T{=}10$, 10 seeds). Each cell: median $A$ / mean $\\rho_E$.','tab:guard')
# 7 runtime
q=rd('runtime_summary.csv'); rows=[]
for (ds,n),gq in q.groupby(['dataset','n'],sort=False):
    gq=gq.set_index('variant'); rb=gq.loc['RBRP','time_sec']; t10=gq.loc['T10','time_sec']; fu=gq.loc['full','time_sec'] if 'full' in gq.index else np.nan
    rows.append(' & '.join([NM[ds],str(n),f'{1000*rb:.1f}',f'{t10:.2f}',f'{t10/rb:.0f}',('--' if pd.isna(fu) else f'{fu:.1f}')]))
w('tab_runtime','Dataset & $n$ & RBRP (ms) & $T{=}10$ (s) & $T{=}10$/RBRP & full (s)',rows,'lrrrrr','Wall-clock time (single thread, $d=60$, representative $k$).','tab:runtime')
# 8 qrcp
a=rd('ablation_a_qrcp_refine.csv'); rows=[]
for _,x in a.iterrows():
    rows.append(' & '.join([NM[x.dataset],str(x.k),f'{x.A_QRCP:.2f}',f"{x['rho_A_QRCP+T10']:.3f}",f"{x['rho_E_QRCP+T10']:.3f}",f(x['rho_A_QRCP+full']),f(x['rho_E_QRCP+full'])]))
w('tab_qrcp','Matrix & $k$ & $A$ (QRCP) & $T{=}10$: $A/A_{\\rm QRCP}$ & $E/E_{\\rm QRCP}$ & full: $A/A_{\\rm QRCP}$ & $E/E_{\\rm QRCP}$',rows,'lrrrrrr','Refining the QRCP skeleton (one deterministic run per row).','tab:qrcp')
print('tables ok')
