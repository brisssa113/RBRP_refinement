# Tables for the additional real matrices (image patches, ECG Hankel, digits kernel). Usage: python gen_real_tables.py <folder with final_summary.csv and final_raw_*.csv of the real-data run> <bootstrap output folder of that run>
import sys, glob, pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'paper'/'tables'; OUT.mkdir(parents=True,exist_ok=True)
src = sys.argv[1] if len(sys.argv)>1 else str(ROOT/'results'/'stage3_realdata_results')
ci  = sys.argv[2] if len(sys.argv)>2 else str(ROOT/'results'/'bootstrap_real')
NM={'image_patches':'Image patches','ecg_hankel':'ECG Hankel','digits_kernel':'Digits kernel'}
S=pd.read_csv(glob.glob(f'{src}/*final_summary.csv')[0]); R=pd.concat([pd.read_csv(x) for x in glob.glob(f'{src}/*final_raw_*.csv')])
P=pd.read_csv(f'{ci}/ci_pooled.csv')
g=lambda ds,k,m,c: S[(S.dataset==ds)&(S.k==k)&(S.method==m)][c].iloc[0]
order=list(dict.fromkeys(zip(S.dataset,S.k)))
def tab(name,hdr,rows,cols,cap,label):
    open(OUT/f'{name}.tex','w').write('\\begin{table}[tbp]\n\\centering\\small\n\\caption{'+cap+'}\\label{'+label+'}\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+hdr+' \\\\\n\\midrule\n'+' \\\\\n'.join(rows)+' \\\\\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n')
rows=[' & '.join([NM[d],str(k)]+[f"{g(d,k,m,'median_A'):.1f}" for m in ['RBRP','QRCP','Vol(f=1)','VolGuard-500','Ours-T10','Hybrid-T10']]+[f"{100*g(d,k,'Vol(f=1)','guard_ok'):.0f}\\%"]) for d,k in order]
tab('tab_real','Matrix & $k$ & RBRP & QRCP & Vol(1) & VolGuard & Ours-$T$10 & Hybrid & Vol(1) feasible',rows,'lrrrrrrrr','Additional real matrices (10 RBRP seeds): median amplification $A$ after each method and the share of Vol(1) runs satisfying $E\\le1.05E_0$.','tab:real')
q=R[R.method=='QRCP'].set_index(['dataset','k'])[['A','E']].rename(columns={'A':'Aq','E':'Eq'})
h=R[R.method=='Hybrid-T10'].join(q,on=['dataset','k']); h['a']=h.A/h.Aq; h['e']=h.E/h.Eq
med=h.groupby(['dataset','k'])[['a','e']].median()
rows=[' & '.join([NM[d],str(k),f"{med.loc[(d,k),'a']:.2f}",f"{med.loc[(d,k),'e']:.2f}",f"{g(d,k,'Vol(f=1)','mean_time_sec'):.2f}",f"{g(d,k,'Ours-T10','mean_time_sec'):.1f}",f"{g(d,k,'Hybrid-T10','mean_time_sec'):.1f}"]) for d,k in order]
tab('tab_real_qrcp','Matrix & $k$ & Hybrid $A/A_{\\rm QRCP}$ & Hybrid $E/E_{\\rm QRCP}$ & Vol(1) time (s) & Ours-$T$10 time (s) & Hybrid time (s)',rows,'lrrrrrr','Pipeline versus QRCP (median over seeds of the ratios; QRCP is a single deterministic skeleton) and mean wall-clock times.','tab:realqrcp')
f=lambda e,l,h: f'{e:.3f} [{l:.3f}, {h:.3f}]'; pc=lambda e,l,h: f'{100*e:.1f} [{100*l:.1f}, {100*h:.1f}]'
rows=[' & '.join([r.x.replace('_','\\_'),r.y.replace('_','\\_'),str(int(r.pairs)),f(r.A_ratio,r.A_lo,r.A_hi),f(r.E_ratio,r.E_lo,r.E_hi),pc(r.x_dom,r.xd_lo,r.xd_hi),pc(r.y_dom,r.yd_lo,r.yd_hi)]) for r in P.itertuples()]
tab('tab_real_pooled','x & y & pairs & median $A$ ratio & median $E$ ratio & x dominates (\\%) & y dominates (\\%)',rows,'llrrrrr','Additional real matrices: paired comparison pooled over the nine configurations and ten seeds, with 95\\% bootstrap intervals (as in Table~\\ref{tab:pooled}).','tab:realpooled')
print('real tables ok')
