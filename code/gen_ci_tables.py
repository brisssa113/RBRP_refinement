# Builds the tables that carry bootstrap intervals (run bootstrap_ci.py first; point --ci to its output folder)
import sys, pandas as pd, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; OUT=ROOT/'paper'/'tables'; OUT.mkdir(parents=True,exist_ok=True)
ci = sys.argv[1] if len(sys.argv) > 1 else str(ROOT/'results'/'bootstrap_synthetic_and_baselines')
P = pd.read_csv(f'{ci}/ci_pooled.csv'); N = pd.read_csv(f'{ci}/ci_noise.csv'); PN = pd.read_csv(f'{ci}/ci_noise_pair.csv'); NS = pd.read_csv(f'{ci}/nu_star.csv')
NM = {'nearly_collinear_clusters':'Collinear clusters','repeated_directions':'Repeated directions','localized_svd':'Localized SVD','kernel_snapshots':'Kernel snapshots','burgers_snapshots':'Burgers','digits':'Digits'}
def tab(name, hdr, rows, cols, cap, label):
    open(OUT/f'{name}.tex','w').write('\\begin{table}[t]\n\\centering\\small\n\\caption{'+cap+'}\\label{'+label+'}\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+hdr+' \\\\\n\\midrule\n'+' \\\\\n'.join(rows)+' \\\\\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n')
f = lambda e,l,h,d=3: f'{e:.{d}f} [{l:.{d}f}, {h:.{d}f}]'
pc = lambda e,l,h: f'{100*e:.1f} [{100*l:.1f}, {100*h:.1f}]'
rows=[' & '.join([r.x.replace('_','\\_'), r.y.replace('_','\\_'), str(int(r.pairs)), f(r.A_ratio,r.A_lo,r.A_hi), f(r.E_ratio,r.E_lo,r.E_hi), pc(r.x_dom,r.xd_lo,r.xd_hi), pc(r.y_dom,r.yd_lo,r.yd_hi)]) for r in P.itertuples()]
tab('tab_pooled','x & y & pairs & median $A$ ratio & median $E$ ratio & x dominates (\\%) & y dominates (\\%)',rows,'llrrrrr',
    'Paired comparison pooled over all configurations and seeds, with 95\\% percentile bootstrap intervals (seeds resampled within configuration). ``x dominates y\'\' means no worse in both $A$ and $E$ and strictly better in one.','tab:pooled')
d2=lambda r: f'{r.ratio_to_RBRP:.2f} [{r.lo:.2f},{r.hi:.2f}]'
rows=[]
LV=(0.1,1.0)
for m,lab in [('Vol(f=1)','Vol(1)'),('VolGuard-500','VolGuard'),('Ours-T10','Ours-$T$10'),('Hybrid-T10','Hybrid')]:
    s_=N[N.method==m]; g=lambda kind,lv=np.nan: d2(s_[(s_.kind==kind)&((s_.level==lv)|(np.isnan(lv)&s_.level.isna()))].iloc[0])
    rows.append(' & '.join([lab,g('rho_A'),g('rho_F')]+[g('rand',v) for v in LV]+[g('worst',v) for v in LV]))
q=lambda kind,lv: PN[(PN.kind==kind)&(PN.level==lv)].iloc[0]
rows.append(' & '.join(['Hybrid/VolGuard (paired)','--','--']+[f'{q("rand",v).hybrid_over_volguard:.3f} [{q("rand",v).lo:.3f},{q("rand",v).hi:.3f}]' for v in LV]+[f'{q("worst",v).hybrid_over_volguard:.3f} [{q("worst",v).lo:.3f},{q("worst",v).hi:.3f}]' for v in LV]))
tab('tab_noise',' & $\\rho_A$ & $\\rho_F$ & \\multicolumn{2}{c}{random noise $\\nu$} & \\multicolumn{2}{c}{worst case $\\eta$}\\\\\n\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nMethod & & & 0.1 & 1 & 0.1 & 1',rows,'lrrrrrr',
    'Reconstruction error under perturbations of the skeleton, relative to the RBRP skeleton: median over all configurations and 20 seeds of $\\mathrm{err}(\\text{method})/\\mathrm{err}(\\text{RBRP})$, with 95\\% bootstrap intervals. Random noise: relative level $\\nu$; worst case: relative size $\\eta$ (lower bound of Proposition~\\ref{prop:pert}). $\\rho_A,\\rho_F$: median ratios of $\\|W\\|_2$, $\\|W\\|_F$. Last row: paired ratio of the pipeline to the guarded volume exchange. For $0.01$ the differences are dominated by those in $E$.','tab:noise')
rows=[' & '.join([NM[r.dataset],str(r.k),f'{r.median:.3g}',f'[{r.q25:.3g}, {r.q75:.3g}]']) for r in NS.itertuples()]
tab('tab_nustar','Matrix & $k$ & median $\\nu^*$ & quartiles',rows,'lrrr','Crossover noise level $\\nu^*=E_0\\sqrt n/\\|W_0\\|_F$ of the RBRP skeletons (median and quartiles over 20 seeds): the random-noise term of the reconstruction error exceeds the approximation error only for $\\nu>\\nu^*$.','tab:nustar')
print('ci tables ok')
