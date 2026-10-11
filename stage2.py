from __future__ import annotations
import concurrent.futures, csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

DATASETS=['nearly_collinear_clusters','repeated_directions','localized_svd','kernel_snapshots']
GAMMAS=[0.5,1.0,1.5,2.0,3.0]
SEEDS=list(range(40))
TIMEOUT=8.0
WORKERS=1
OUT=Path('D:\\fyp\\stage2_global_final'); OUT.mkdir(exist_ok=True)

def one(task):
    ds,g,s=task
    cmd=[sys.executable,'-u','D:\\fyp\\stage2_worker.py','--dataset',ds,'--gamma',str(g),'--seed',str(s)]
    env=os.environ.copy(); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    t=time.perf_counter()
    try:
        cp=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=TIMEOUT,env=env)
        dt=time.perf_counter()-t
        if cp.returncode==0:
            row=json.loads(cp.stdout.strip().splitlines()[-1]); row['driver_wall_sec']=dt; return row
        return error_row(ds,g,s,'error',cp.stderr[-1000:],dt)
    except subprocess.TimeoutExpired as e:
        return error_row(ds,g,s,'timeout',f'exceeded {TIMEOUT}s',time.perf_counter()-t)
    except Exception as e:
        return error_row(ds,g,s,'error',repr(e),time.perf_counter()-t)

def error_row(ds,g,s,status,msg,dt):
    return {'dataset':ds,'seed':s,'data_seed':s,'refinement_seed':-1,'k':np.nan,'b':4,'gamma':g,
            'candidate_count_c':int(np.ceil(g*4)),'initial_E':np.nan,'final_E':np.nan,'rho_E':np.nan,
            'initial_A':np.nan,'final_A':np.nan,'rho_A':np.nan,'initial_max_row_l1':np.nan,'final_max_row_l1':np.nan,
            'initial_cond_Xs':np.nan,'final_cond_Xs':np.nan,'initial_identity_error':np.nan,'final_identity_error':np.nan,
            'accepted_exchanges':np.nan,'iterations':np.nan,'guard_ok':False,'amp_improved':False,'success':False,
            'runtime_sec':dt,'rbrp_ranks':'','S0':'','Sf':'','status':status,'error_message':msg,'wall_time_sec':dt,'driver_wall_sec':dt}

def main():
    tasks=[(d,g,s) for d in DATASETS for g in GAMMAS for s in SEEDS]
    rows=[]; t=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i,row in enumerate(ex.map(one,tasks),1):
            rows.append(row)
            if i%20==0 or i==len(tasks): print(f'progress {i}/{len(tasks)} elapsed={time.perf_counter()-t:.1f}s',flush=True)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with open(OUT/'stage2_step2_9_raw_results.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    # Summaries
    summ=[]
    for d in DATASETS:
      for g in GAMMAS:
        sub=[r for r in rows if r['dataset']==d and r['gamma']==g]; v=[r for r in sub if r['status']=='ok' and np.isfinite(r['rho_A'])]
        def mean(x): return float(np.mean(x)) if x else np.nan
        def med(x): return float(np.median(x)) if x else np.nan
        summ.append({'dataset':d,'gamma':g,'c':int(np.ceil(g*4)),'n_total':len(sub),'n_valid':len(v),'n_timeout':sum(r['status']=='timeout' for r in sub),'n_error':sum(r['status']=='error' for r in sub),
                     'mean_rho_A':mean([r['rho_A'] for r in v]),'median_rho_A':med([r['rho_A'] for r in v]),'mean_rho_E':mean([r['rho_E'] for r in v]),
                     'success_rate':mean([r['success'] for r in v]),'amp_improved_rate':mean([r['amp_improved'] for r in v]),'guard_ok_rate':mean([r['guard_ok'] for r in v]),
                     'mean_accepted_exchanges':mean([r['accepted_exchanges'] for r in v]),'mean_runtime_sec':mean([r['runtime_sec'] for r in v])})
    with open(OUT/'stage2_step2_9_summary_by_dataset_gamma.csv','w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    v=[r for r in rows if r['status']=='ok' and np.isfinite(r['rho_A'])]
    agg=[]
    for g in GAMMAS:
      sub=[r for r in v if r['gamma']==g]
      agg.append({'gamma':g,'c':int(np.ceil(g*4)),'n_valid':len(sub),'n_total':len([r for r in rows if r['gamma']==g]),'n_timeout':sum(r['status']=='timeout' for r in rows if r['gamma']==g),'n_error':sum(r['status']=='error' for r in rows if r['gamma']==g),
                  'mean_rho_A':float(np.mean([r['rho_A'] for r in sub])) if sub else np.nan,'median_rho_A':float(np.median([r['rho_A'] for r in sub])) if sub else np.nan,
                  'mean_rho_E':float(np.mean([r['rho_E'] for r in sub])) if sub else np.nan,'success_rate':float(np.mean([r['success'] for r in sub])) if sub else np.nan,
                  'amp_improved_rate':float(np.mean([r['amp_improved'] for r in sub])) if sub else np.nan,'guard_ok_rate':float(np.mean([r['guard_ok'] for r in sub])) if sub else np.nan})
    with open(OUT/'stage2_step2_9_aggregate_by_gamma.csv','w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    print('FINAL',len(rows),'valid',len(v),'timeouts',sum(r['status']=='timeout' for r in rows),'errors',sum(r['status']=='error' for r in rows),'wall',time.perf_counter()-t)

if __name__=='__main__': main()
