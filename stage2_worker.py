from __future__ import annotations
import argparse, json, math, sys, time
import importlib.util
from pathlib import Path
import numpy as np

STAGE1 = Path('D:\\fyp\\stage1.py')
PILOT = Path('D:\\fyp\\stage2_global_pilot.py')
spec = importlib.util.spec_from_file_location('stage1', STAGE1)
mod1 = importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod1; spec.loader.exec_module(mod1)
spec2 = importlib.util.spec_from_file_location('stage2_global_pilot', PILOT)
p = importlib.util.module_from_spec(spec2); sys.modules[spec2.name]=p; spec2.loader.exec_module(p)

builders = p.DATASETS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--gamma', type=float, required=True)
    ap.add_argument('--seed', type=int, required=True)
    args=ap.parse_args()
    ds=args.dataset; gamma=args.gamma; seed=args.seed
    k=builders[ds]['k']
    X=builders[ds]['builder'](seed)
    refinement_seed = 1_000_000 + list(p.DATASETS).index(ds)*100_000 + list(p.GAMMAS).index(gamma)*1_000 + seed
    t=time.perf_counter()
    row,hist=p.run_one(ds,X,k,gamma,refinement_seed)
    row['data_seed']=seed; row['refinement_seed']=refinement_seed; row['status']='ok'; row['error_message']=''
    row['wall_time_sec']=time.perf_counter()-t
    print(json.dumps(row,allow_nan=True))

if __name__=='__main__': main()
