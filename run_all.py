#!/usr/bin/env python
"""One entry point for the reproducibility package.

  python run_all.py               # DEFAULT: tables + figures from the stored results/ (about 2 minutes, no experiments re-run)
  python run_all.py tables      # seconds: paper/tables/*.tex from the stored CSVs in results/ (+ bootstrap CIs, ~1 min)
  python run_all.py figures     # paper/figs/ from results/
  python run_all.py smoke       # every experiment script on a tiny problem, written to a temp dir (sanity check, ~10 min)
  python run_all.py experiments # RE-RUN all experiments from scratch into results/ (many hours, single thread recommended)
  python run_all.py realdata    # only the additional real matrices (image patches, ECG Hankel, digits kernel)
  python run_all.py paper       # latexmk paper/main.tex (needs a LaTeX installation)
Set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 for the timing experiments (runtime table).
"""
import os, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent; C = ROOT / 'code'; R = ROOT / 'results'
PY = sys.executable

def run(*args, cwd=ROOT):
    print('>>', ' '.join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], check=True, cwd=cwd)

def _stage_bootstrap_inputs(dst: Path, *srcs):
    dst.mkdir(parents=True, exist_ok=True)
    for s in srcs:
        for f in Path(s).glob('*_raw_*.csv'):
            (dst / f.name).write_bytes(f.read_bytes())
    return dst

def tables():
    run(C / 'gen_tables.py')
    with tempfile.TemporaryDirectory() as t:
        d = _stage_bootstrap_inputs(Path(t), R / 'stage3_final_results', R / 'stage3_noise_results')
        run(C / 'bootstrap_ci.py', '--indir', d, '--out', R / 'bootstrap_synthetic_and_baselines')
    run(C / 'gen_ci_tables.py')
    if (R / 'stage3_realdata_results' / 'final_summary.csv').exists():
        with tempfile.TemporaryDirectory() as t:
            d = _stage_bootstrap_inputs(Path(t), R / 'stage3_realdata_results')
            run(C / 'bootstrap_ci.py', '--indir', d, '--out', R / 'bootstrap_real')
        run(C / 'gen_real_tables.py')
    else:
        print('(real-data tables/figure kept as in paper/: copy your stage3_realdata_results/ into results/ or run `python run_all.py realdata` to regenerate them)')

def figures():
    run(C / 'make_figures.py')

def smoke():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for i, (s, extra) in enumerate([('stage2_sorted_scan', []), ('stage2_t_sweep', []), ('stage3_rank_sweep', []), ('stage3_runtime_scaling', []),
                                        ('stage3_realistic_baseline', []), ('stage3_volume_baseline', []), ('stage3_extra_ablations', []),
                                        ('stage3_final_experiments', []), ('stage3_noise_downstream', []),
                                        ('stage3_realdata', ['--only', 'digits_kernel'])]):
            run(C / f'{s}.py', '--smoke', '--out', t / f'o{i}', *extra)

def realdata():
    run(C / 'stage3_realdata.py')                       # -> results/stage3_realdata_results
    tables()

def experiments():
    run(C / 'stage1.py', '--block', 4, '--outdir', R / 'stage1_b4_results')
    run(C / 'stage1.py', '--block', 8, '--outdir', R / 'stage1_b8_results')    # only quoted in the text of Section 3
    for s in ['stage2_sorted_scan', 'stage2_t_sweep', 'stage3_rank_sweep', 'stage3_runtime_scaling', 'stage3_realistic_baseline',
              'stage3_extra_ablations', 'stage3_final_experiments', 'stage3_noise_downstream']:
        run(C / f'{s}.py')                              # defaults = paper settings (seeds, output folders, reference CSVs)
    realdata()
    figures()

def paper():
    subprocess.run(['latexmk', '-pdf', 'main.tex'], check=True, cwd=ROOT / 'paper')

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if cmd == 'all':
        tables(); figures()
    elif cmd in {'tables', 'figures', 'smoke', 'experiments', 'realdata', 'paper'}:
        globals()[cmd]()
    else:
        print(__doc__); sys.exit(1)
