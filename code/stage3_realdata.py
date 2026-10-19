"""Stage 3.8: public real matrices.

Matrices (rows = points to interpolate, as in all previous experiments):
  image_patches : 3000 random 12x12 gray patches (144 columns) from four scikit-image test images          [offline]
  ecg_hankel    : 3588 x 128 delay-embedding matrix of the real ECG record of scipy.datasets                [downloads one file from GitHub]
  digits_kernel : 1000 x 1000 Gaussian kernel matrix (median-distance bandwidth) of handwritten digits    [offline]
  olivetti      : 400 x 4096 Olivetti faces                                                                 [needs sklearn.fetch_olivetti_faces: figshare download; NOT tested by the author of this script]
Methods and protocol: exactly those of stage3_final_experiments.py (RBRP, QRCP, Vol(f=1), VolGuard-500, Ours-T10, Hybrid-T10; same guard 1.05;
same RBRP block size), 10 RBRP seeds by default, no full scan by default (--full-seeds N runs it for the smallest rank).
Optional --noise also runs the downstream perturbation experiment of stage3_noise_downstream.py on the same matrices.

Usage (next to stage1.py and the earlier stage2/stage3 scripts, including realdata_loaders.py):
    python stage3_realdata.py --report-only          # spectra / optimal errors, no experiment
    python stage3_realdata.py --smoke
    python stage3_realdata.py                        # image_patches, ecg_hankel, digits_kernel
    python stage3_realdata.py --only olivetti        # after checking --report-only --only olivetti
    python stage3_realdata.py --summarize-only
"""
from __future__ import annotations
import argparse, importlib.util, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # repository root (results/ lives here)
sys.path.insert(0, str(HERE))
import realdata_loaders as L

# name -> (loader(smoke), ranks)
REAL = {
    'image_patches': (lambda s: L.load_image_patches(n=600 if s else 3000), [8, 16, 32]),
    'ecg_hankel': (lambda s: L.load_ecg_hankel(n_samples=8000 if s else 36000), [8, 16, 32]),
    'digits_kernel': (lambda s: L.load_digits_kernel(n=300 if s else 1000), [8, 16, 32]),
    'olivetti': (lambda s: L.load_olivetti_faces(), [10, 20, 40]),
}
DEFAULT = ['image_patches', 'ecg_hankel', 'digits_kernel']


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def report(name, smoke):
    X = REAL[name][0](smoke)
    s = np.linalg.svd(X, compute_uv=False); tot = (s ** 2).sum()
    ks = REAL[name][1]
    print(f'{name}: {X.shape[0]} x {X.shape[1]}; optimal relative rank-k error: ' +
          ', '.join(f'k={k}: {np.sqrt((s[k:] ** 2).sum() / tot):.4f}' for k in ks) +
          f';  s_k/s_1: ' + ', '.join(f'k={k}: {s[k - 1] / s[0]:.1e}' for k in ks), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1', default=str(HERE / 'stage1.py'))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'stage3_realdata_results'))
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--full-seeds', type=int, default=0)
    ap.add_argument('--only', default=None)
    ap.add_argument('--noise', action='store_true')
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--summarize-only', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = [a.only] if a.only else DEFAULT
    if a.smoke:
        a.seeds = 2
    if a.report_only:
        for n in names:
            try: report(n, a.smoke)
            except Exception as e: print(f'{n}: FAILED to load ({type(e).__name__}: {str(e)[:120]})')
        return
    fe = load_module('stage3_final_experiments', 'stage3_final_experiments.py')
    if a.summarize_only:
        fe.summarize(out); return
    base = load_module('stage2_sorted_scan', 'stage2_sorted_scan.py')
    rb = load_module('stage3_realistic_baseline', 'stage3_realistic_baseline.py')
    vb = load_module('stage3_volume_baseline', 'stage3_volume_baseline.py')
    mod = base.load_stage1(Path(a.stage1))
    cache = {}
    orig = rb.get_matrix
    def get_matrix(name, smoke):
        if name in REAL:
            if (name, smoke) not in cache:
                cache[(name, smoke)] = REAL[name][0](smoke)
            return cache[(name, smoke)]
        return orig(name, smoke)
    rb.get_matrix = get_matrix
    t0 = time.perf_counter()
    for n in names:
        if n not in REAL:
            print('unknown matrix', n); continue
        ks = REAL[n][1][:1] if a.smoke else REAL[n][1]
        fe.FULL_KS[n] = [min(REAL[n][1])]
        print(f'=== {n}: k in {ks} ===', flush=True)
        try:
            report(n, a.smoke)
            fe.run_matrix(base, mod, rb, vb, n, 'real', ks, a.seeds, a.full_seeds, {}, a.smoke, out)
        except Exception as e:
            print(f'{n}: FAILED ({type(e).__name__}: {str(e)[:200]})'); continue
    if a.noise:
        nd = load_module('stage3_noise_downstream', 'stage3_noise_downstream.py')
        nout = Path(str(out) + '_noise'); nout.mkdir(parents=True, exist_ok=True); mc = []
        for n in names:
            if n in REAL:
                nd.run_matrix(base, mod, rb, vb, n, 'real', REAL[n][1][:1] if a.smoke else REAL[n][1], a.seeds, a.smoke, nout, mc)
        import pandas as pd
        if mc: pd.DataFrame(mc).to_csv(nout / 'noise_mc.csv', index=False)
        nd.summarize(nout)
    print(f'\nwall time {time.perf_counter() - t0:.0f}s')
    fe.summarize(out)


if __name__ == '__main__':
    main()
