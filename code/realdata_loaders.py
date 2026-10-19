import numpy as np

def load_image_patches(n=3000, p=12, seed=0):
    """Random p x p gray-scale patches (values in [0,1]) from four scikit-image test images -> n x p^2 matrix."""
    from skimage import data, color
    imgs = [data.camera() / 255.0, color.rgb2gray(data.astronaut()), data.moon() / 255.0, data.coins() / 255.0]
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n):
        im = imgs[t % len(imgs)]
        i = rng.integers(0, im.shape[0] - p); j = rng.integers(0, im.shape[1] - p)
        rows.append(im[i:i + p, j:j + p].ravel())
    return np.asarray(rows, dtype=float)

def load_ecg_hankel(n_samples=36000, d=128, stride=10):
    """Delay-embedding (Hankel-type) matrix of the real ECG record of scipy.datasets (standardized): rows = overlapping windows."""
    import scipy.datasets as sd
    s = np.asarray(sd.electrocardiogram(), dtype=float)[:n_samples]
    s = (s - s.mean()) / s.std()
    starts = np.arange(0, len(s) - d, stride)
    return np.stack([s[a:a + d] for a in starts])

def load_digits_kernel(n=1000, seed=0):
    """Gaussian kernel matrix (bandwidth = median pairwise distance) of n handwritten digits -> n x n."""
    from sklearn.datasets import load_digits
    X = load_digits().data.astype(float)
    rng = np.random.default_rng(seed)
    X = X[rng.choice(len(X), n, replace=False)]
    G = X @ X.T; sq = np.diag(G); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0)
    sigma = np.sqrt(np.median(D2[np.triu_indices(n, 1)]))
    return np.exp(-D2 / (2 * sigma ** 2))

def load_olivetti_faces():
    """400 x 4096 Olivetti faces (needs a download: sklearn.datasets.fetch_olivetti_faces)."""
    from sklearn.datasets import fetch_olivetti_faces
    return fetch_olivetti_faces(download_if_missing=True).data.astype(float)
