"""
fod_features.py
===============
Object-centric 64-feature extractor for FOD (foreign object debris) images.

Runs on a PC or on the Zynq PS (ARM Cortex-A9 / PYNQ). Dependencies: OpenCV + NumPy only
(no SciPy / scikit-image), so the same file can be used on both.

Pipeline
--------
1. Load BGR image, resize to IMG_SIZE x IMG_SIZE, convert to CIE Lab (float).
2. Background model: robust (iteratively trimmed) polynomial surface fitted to Lab.
   It follows lighting gradients / vignetting and ignores the small foreign object.
3. Anomaly map = Lab distance between the (lightly blurred) image and the background model.
4. Threshold (adaptive, from the noise level of the map) -> morphology -> keep the object.
5. Compute 64 features from inside the mask + contrast against a ring around it.

Feature groups (64 total)
-------------------------
    Object colour           18
    Contrast vs background   7
    Shape                   16
    Gradient (object)       11
    Texture (object)        10
    Background context       2

Heavy-tailed quantities (area, perimeter, Hu moments, texture ratio) are log-transformed
inside the extractor. All features are invariant to object position, rotation and mirroring
(gradient orientations are measured relative to the object's own principal axis). Final standardisation / INT8 scaling lives in fod_quant.py.

Note: the image is resized to a SQUARE. Dataset images are 400x400 so nothing is distorted.
If you ever feed non-square images, pad them to square first (shape features depend on it).
"""

import cv2
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 256        # working resolution (object is ~25 px at this size for the small samples)
BG_FIT_SIZE = 64      # background surface is fitted on a 64x64 copy (fast)
BG_POLY_DEG = 2       # polynomial degree of the background surface
SMOOTH_SIGMA = 1.5    # blur before the anomaly map (suppresses pavement grain)
THR_K = 6.0           # threshold = median + THR_K * robust-sigma of the anomaly map
THR_MIN = 6.0         # ... but never below this Lab distance
MIN_OBJ_AREA = 12     # pixels (at IMG_SIZE) below which we declare "no object found"
MERGE_DIST = 15       # fragments of the same object closer than this are merged (pixels)
C0 = 5.0              # chroma shrinkage for hue features (grey objects -> hue ~ 0)

N_FEATURES = 64

FEATURE_NAMES = (
    ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
     "hue_cos", "hue_sin", "chroma_mean"]
    + [f"hue_hist_{i}" for i in range(8)]
    + ["highlight_frac"]                                            # 18 colour
    + ["dL", "da", "db", "dE_mean", "dE_max", "boundary_grad",
       "log_texture_ratio"]                                         # 7 contrast
    + ["log_area_ratio", "log_perimeter_ratio", "rect_aspect", "rect_extent",
       "solidity", "circularity", "eccentricity", "n_holes", "n_convex_defects"]
    + [f"hu_{i}" for i in range(1, 8)]                              # 16 shape
    + [f"grad_hist_{i}" for i in range(8)]
    + ["edge_density_inside", "edge_density_boundary", "grad_mean"]  # 11 gradient
    + [f"lbp_{i}" for i in range(8)]
    + ["L_std_inside", "L_entropy"]                                 # 10 texture
    + ["bg_L_mean", "bg_texture_std"]                               # 2 context
)
assert len(FEATURE_NAMES) == N_FEATURES

_COLOR = slice(0, 18)
_CONTRAST = slice(18, 25)
_SHAPE = slice(25, 41)
_GRAD = slice(41, 52)
_TEX = slice(52, 62)
_CTX = slice(62, 64)


# ============================================================
# HELPERS
# ============================================================
_ELL_CACHE = {}


def _ell(k):
    """Cached elliptical structuring element of size k x k."""
    if k not in _ELL_CACHE:
        _ELL_CACHE[k] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return _ELL_CACHE[k]


_GRID_CACHE = {}


def _grid_design(n):
    """Polynomial design matrix for an n x n grid over [-1, 1]^2 (cached)."""
    key = (n, BG_POLY_DEG)
    if key not in _GRID_CACHE:
        g = np.linspace(-1.0, 1.0, n)
        xx, yy = np.meshgrid(g, g)
        x, y = xx.ravel(), yy.ravel()
        cols = [(x ** i) * (y ** j)
                for i in range(BG_POLY_DEG + 1)
                for j in range(BG_POLY_DEG + 1 - i)]
        _GRID_CACHE[key] = np.stack(cols, axis=1)
    return _GRID_CACHE[key]


# ============================================================
# 1. LOADING
# ============================================================
def load_image(path):
    """Read an image file -> (bgr_uint8, lab_float32) at IMG_SIZE x IMG_SIZE."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return prepare_bgr(bgr)


def prepare_bgr(bgr):
    """Resize a BGR uint8 array (e.g. a camera frame) -> (bgr_uint8, lab_float32)."""
    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # float32 input -> L in [0,100], a/b roughly [-127,127]
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return bgr, lab


# ============================================================
# 2. BACKGROUND MODEL + SEGMENTATION
# ============================================================
def fit_background(lab):
    """Robust polynomial surface per Lab channel -> (IMG_SIZE, IMG_SIZE, 3) float32."""
    n = BG_FIT_SIZE
    small = cv2.resize(lab, (n, n), interpolation=cv2.INTER_AREA)
    small = small.reshape(-1, 3).astype(np.float64)
    A = _grid_design(n)

    keep = np.ones(n * n, dtype=bool)
    coef = None
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(A[keep], small[keep], rcond=None)
        d = np.linalg.norm(small - A @ coef, axis=1)
        med = np.median(d[keep])
        sig = 1.4826 * np.median(np.abs(d[keep] - med)) + 1e-3
        new_keep = d < med + 3.0 * sig
        if new_keep.sum() < 0.3 * n * n:      # never throw away most of the image
            break
        keep = new_keep

    full = _grid_design(IMG_SIZE) @ coef
    return full.reshape(IMG_SIZE, IMG_SIZE, 3).astype(np.float32)


def segment(lab):
    """
    Returns dict with:
        mask     uint8 {0,1}   object mask (all zeros if nothing found)
        anomaly  float32       Lab distance to the background model
        lab_s    float32       lightly blurred Lab image
        thr      float         threshold that was used
        found    bool
    """
    lab_s = cv2.GaussianBlur(lab, (0, 0), SMOOTH_SIGMA)
    bg = fit_background(lab)
    anomaly = np.linalg.norm(lab_s - bg, axis=2).astype(np.float32)

    med = float(np.median(anomaly))
    sig = 1.4826 * float(np.median(np.abs(anomaly - med)))
    thr = max(med + THR_K * sig, THR_MIN)

    binary = (anomaly > thr).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _ell(3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _ell(7))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros_like(binary)
    found = False

    if n > 1:
        # strongest component = largest total anomaly (area x contrast)
        score = np.bincount(labels.ravel(), weights=anomaly.ravel(), minlength=n)
        score[0] = 0.0
        best = int(score.argmax())
        if stats[best, cv2.CC_STAT_AREA] >= MIN_OBJ_AREA:
            main = (labels == best).astype(np.uint8)
            near = cv2.dilate(main, _ell(2 * MERGE_DIST + 1))
            ids = np.unique(labels[(near > 0) & (labels > 0)])
            ids = ids[stats[ids, cv2.CC_STAT_AREA] >= 8]     # merge nearby fragments
            mask = np.isin(labels, ids).astype(np.uint8)
            found = True

    return dict(mask=mask, anomaly=anomaly, lab_s=lab_s, thr=thr, found=found)


# ============================================================
# 3. FEATURE COMPUTATION
# ============================================================
_LBP_LUT = np.array([0, 1, 1, 2, 3, 4, 5, 6, 6, 7])   # 10 rotation-invariant-uniform codes -> 8 bins


def _lbp_hist8(L, sel):
    """
    8-neighbour rotation-invariant uniform LBP on L, histogram over pixels in `sel`.
    Codes 0..8 = number of 1-bits for uniform patterns, 9 = non-uniform; merged into 8 bins.
    """
    c = L[1:-1, 1:-1]
    nb = [L[:-2, :-2], L[:-2, 1:-1], L[:-2, 2:], L[1:-1, 2:],
          L[2:, 2:], L[2:, 1:-1], L[2:, :-2], L[1:-1, :-2]]           # clockwise ring
    bits = np.stack([n >= c for n in nb], axis=0)
    ones = bits.sum(axis=0)
    trans = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    code = np.where(trans <= 2, ones, 9)
    codes = _LBP_LUT[code[sel[1:-1, 1:-1]]]
    hist = np.bincount(codes, minlength=8).astype(np.float32)
    return hist / max(hist.sum(), 1.0)


def compute_features(lab, seg):
    """64-vector from a Lab image and the segmentation dict."""
    f = np.zeros(N_FEATURES, dtype=np.float32)

    mask = seg["mask"]
    anomaly = seg["anomaly"]
    Ls = seg["lab_s"][:, :, 0]
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    m = mask.astype(bool)
    area = int(m.sum())

    # ---------- background context (always available) ----------
    near = cv2.dilate(mask, _ell(7)).astype(bool)
    bgmask = ~near
    if bgmask.sum() < 100:
        bgmask = np.ones_like(m)
    hp = L - cv2.GaussianBlur(L, (0, 0), 3.0)             # high-pass: pavement grain
    bg_L = float(L[bgmask].mean())
    bg_tex = float(hp[bgmask].std())
    f[_CTX] = [bg_L / 100.0, bg_tex]

    # ---------- nothing found: only anomaly peak + context are meaningful ----------
    if area < MIN_OBJ_AREA:
        f[22] = float(anomaly.max())                      # dE_max slot
        return f

    Lo, ao, bo = L[m], a[m], b[m]

    # ==========================================================
    # OBJECT COLOUR (18)
    # ==========================================================
    chroma = np.hypot(ao, bo)
    hue = np.arctan2(bo, ao)
    denom = chroma.sum() + C0 * chroma.size               # shrinks hue -> 0 for grey objects
    hue_hist, _ = np.histogram(hue, bins=8, range=(-np.pi, np.pi), weights=chroma)
    highlight = np.mean((Lo > bg_L + 15.0) & (chroma < 20.0))

    f[0:3] = [Lo.mean(), ao.mean(), bo.mean()]
    f[3:6] = [Lo.std(), ao.std(), bo.std()]
    f[6] = (chroma * np.cos(hue)).sum() / denom
    f[7] = (chroma * np.sin(hue)).sum() / denom
    f[8] = chroma.mean() / 100.0
    f[9:17] = hue_hist / denom
    f[17] = highlight

    # ==========================================================
    # CONTRAST VS BACKGROUND (7)
    # ==========================================================
    ring = (cv2.dilate(mask, _ell(31)) > 0) & ~near
    if ring.sum() < 50:
        ring = bgmask

    gx = cv2.Sobel(Ls, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(Ls, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.sqrt(gx * gx + gy * gy)

    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, _ell(3)).astype(bool)
    tex_ratio = (hp[m].std() + 0.05) / (hp[ring].std() + 0.05)

    f[18] = Lo.mean() - L[ring].mean()
    f[19] = ao.mean() - a[ring].mean()
    f[20] = bo.mean() - b[ring].mean()
    f[21] = anomaly[m].mean()
    f[22] = anomaly[m].max()
    f[23] = grad[boundary].mean() if boundary.any() else 0.0
    f[24] = np.log(tex_ratio)

    # ==========================================================
    # SHAPE (16)
    # ==========================================================
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hier = hier[0]
    outer_idx = [i for i in range(len(contours)) if hier[i][3] == -1]
    outer = [contours[i] for i in outer_idx]
    main_i = max(outer_idx, key=lambda i: cv2.contourArea(contours[i]))
    main_c = contours[main_i]

    n_holes = sum(1 for i in range(len(contours))
                  if hier[i][3] != -1 and cv2.contourArea(contours[i]) >= 4)

    all_pts = np.concatenate(outer, axis=0)
    hull = cv2.convexHull(all_pts)
    hull_area = cv2.contourArea(hull)
    perimeter = sum(cv2.arcLength(c, True) for c in outer)

    (_, _), (rw, rh), _ = cv2.minAreaRect(all_pts)
    rect_aspect = min(rw, rh) / max(rw, rh, 1e-6)          # 1 = square, ->0 = elongated
    rect_extent = area / max(rw * rh, 1e-6)
    solidity = area / max(hull_area, 1e-6)
    circularity = float(np.clip(4.0 * np.pi * area / max(perimeter, 1e-6) ** 2, 0.0, 1.0))

    mo = cv2.moments(mask, binaryImage=True)
    mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
    common = np.sqrt(max(((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2, 0.0))
    lam1 = (mu20 + mu02) / 2.0 + common
    lam2 = (mu20 + mu02) / 2.0 - common
    ecc = np.sqrt(max(1.0 - lam2 / lam1, 0.0)) if lam1 > 1e-8 else 0.0
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)      # principal-axis angle

    
    n_defects = 0
    if len(main_c) >= 5:
        try:
            hidx = cv2.convexHull(main_c, returnPoints=False)
            defects = cv2.convexityDefects(main_c, hidx)
            if defects is not None:
                defects = defects.reshape(-1, 4)
                n_defects = int(np.sum(defects[:, 3] / 256.0 > 2.5))
        except cv2.error:
            n_defects = 0

    # Hu moments: log-MAGNITUDE only. The sign of hu5..hu7 flips under mirroring and is
    # random noise for near-symmetric shapes, so it is deliberately discarded.
    hu = cv2.HuMoments(mo).ravel()
    hu_log = -np.log10(np.maximum(np.abs(hu), 1e-12))          # all >= 0, at most 12

    f[25] = np.log10(area / float(IMG_SIZE * IMG_SIZE))
    f[26] = np.log10(max(perimeter, 1.0) / (4.0 * IMG_SIZE))
    f[27] = rect_aspect
    f[28] = rect_extent
    f[29] = solidity
    f[30] = circularity
    f[31] = ecc
    f[32] = min(n_holes, 5)
    f[33] = min(n_defects, 12)
    f[34:41] = hu_log

    # ==========================================================
    # GRADIENT INSIDE / AROUND OBJECT (11)
    # ==========================================================
    region = cv2.dilate(mask, _ell(5)).astype(bool)         # object + its boundary
    ang = np.arctan2(gy, gx) % np.pi                        # unsigned orientation
    rel = (ang[region] - theta) % np.pi                     # relative to principal axis
    rel = np.minimum(rel, np.pi - rel)                      # fold to [0, 90 deg]: mirror-invariant
    gh, _ = np.histogram(rel, bins=8, range=(0.0, np.pi / 2), weights=grad[region])
    gh = gh / max(gh.sum(), 1e-8)

    L8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(L8, (0, 0), 1.0), 30, 90) > 0
    inner = cv2.erode(mask, _ell(3)).astype(bool)
    if inner.sum() < 8:
        inner = m

    f[41:49] = gh
    f[49] = edges[inner].mean()
    f[50] = edges[boundary].mean() if boundary.any() else 0.0
    f[51] = grad[m].mean()

    # ==========================================================
    # TEXTURE INSIDE OBJECT (10)
    # ==========================================================
    p, _ = np.histogram(Lo, bins=32, range=(0.0, 100.0))
    p = p[p > 0] / float(Lo.size)
    entropy = float(-(p * np.log2(p)).sum()) / 5.0          # normalised to [0, 1]

    f[52:60] = _lbp_hist8(L, inner)
    f[60] = Lo.std() / 10.0
    f[61] = entropy

    return f


# ============================================================
# 4. PUBLIC API
# ============================================================
def extract_from_bgr(bgr, return_debug=False):
    """64 features from a BGR uint8 array (camera frame / cv2.imread result)."""
    bgr_s, lab = prepare_bgr(bgr)
    seg = segment(lab)
    feats = np.nan_to_num(compute_features(lab, seg), nan=0.0, posinf=0.0, neginf=0.0)
    feats = feats.astype(np.float32)
    assert feats.shape == (N_FEATURES,)
    if return_debug:
        dbg = dict(seg, bgr=bgr_s)
        return feats, dbg
    return feats


def extract_features(image_path, return_debug=False):
    """64 features from an image file. With return_debug=True also returns the mask etc."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return extract_from_bgr(bgr, return_debug=return_debug)


# ============================================================
# QUICK SELF-TEST:  python fod_features.py  path/to/image.png
# ============================================================
if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("usage: python fod_features.py <image.png>")
        sys.exit(1)

    t0 = time.perf_counter()
    feats, dbg = extract_features(Path(sys.argv[1]), return_debug=True)
    dt = (time.perf_counter() - t0) * 1000

    print(f"Object found : {dbg['found']}   (mask area = {int(dbg['mask'].sum())} px, thr = {dbg['thr']:.2f})")
    print(f"Extraction   : {dt:.1f} ms")
    print(f"Shape        : {feats.shape}   min={feats.min():.3f}  max={feats.max():.3f}")
    for name, v in zip(FEATURE_NAMES, feats):
        print(f"  {name:22s} {v: .4f}")
