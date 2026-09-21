"""
check_masks.py
==============
Visual sanity check of the segmentation BEFORE you trust the features.

For every image it saves a 3-panel picture:
    [ original | anomaly heat-map | mask outline on image ]

Usage
-----
    # specific images
    python check_masks.py img1.png img2.png

    # random sample from a dataset folder (root/split/class/*.png) - N per class
    python check_masks.py --root D:\\PROJECTS\\64feature_NN\\dataset_images --split test --per-class 6

Output: ./mask_check/*.png  and  ./mask_check/montage_<class>.png
Also prints extraction time per image (run this ON THE BOARD to get real timing).
"""

import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np

from fod_features import extract_features, FEATURE_NAMES

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def render_panel(feats, dbg, label=""):
    bgr = dbg["bgr"]
    mask = dbg["mask"]
    anomaly = dbg["anomaly"]
    thr = dbg["thr"]

    heat = np.clip(anomaly / (3.0 * thr) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    overlay = bgr.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 1)

    status = "FOUND" if dbg["found"] else "NO OBJECT"
    txt = f"{status} area={int(mask.sum())} thr={thr:.1f}"
    cv2.putText(overlay, txt, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(bgr, label, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

    return np.hstack([bgr, heat, overlay])


def collect_from_root(root, split, per_class, seed):
    rng = random.Random(seed)
    items = []                                     # (class_name, path)
    for cdir in sorted(p for p in (Path(root) / split).iterdir() if p.is_dir()):
        files = sorted(f for f in cdir.iterdir() if f.suffix.lower() in EXTS)
        rng.shuffle(files)
        items += [(cdir.name, f) for f in files[:per_class]]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="image files")
    ap.add_argument("--root", help="dataset root containing split/class/ folders")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mask_check")
    args = ap.parse_args()

    items = [("images", Path(p)) for p in args.images]
    if args.root:
        items += collect_from_root(args.root, args.split, args.per_class, args.seed)
    if not items:
        ap.error("give image paths or --root")

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    # warm-up (first call pays one-time OpenCV/NumPy start-up costs; don't count it)
    extract_features(items[0][1])

    panels = {}
    times, missed = [], 0
    for cls, path in items:
        t0 = time.perf_counter()
        feats, dbg = extract_features(path, return_debug=True)
        times.append((time.perf_counter() - t0) * 1000)
        missed += (not dbg["found"])

        panel = render_panel(feats, dbg, f"{cls}/{path.stem}")
        cv2.imwrite(str(out / f"{cls}_{path.stem}_check.png"), panel)
        panels.setdefault(cls, []).append(panel)

    for cls, plist in panels.items():
        cv2.imwrite(str(out / f"montage_{cls}.png"), np.vstack(plist))

    print(f"{len(items)} images | no-object: {missed} | "
          f"extract time: mean {np.mean(times):.1f} ms, max {np.max(times):.1f} ms")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
