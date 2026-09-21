"""
build_dataset.py
================
Run the extractor over a dataset laid out as   root/<split>/<class>/*.png
(e.g. D:\\PROJECTS\\64feature_NN\\dataset_images\\train\\fuelcap\\image_00001.png)
and write everything the training script and the board need.

    python build_dataset.py --root D:\\PROJECTS\\64feature_NN\\dataset_images --out features

Outputs in --out
    features_<split>.npz   X (N,64) float32, y (N,), paths
    int8_<split>.npz       Xq (N,64) int8,   y (N,)      <- what the FPGA will receive
    scaler.npz             mean / std / clip   (fitted on the TRAIN split only)
    classes.json           class index -> name
    feature_names.json

A per-class report of segmentation failures (no object found) is printed; if a class has
a high failure rate, look at it with check_masks.py before training.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fod_features import extract_features, FEATURE_NAMES, N_FEATURES
from fod_quant import FeatureScaler

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def discover(root):
    """Return {split: {class_name: [paths]}}. Supports root/split/class and root/class."""
    root = Path(root)
    children = sorted(p for p in root.iterdir() if p.is_dir())
    if any(list_images(c) for c in children):                    # root/class/*.png
        return {"all": {c.name: list_images(c) for c in children}}
    return {s.name: {c.name: list_images(c)
                     for c in sorted(p for p in s.iterdir() if p.is_dir())}
            for s in children}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="features")
    ap.add_argument("--train-split", default="train", help="split used to fit the scaler")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = discover(args.root)
    if args.train_split not in data:
        raise SystemExit(f"train split '{args.train_split}' not found; splits are {list(data)}")

    classes = sorted(data[args.train_split])
    class_id = {c: i for i, c in enumerate(classes)}
    print("Classes:", class_id)

    results = {}
    for split, cmap in data.items():
        X, y, paths = [], [], []
        stats = {}                                                # class -> [n, failed, area_sum]
        t0 = time.perf_counter()
        for cname, files in cmap.items():
            if cname not in class_id:
                print(f"  [{split}] skipping unknown class folder '{cname}'")
                continue
            st = stats.setdefault(cname, [0, 0, 0.0])
            for k, p in enumerate(files):
                try:
                    f, dbg = extract_features(p, return_debug=True)
                except Exception as e:                            # unreadable file etc.
                    print(f"  [{split}] FAILED {p}: {e}")
                    continue
                X.append(f)
                y.append(class_id[cname])
                paths.append(str(p))
                st[0] += 1
                st[1] += (not dbg["found"])
                st[2] += float(dbg["mask"].sum())
                if (k + 1) % 200 == 0:
                    print(f"  [{split}/{cname}] {k + 1}/{len(files)}")
        X = np.asarray(X, dtype=np.float32).reshape(-1, N_FEATURES)
        y = np.asarray(y, dtype=np.int64)
        results[split] = (X, y, paths)

        print(f"\n[{split}] {len(y)} images in {time.perf_counter() - t0:.1f} s")
        for cname, (n, bad, area) in stats.items():
            if n:
                print(f"    {cname:15s} n={n:5d}  no-object={100 * bad / n:5.1f}%  "
                      f"mean mask area={area / n:7.0f} px")

    # ---- scaler: fit on train only ----
    Xtr = results[args.train_split][0]
    scaler = FeatureScaler().fit(Xtr)
    scaler.save(out / "scaler.npz")

    const = [n for n, s in zip(FEATURE_NAMES, Xtr.std(axis=0)) if s < 1e-6]
    if const:
        print("\nWARNING: constant features in the training set (carry no information):", const)

    # ---- write outputs ----
    for split, (X, y, paths) in results.items():
        np.savez(out / f"features_{split}.npz", X=X, y=y, paths=np.array(paths))
        np.savez(out / f"int8_{split}.npz", Xq=scaler.quantize(X), y=y)
        sat = (np.abs(scaler.standardize(X)) >= scaler.clip - 1e-6).mean() * 100
        print(f"[{split}] saved. Values clipped at +-{scaler.clip:g} sigma: {sat:.2f}%")

    (out / "classes.json").write_text(json.dumps(classes, indent=2))
    (out / "feature_names.json").write_text(json.dumps(list(FEATURE_NAMES), indent=2))
    print(f"\nDone -> {out.resolve()}")


if __name__ == "__main__":
    main()
