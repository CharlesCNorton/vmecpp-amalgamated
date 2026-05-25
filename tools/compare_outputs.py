"""Exact dataset + attribute comparison of two VMEC++ .out.h5 files.

Reports which datasets differ and by how much (max abs / max rel), separating
benign HDF5 container metadata from real numerical differences.

Usage:
  python compare_outputs.py a.out.h5 b.out.h5

Requires h5py and numpy.
"""
import sys
import numpy as np
import h5py

if len(sys.argv) != 3:
    sys.exit("usage: python compare_outputs.py a.out.h5 b.out.h5")
fa, fb = sys.argv[1], sys.argv[2]


def walk(h5):
    items = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            items[name] = obj[()]
    h5.visititems(visit)
    return items


def all_attrs(h5):
    """Collect attributes from the root, every group and every dataset."""
    out = {}

    def grab(name, obj):
        for ak, av in obj.attrs.items():
            out[f"{name}@{ak}"] = av
    for ak, av in h5.attrs.items():
        out[f"/@{ak}"] = av
    h5.visititems(grab)
    return out


with h5py.File(fa, "r") as A, h5py.File(fb, "r") as B:
    # attributes (units, descriptions, version strings, ...)
    aa, ab = all_attrs(A), all_attrs(B)
    attr_keys = set(aa) | set(ab)
    attr_diff = 0
    for k in sorted(attr_keys):
        if k not in aa or k not in ab:
            print(f"ATTR only in one file: {k}")
            attr_diff += 1
            continue
        if not np.array_equal(np.asarray(aa[k]), np.asarray(ab[k])):
            print(f"ATTR DIFF {k}: {aa[k]!r} vs {ab[k]!r}")
            attr_diff += 1
    print(f"attributes: {len(attr_keys)} total, {attr_diff} differ")

    da, db = walk(A), walk(B)
    ka, kb = set(da), set(db)
    only_a, only_b = ka - kb, kb - ka
    if only_a:
        print("only in A:", sorted(only_a))
    if only_b:
        print("only in B:", sorted(only_b))

    n_equal = n_diff = 0
    worst = []  # (max_abs, max_rel, name, shape)
    diff_names = []
    for k in sorted(ka & kb):
        va, vb = np.asarray(da[k]), np.asarray(db[k])
        if va.shape != vb.shape:
            print(f"SHAPE DIFF {k}: {va.shape} vs {vb.shape}")
            n_diff += 1
            diff_names.append(k)
            continue
        if va.dtype.kind in "fc" and vb.dtype.kind in "fc":
            if np.array_equal(va, vb):
                n_equal += 1
                continue
            d = np.abs(va.astype(np.float64) - vb.astype(np.float64))
            denom = np.maximum(np.abs(va.astype(np.float64)), 1e-300)
            rel = (d / denom)
            mab = float(np.nanmax(d)) if d.size else 0.0
            mrl = float(np.nanmax(rel)) if rel.size else 0.0
            worst.append((mab, mrl, k, va.shape))
            diff_names.append(k)
            n_diff += 1
        else:
            if np.array_equal(va, vb):
                n_equal += 1
            else:
                print(f"NON-FLOAT DIFF {k}: dtype {va.dtype}")
                diff_names.append(k)
                n_diff += 1

    print(f"\ndatasets: {len(ka & kb)} common, {n_equal} exactly equal, "
          f"{n_diff} differ")
    if worst:
        worst.sort(reverse=True)
        print("\ntop float datasets by max abs diff:")
        for mab, mrl, k, shp in worst[:25]:
            print(f"  {mab:.3e} abs  {mrl:.3e} rel  {k} {shp}")
        allmax_abs = max(w[0] for w in worst)
        allmax_rel = max(w[1] for w in worst)
        print(f"\nGLOBAL max abs diff: {allmax_abs:.3e}")
        print(f"GLOBAL max rel diff: {allmax_rel:.3e}")
    if n_diff == 0:
        print("\nALL DATASETS BIT-FOR-BIT IDENTICAL")
