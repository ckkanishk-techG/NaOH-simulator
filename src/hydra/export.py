"""Data exports: CSV, NPZ, HDF5 (with provenance attributes) and JSON."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_csv(series: dict[str, np.ndarray], keys: list[str] | None = None) -> str:
    keys = keys or [k for k, v in series.items() if np.ndim(v) == 1 and len(v) == len(series["t"])]
    keys = ["t"] + [k for k in keys if k != "t"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(keys)
    for i in range(len(series["t"])):
        w.writerow([f"{series[k][i]:.10g}" for k in keys])
    return buf.getvalue()


def to_npz(series: dict[str, np.ndarray], path: Path) -> Path:
    kw: dict[str, Any] = dict(series)
    np.savez_compressed(path, **kw)
    return path


def to_hdf5(series: dict[str, np.ndarray], path: Path, attrs: dict[str, Any] | None = None) -> Path:
    import h5py

    with h5py.File(path, "w") as f:
        g = f.create_group("series")
        for k, v in series.items():
            g.create_dataset(k, data=np.asarray(v), compression="gzip")
        for k, v in (attrs or {}).items():
            f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return path
