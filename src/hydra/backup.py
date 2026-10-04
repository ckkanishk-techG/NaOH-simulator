"""Backups (server mode): consistent database snapshot + run arrays + ELN files in one tar.gz, with retention."""

from __future__ import annotations

import tarfile
import tempfile
import time
from pathlib import Path

from .db import Database


def create_backup(db: Database, data_dir: Path, keep: int = 7) -> Path:
    """Write ``<data_dir>/backups/hydra-YYYYmmdd-HHMMSS.tar.gz`` (the HMAC key is NOT included) and prune old ones."""
    bdir = Path(data_dir) / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = bdir / f"hydra-{stamp}.tar.gz"
    n = 1
    while out.exists():
        out = bdir / f"hydra-{stamp}-{n}.tar.gz"
        n += 1
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / "hydra.sqlite3"
        if db.kind == "sqlite":
            db.backup(snap)
        with tarfile.open(out, "w:gz") as tf:
            if snap.exists():
                tf.add(snap, arcname="hydra.sqlite3")
            for sub in ("runs", "eln"):
                p = Path(data_dir) / sub
                if p.exists():
                    tf.add(p, arcname=sub)
    old = sorted(bdir.glob("hydra-*.tar.gz"))
    for f in old[:-keep] if keep > 0 else []:
        f.unlink()
    return out


def restore_backup(archive: Path, dest: Path) -> Path:
    """Extract a backup into ``dest`` (refuses members that would escape it)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"unsafe path in archive: {m.name}")
        tf.extractall(dest, filter="data")
    return dest
