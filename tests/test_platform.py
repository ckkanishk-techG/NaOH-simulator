"""Auth, job limits, backup/restore, export, run reproducibility."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from hydra.auth import AuthError, AuthStore, hash_password, verify_password
from hydra.backup import create_backup, restore_backup
from hydra.db import Database
from hydra.export import to_csv, to_hdf5, to_npz
from hydra.jobs import run_limited


def test_scrypt_and_tokens(tmp_path: Path) -> None:
    h = hash_password("pw-12345678")
    assert verify_password("pw-12345678", h) and not verify_password("nope", h) and "pw-12345678" not in h
    now = [1000.0]
    a = AuthStore(Database(f"sqlite:///{tmp_path/'a.db'}"), tmp_path / "k", clock=lambda: now[0])
    a.create_user("u", "pw-12345678", "researcher")
    tok = a.login("u", "pw-12345678")
    assert a.verify_token(tok).role == "researcher"
    assert oct(os.stat(tmp_path / "k").st_mode & 0o777) == "0o600"
    with pytest.raises(AuthError):
        a.verify_token(tok[:-3] + "AAA")
    a.set_role("u", "viewer")
    assert a.verify_token(tok).role == "viewer"  # demotion is immediate
    now[0] += 10**7
    with pytest.raises(AuthError, match="expired"):
        a.verify_token(tok)


def _alloc(n: int) -> int:
    return len(bytearray(n))


def test_run_limited_ram(tmp_path: Path) -> None:
    assert run_limited(_alloc, 1_000_000, ram_mb=2048) == 1_000_000
    with pytest.raises(Exception):  # noqa: B017
        run_limited(_alloc, 4_000_000_000, ram_mb=512)


def test_export_and_backup(tmp_path: Path) -> None:
    s = {"t": np.linspace(0, 1, 5), "x": np.arange(5.0)}
    assert to_csv(s).splitlines()[0].startswith("t")
    p = to_npz(s, tmp_path / "s.npz")
    assert np.allclose(np.load(p)["x"], s["x"])
    pytest.importorskip("h5py")
    import h5py

    with h5py.File(to_hdf5(s, tmp_path / "s.h5"), "r") as f:
        assert np.allclose(f["series/x"][:], s["x"])
    data = tmp_path / "data"
    data.mkdir()
    (data / "secret.key").write_bytes(b"k" * 32)
    (data / "runs").mkdir()
    (data / "runs" / "r.npz").write_bytes(b"abc")
    db = Database(f"sqlite:///{data/'h.db'}")
    db.execute("CREATE TABLE t (a INTEGER)")
    db.execute("INSERT INTO t VALUES (1)")
    arc = create_backup(db, data)
    import tarfile

    with tarfile.open(arc) as tf:
        names = tf.getnames()
    assert not any("secret.key" in n for n in names) and any("r.npz" in n for n in names)
    out = restore_backup(arc, tmp_path / "restored")
    assert (out / "runs" / "r.npz").read_bytes() == b"abc"
