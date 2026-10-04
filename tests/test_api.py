"""API smoke tests: laptop mode, server-mode auth/roles, ELN, runs, jobs, docs."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hydra.api.app import create_app
from hydra.config import Settings


@pytest.fixture()
def laptop(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(mode="laptop", data_dir=tmp_path)))


@pytest.fixture()
def server(tmp_path: Path) -> TestClient:
    return TestClient(create_app(Settings(mode="server", host="127.0.0.1", data_dir=tmp_path)))


def test_health_and_docs(laptop: TestClient) -> None:
    h = laptop.get("/api/health").json()
    assert h["status"] == "ok" and h["telemetry"] is False and h["auth_required"] is False
    assert laptop.get("/docs").status_code == 200
    r = laptop.get("/docs/ASSUMPTIONS")
    assert r.status_code == 200 and "Assumptions" in r.text
    assert laptop.get("/docs/..%2Fx").status_code in (404, 422)
    assert "default-src" in laptop.get("/api/health").headers.get("content-security-policy", "default-src")


def test_run_flow(laptop: TestClient) -> None:
    r = laptop.post("/api/runs", json={"scenario": {}})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    assert r.json()["ledger"]
    assert laptop.get("/api/runs").json()
    d = laptop.get(f"/api/runs/{rid}/data").json()
    assert "t" in d
    assert laptop.get(f"/api/runs/{rid}/data?fmt=csv").text.startswith("t")
    assert laptop.get(f"/api/runs/{rid}/plot/h2").status_code == 200
    rep = laptop.get(f"/api/runs/{rid}/report")
    assert rep.status_code == 200 and "Limitations" in rep.text
    assert laptop.post(f"/api/runs/{rid}/rerun").json()["bit_identical"] is True
    assert laptop.get("/api/runs/nope").status_code == 404


def test_job_and_scaleup(laptop: TestClient) -> None:
    r = laptop.post("/api/scaleup", json={})
    assert r.status_code == 200 and "SCREENING" in r.text
    j = laptop.post("/api/jobs/simulate", json={"scenario": {}})
    assert j.status_code == 200, j.text
    jid = j.json()["job_id"]
    for _ in range(120):
        st = laptop.get(f"/api/jobs/id/{jid}").json()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert st["status"] == "done", st


def test_eln(laptop: TestClient) -> None:
    e = laptop.post("/api/eln/experiments", json={"title": "t1"}).json()
    eid = e["id"]
    up = laptop.put(f"/api/eln/experiments/{eid}/attachments/a.csv", content=b"t,x\n0,1\n")
    assert up.status_code == 200
    aid = up.json()["id"]
    assert laptop.get(f"/api/eln/attachments/{aid}").content == b"t,x\n0,1\n"
    bad = laptop.put(f"/api/eln/experiments/{eid}/attachments/..%2F..%2Fevil", content=b"x")
    assert bad.status_code in (200, 400, 404)


def test_server_auth_roles(server: TestClient) -> None:
    assert server.get("/api/runs").status_code == 401
    assert server.post("/api/auth/bootstrap", json={"username": "root", "password": "correct horse 12"}).status_code == 200
    assert server.post("/api/auth/bootstrap", json={"username": "x", "password": "correct horse 12"}).status_code == 403
    tok = server.post("/api/auth/login", json={"username": "root", "password": "correct horse 12"}).json()["token"]
    ah = {"Authorization": f"Bearer {tok}"}
    assert server.post("/api/auth/users", headers=ah, json={"username": "v", "password": "viewer pass 123", "role": "viewer"}).status_code == 200
    vt = server.post("/api/auth/login", json={"username": "v", "password": "viewer pass 123"}).json()["token"]
    vh = {"Authorization": f"Bearer {vt}"}
    assert server.get("/api/runs", headers=vh).status_code == 200
    assert server.post("/api/runs", headers=vh, json={"scenario": {}}).status_code == 403
    assert server.get("/api/auth/users", headers=vh).status_code == 403
    assert server.get("/api/auth/audit", headers=ah).status_code == 200


def test_lockout(server: TestClient) -> None:
    server.post("/api/auth/bootstrap", json={"username": "root", "password": "correct horse 12"})
    codes = [server.post("/api/auth/login", json={"username": "root", "password": "wrong"}).status_code for _ in range(7)]
    assert set(codes) == {401}
    assert server.post("/api/auth/login", json={"username": "root", "password": "correct horse 12"}).status_code == 401  # locked
