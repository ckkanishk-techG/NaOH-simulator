import socket
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hydra.api.app import create_app
from hydra.netguard import OutboundNetworkError, block_outbound

ROOT = Path(__file__).resolve().parents[1]


def test_guard_blocks_external_connect():
    with block_outbound(allow_loopback=False) as attempts:
        with pytest.raises(OutboundNetworkError):
            socket.create_connection(("93.184.216.34", 80), timeout=1)
        assert attempts
        attempts.clear()


def test_guard_blocks_dns_and_http():
    with block_outbound(allow_loopback=False) as attempts:
        with pytest.raises(Exception):  # noqa: B017 - urllib wraps the error
            urllib.request.urlopen("http://example.com", timeout=1)
        assert attempts
        attempts.clear()


def test_full_api_run_makes_no_outbound_calls():
    c = TestClient(create_app())
    assert c.get("/").status_code == 200
    assert c.get("/api/health").json()["telemetry"] is False
    assert c.get("/api/disclaimer").status_code == 200
    # the autouse fixture asserts no blocked attempts at teardown


def test_no_remote_urls_in_shipped_web_assets():
    bad = []
    for p in (ROOT / "src/hydra/web").rglob("*"):
        if p.suffix in {".html", ".css", ".js", ".ts", ".tsx"}:
            txt = p.read_text(errors="ignore")
            for needle in ("http://", "https://", "//cdn", "googleapis", "gtag", "analytics"):
                if needle in txt:
                    bad.append((p.name, needle))
    assert not bad, bad
