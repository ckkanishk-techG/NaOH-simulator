import pytest
from pydantic import ValidationError

from hydra.config import Settings


def test_laptop_default_is_loopback_no_telemetry():
    s = Settings()
    assert s.host == "127.0.0.1" and s.mode == "laptop" and s.telemetry is False


def test_laptop_rejects_lan_binding():
    with pytest.raises(ValidationError):
        Settings(mode="laptop", host="0.0.0.0")


def test_server_allows_lan_binding():
    assert Settings(mode="server", host="0.0.0.0").host == "0.0.0.0"
