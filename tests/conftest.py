"""Every test runs with outbound networking blocked (loopback allowed)."""

import pytest

from hydra.netguard import block_outbound


@pytest.fixture(autouse=True)
def _no_outbound_network():
    with block_outbound(allow_loopback=True) as attempts:
        yield
        assert not attempts, f"outbound network attempted: {attempts}"
