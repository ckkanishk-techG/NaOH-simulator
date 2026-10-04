"""Outbound-network guard.

HYDRA must make zero external network calls. ``block_outbound`` patches the socket
layer so any attempt to connect to a non-loopback address raises ``OutboundNetworkError``.
Used by the test-suite (autouse) and optionally at runtime (``HYDRA_ENFORCE_OFFLINE=1``).
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


class OutboundNetworkError(RuntimeError):
    """Raised when code tries to reach a non-loopback host."""


def _is_loopback(host: object) -> bool:
    if isinstance(host, bytes):
        host = host.decode()
    if not isinstance(host, str):
        return False
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _check_address(address: Any, allow_loopback: bool) -> None:
    if isinstance(address, (str, bytes)):  # AF_UNIX path
        return
    host = address[0] if isinstance(address, tuple) and address else address
    if allow_loopback and _is_loopback(host):
        return
    raise OutboundNetworkError(f"blocked outbound network access to {address!r}")


@contextmanager
def block_outbound(allow_loopback: bool = True) -> Iterator[list[str]]:
    """Context manager blocking external connections; yields a list of blocked attempts."""
    attempts: list[str] = []
    orig_connect: Callable[..., Any] = socket.socket.connect
    orig_connect_ex: Callable[..., Any] = socket.socket.connect_ex
    orig_gai: Callable[..., Any] = socket.getaddrinfo

    def guard(addr: Any) -> None:
        try:
            _check_address(addr, allow_loopback)
        except OutboundNetworkError:
            attempts.append(repr(addr))
            raise

    def connect(self: socket.socket, addr: Any) -> Any:
        guard(addr)
        return orig_connect(self, addr)

    def connect_ex(self: socket.socket, addr: Any) -> Any:
        guard(addr)
        return orig_connect_ex(self, addr)

    def getaddrinfo(host: Any, *a: Any, **k: Any) -> Any:
        if not (allow_loopback and _is_loopback(host)):
            attempts.append(f"dns:{host!r}")
            raise OutboundNetworkError(f"blocked DNS lookup for {host!r}")
        return orig_gai(host, *a, **k)

    socket.socket.connect = connect  # type: ignore[method-assign,assignment]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign,assignment]
    socket.getaddrinfo = getaddrinfo
    try:
        yield attempts
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = orig_connect_ex  # type: ignore[method-assign]
        socket.getaddrinfo = orig_gai
