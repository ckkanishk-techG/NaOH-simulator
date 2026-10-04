"""``hydra-sim``: single command, loopback-only, opens the browser."""

from __future__ import annotations

import argparse
import threading
import webbrowser

from .config import Settings
from .netguard import block_outbound


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hydra-sim", description="HYDRA local digital twin")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--mode", choices=["laptop", "server"], default="laptop")
    ap.add_argument("--host", default=None)
    args = ap.parse_args(argv)

    import uvicorn

    from .api.app import create_app

    host = args.host or ("127.0.0.1" if args.mode == "laptop" else "0.0.0.0")  # noqa: S104
    settings = Settings(mode=args.mode, host=host, port=args.port)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    url = f"http://{settings.host}:{settings.port}/"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"HYDRA listening on {url} (mode={settings.mode}, telemetry=none)")
    guard = block_outbound(allow_loopback=True) if settings.enforce_offline else None
    if guard is not None:
        guard.__enter__()  # held for the process lifetime; any external connect raises
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
