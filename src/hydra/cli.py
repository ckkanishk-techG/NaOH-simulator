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


def worker_main(argv: list[str] | None = None) -> None:
    """``hydra-worker``: RQ worker for server mode (needs the ``server`` extra and a local Redis)."""
    ap = argparse.ArgumentParser(prog="hydra-worker")
    ap.add_argument("--redis", default="redis://localhost:6379/0")
    ap.add_argument("--queue", default="hydra")
    args = ap.parse_args(argv)
    from redis import Redis
    from rq import Queue, Worker

    conn = Redis.from_url(args.redis)
    Worker([Queue(args.queue, connection=conn)], connection=conn).work()


def user_main(argv: list[str] | None = None) -> None:
    """``hydra-user``: create/list users and set roles in the server-mode database."""
    import getpass

    from .auth import AuthStore
    from .db import Database

    ap = argparse.ArgumentParser(prog="hydra-user")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--role", choices=["viewer", "researcher", "admin"], default="researcher")
    sub.add_parser("list")
    r = sub.add_parser("role")
    r.add_argument("username")
    r.add_argument("role", choices=["viewer", "researcher", "admin"])
    args = ap.parse_args(argv)
    st = Settings(mode="server")
    store = AuthStore(Database(st.db_url), st.data_dir / "secret.key")
    if args.cmd == "add":
        store.create_user(args.username, getpass.getpass("password: "), args.role)
    elif args.cmd == "role":
        store.set_role(args.username, args.role)
    else:
        for u in store.list_users():
            print(u)


if __name__ == "__main__":
    main()
