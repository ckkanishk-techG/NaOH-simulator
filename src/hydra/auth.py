"""Local user accounts for server mode: scrypt password hashes, roles, signed expiring tokens, login throttling, audit log.

Roles: ``viewer`` (read), ``researcher`` (read + create/run), ``admin`` (everything incl. users and backups).
Laptop mode has no auth (loopback only). No third-party identity service is ever contacted."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import Database

ROLES = ("viewer", "researcher", "admin")
RANK = {r: i for i, r in enumerate(ROLES)}
SCRYPT = {"n": 2**14, "r": 8, "p": 1}  # magic: standard interactive scrypt cost
TOKEN_TTL_S = 8 * 3600  # magic: working-day session


class AuthError(Exception):
    pass


@dataclass
class User:
    username: str
    role: str
    id: int = 0


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)  # magic: 128-bit salt
    dk = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT, dklen=32)  # magic: 256-bit key
    return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt_b, dk_b = stored.split("$")
        salt, dk = base64.b64decode(salt_b), base64.b64decode(dk_b)
        test = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(dk))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(test, dk)


class AuthStore:
    def __init__(self, db: Database, key_path: Path, max_attempts: int = 5, lock_s: float = 300.0,
                 clock: Any = time.time) -> None:
        self.db, self.clock = db, clock
        db.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, "
                   "pw_hash TEXT NOT NULL, role TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL)"
                   if db.kind == "sqlite" else
                   "CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, "
                   "role TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0, created DOUBLE PRECISION NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS audit (ts REAL, username TEXT, action TEXT, detail TEXT)")
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if not key_path.exists():
            key_path.write_bytes(secrets.token_bytes(32))  # magic: 256-bit HMAC key
            os.chmod(key_path, 0o600)
        self.key = key_path.read_bytes()
        self.max_attempts, self.lock_s = max_attempts, lock_s
        self.fail: dict[str, list[float]] = defaultdict(list)

    # --- users --------------------------------------------------------------------------------------
    def create_user(self, username: str, password: str, role: str) -> User:
        if role not in ROLES:
            raise AuthError(f"unknown role {role!r}")
        if len(password) < 10:  # magic: minimum length
            raise AuthError("password must be at least 10 characters")
        if not username.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise AuthError("username may contain letters, digits, '_', '-', '.' only")
        try:
            self.db.execute("INSERT INTO users (username, pw_hash, role, created) VALUES (?,?,?,?)",
                            (username, hash_password(password), role, self.clock()))
        except Exception as exc:
            raise AuthError(f"cannot create user {username!r}: {exc}") from exc
        self.log(username, "create_user", role)
        row = self.db.one("SELECT id FROM users WHERE username=?", (username,))
        return User(username, role, int(row["id"]) if row else 0)

    def list_users(self) -> list[dict[str, Any]]:
        return self.db.query("SELECT id, username, role, disabled, created FROM users ORDER BY id")

    def set_role(self, username: str, role: str) -> None:
        if role not in ROLES:
            raise AuthError("unknown role")
        self.db.execute("UPDATE users SET role=? WHERE username=?", (role, username))
        self.log(username, "set_role", role)

    def disable(self, username: str, disabled: bool = True) -> None:
        self.db.execute("UPDATE users SET disabled=? WHERE username=?", (1 if disabled else 0, username))
        self.log(username, "disable" if disabled else "enable", "")

    def n_users(self) -> int:
        return int(self.db.query("SELECT COUNT(*) AS n FROM users")[0]["n"])

    # --- login / tokens --------------------------------------------------------------------------------
    def _locked(self, username: str) -> bool:
        now = self.clock()
        recent = [t for t in self.fail[username] if now - t < self.lock_s]
        self.fail[username] = recent
        return len(recent) >= self.max_attempts

    def login(self, username: str, password: str) -> str:
        if self._locked(username):
            self.log(username, "login_locked", "")
            raise AuthError("too many failed attempts; try again later")
        row = self.db.one("SELECT username, pw_hash, role, disabled FROM users WHERE username=?", (username,))
        if row is None:
            verify_password(password, hash_password("x" * 12))  # constant-ish time for unknown users
        ok = row is not None and not row["disabled"] and verify_password(password, row["pw_hash"])
        if row is None or not ok:
            self.fail[username].append(self.clock())
            self.log(username, "login_failed", "")
            raise AuthError("invalid credentials")
        self.fail.pop(username, None)
        self.log(username, "login", "")
        return self.issue_token(row["username"], row["role"])

    def issue_token(self, username: str, role: str, ttl: float = TOKEN_TTL_S) -> str:
        body = base64.urlsafe_b64encode(json.dumps({"u": username, "r": role, "exp": self.clock() + ttl}).encode()).decode()
        sig = base64.urlsafe_b64encode(hmac.new(self.key, body.encode(), hashlib.sha256).digest()).decode()
        return f"{body}.{sig}"

    def verify_token(self, token: str) -> User:
        try:
            body, sig = token.split(".")
            good = base64.urlsafe_b64encode(hmac.new(self.key, body.encode(), hashlib.sha256).digest()).decode()
            if not hmac.compare_digest(sig, good):
                raise AuthError("bad signature")
            data = json.loads(base64.urlsafe_b64decode(body))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AuthError("malformed token") from exc
        if data["exp"] < self.clock():
            raise AuthError("token expired")
        row = self.db.one("SELECT username, role, disabled FROM users WHERE username=?", (data["u"],))
        if row is None or row["disabled"]:
            raise AuthError("user disabled or removed")
        return User(row["username"], row["role"])  # role re-read from the DB so demotions take effect immediately

    # --- audit ---------------------------------------------------------------------------------------------
    def log(self, username: str, action: str, detail: str) -> None:
        self.db.execute("INSERT INTO audit (ts, username, action, detail) VALUES (?,?,?,?)", (self.clock(), username, action, detail[:500]))

    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.query("SELECT ts, username, action, detail FROM audit ORDER BY ts DESC LIMIT ?", (limit,))


def allowed(user_role: str, needed: str) -> bool:
    return RANK[user_role] >= RANK[needed]
