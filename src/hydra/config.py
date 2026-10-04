"""Runtime configuration. Laptop mode is the default and is loopback-only, no auth."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def default_data_dir() -> Path:
    return Path(os.environ.get("HYDRA_DATA_DIR", Path.home() / ".hydra"))


class Settings(BaseModel):
    mode: Literal["laptop", "server"] = "laptop"
    host: str = "127.0.0.1"
    port: int = Field(8765, ge=1, le=65535)
    data_dir: Path = Field(default_factory=default_data_dir)
    database_url: str | None = None  # default: SQLite under data_dir
    enforce_offline: bool = True
    telemetry: Literal[False] = False  # there is no telemetry; field exists to be explicit
    max_job_cpus: int = Field(1, ge=1)
    max_job_ram_mb: int = Field(2048, ge=64)

    @model_validator(mode="after")
    def _laptop_is_loopback_only(self) -> Settings:
        if self.mode == "laptop" and self.host not in LOOPBACK:
            raise ValueError("laptop mode binds to loopback only; use server mode for LAN access")
        return self

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{self.data_dir / 'hydra.sqlite3'}"

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.environ.get("HYDRA_MODE", "laptop")
        host = os.environ.get("HYDRA_HOST", "127.0.0.1" if mode == "laptop" else "0.0.0.0")  # noqa: S104
        return cls(
            mode=mode,
            host=host,
            port=int(os.environ.get("HYDRA_PORT", "8765")),
            enforce_offline=os.environ.get("HYDRA_ENFORCE_OFFLINE", "1") != "0",
        )
