# HYDRA – Al–NaOH–H₂O digital twin

Local-only, offline research model of on-demand H₂ from the Al–NaOH–H₂O reaction feeding a Pt-free AEM
fuel cell. **Predictive model, not a safety certification.** All constants are uncalibrated priors until
fitted to real data (see `docs/BUILD_STAGES.md`).

## Status
Stage 1 of 16 (skeleton): packaging, FastAPI shell, run provenance (SQLite), loopback-only laptop mode,
outbound-network guard. No physics yet.

## Install
- Laptop: `./scripts/install.sh` (or `scripts/install.bat` on Windows), then `hydra-sim`
  (binds 127.0.0.1 only, opens your browser).
- Docker/server: `docker compose up --build` (skeleton; hardening in stage 16).

## Offline guarantees
No CDNs, telemetry, or remote fonts. `tests/` runs with sockets to non-loopback hosts blocked and a test
scans shipped web assets for remote URLs. The CLI also enforces the guard at runtime.

## Dev
`pip install -e ".[dev]" && pytest && ruff check . && mypy`
