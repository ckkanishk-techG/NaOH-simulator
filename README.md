# HYDRA – Al–NaOH–H₂O digital twin

Local-only, offline research model of on-demand H₂ from the Al–NaOH–H₂O reaction
(Al + OH⁻ + 3H₂O → Al(OH)₄⁻ + 3/2 H₂) feeding a Pt-free AEM fuel cell.
**Predictive model, not a safety certification.** All kinetic, thermal and fuel-cell constants are
**uncalibrated priors** until you fit them to your own data and check them on held-out data.

## Status
Backend stages 2–16 are implemented (React UI not yet). Priorities: physical correctness → honest
uncertainty → calibration → performance → UI polish.

| Area | Module |
|---|---|
| Thermo, property DB, ranges | `hydra.thermo` |
| Reactor L1 (lumped, events), fast reduced model (NumPy/Numba/JAX) | `hydra.core` |
| L2 size classes, regime map; L3/L4 axisymmetric FV, GCI, OpenFOAM export | `hydra.transport` |
| Electrochemistry, fuel cell, stack, BOP, coupled system | `hydra.electrochem`, `hydra.fuelcell`, `hydra.system` |
| Calibration, Bayes, sensitivity, model selection, validation, benchmarks | `hydra.inference` |
| Optimal experiment design, design optimisation, cost | `hydra.design`, `hydra.optimize` |
| Surrogate (with conformal bounds and physics fallback) | `hydra.surrogate` |
| Control (PID/DMC), live twin (EKF/EnKF), read-only hardware | `hydra.control`, `hydra.hardware` |
| Scale-up, economics, LCA (screening level) | `hydra.scaleup` |
| Platform: DB, auth, jobs, ELN, plots, reports, backups, REST/WebSocket API | `hydra.api` and friends |

Docs are served offline at `/docs` (assumptions table is generated live) and live in `src/hydra/docs/`
(ARCHITECTURE, VALIDATION, CALIBRATION, SECURITY, LIMITATIONS, API, HARDWARE, BUILD_STAGES).

## Install
- Laptop: `./scripts/install.sh` (or `scripts/install.bat`), then `hydra-sim` (binds 127.0.0.1 only).
- Extras: `.[fast]` Numba, `.[jax]`, `.[bayes]`, `.[hardware]`, `.[report]`, `.[server]`, `.[dev]`.
- Server: `docker compose up --build`; create users with `hydra-user add <name> --role researcher`; first
  admin via `POST /api/auth/bootstrap`. Put a TLS proxy in front. Worker: `hydra-worker`.
- Benchmarks: put your own digitised datasets in `benchmarks/datasets/` and run `hydra-benchmarks`.

## Offline guarantees
No CDNs, telemetry or remote fonts. Tests block non-loopback sockets; the CLI enforces the same guard at runtime.

## Known limitations
See `src/hydra/docs/LIMITATIONS.md`. In short: uncalibrated priors, Pitzer valid near 25 °C, no shipped
validation data, bubble blocking/carbonation/ageing/precipitation off by default, surrogate valid below ~90 °C,
scale-up/economics/LCA are screening-level.

## Dev
`pip install -e ".[dev]" && pytest && ruff check . && python -m mypy src`
