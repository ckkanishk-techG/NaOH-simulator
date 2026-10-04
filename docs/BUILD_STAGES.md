# Build stages

Status per stage: validated vs assumed. Nothing is claimed accurate until measured on held-out data.

| # | Stage | Status |
|---|-------|--------|
| 1 | Repo, packaging, local-only skeleton, no-network test | **done** |
| 2 | L1 core + energy + gas + safety engine | next |
| 3 | Thermodynamics (Pitzer, solubility, precipitation) | – |
| 4 | Electrochemical mixed-potential model | – |
| 5 | L2 particles, bubbles, mass transfer | – |
| 6 | Fuel cell + stack + system | – |
| 7 | Calibration + sensitivity + model selection | – |
| 8 | FastAPI + React UI + reporting + ELN | – |
| 9–16 | OED, optimization, L3/L4, surrogate, control, live twin, scale-up, server hardening | – |

`prototype/` holds the earlier single-file JS L1 prototype (uncalibrated priors) as a reference only;
the Python L1 core in stage 2 is the source of truth.
