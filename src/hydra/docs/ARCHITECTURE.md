# Architecture

`thermo` -> `core` (L1 lumped engine, fast reduced model) -> `transport` (L2 size classes, L3/L4 finite volume) -> `fuelcell` -> `system`.
`inference` (calibration, Bayes, validation), `design` (OED), `optimize`, `surrogate`, `control`, `hardware` (live twin), `scaleup`.
Platform: `db`, `auth`, `jobs`, `runs`, `eln`, `plots`, `export`, `report`, `backup`, and the FastAPI app in `api`.
Property database (`thermo.propdb`) is the single source of constants: unit, default, range, status, source placeholder.
