# Limitations (read before trusting any number)

- **Every kinetic, thermal and fuel-cell constant is an uncalibrated prior.** Outputs are *not* validated until you calibrate against your own measurements and check them on held-out data (see CALIBRATION).
- Pitzer activities are valid near 25 °C only; range violations are collected and shown with each run.
- The fast reduced model has no pressure, valve or boiling physics; use L1 for those.
- No literature or benchmark data ships with HYDRA. The benchmark loader is empty until you add your own.
- Bubble blocking, carbonation, membrane ageing and gibbsite precipitation are **off by default**.
- The surrogate is trained for one parameter set and configuration and is valid below about 90 °C; outside its domain it falls back to physics.
- Scale-up, economics and LCA are **screening-level**: placeholders for prices and factors, no vehicle validation.
- Safety screens are advisory. HYDRA never commands hardware; the HIL link is read-only.
