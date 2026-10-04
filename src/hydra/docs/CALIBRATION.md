# Calibration guide

1. Record an experiment in the ELN (sensor CSV, setup, conditions); mark a split (train / validation / blind).
2. Run bounded least squares, then profile likelihood to see which parameters the data actually constrain.
3. Sample the posterior (ensemble sampler or NUTS) and inspect the posterior predictive.
4. Lock blind predictions (SHA-256) *before* looking at the blind data; report reliability and extrapolation error.
5. Never claim accuracy that was not measured on held-out data.
