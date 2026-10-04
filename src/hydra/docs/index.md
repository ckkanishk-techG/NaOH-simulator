# HYDRA: Al–NaOH–H₂O digital twin

HYDRA is a **local-only**, offline research model of on-demand hydrogen from aluminium + sodium hydroxide + water, coupled to a
platinum-free AEM fuel cell (NiMo anode, Ag cathode, nickel-foam electrodes). It is a predictive research model,
**not a safety certification**.

* [Architecture](/docs/ARCHITECTURE) - packages, fidelity levels, data flow
* [Assumptions](/docs/ASSUMPTIONS) - every constant: unit, default, plausible range, status, source placeholder
* [Validation and verification](/docs/VALIDATION) - what is verified, what is validated, what is only assumed
* [Calibration guide](/docs/CALIBRATION) - from lab notebook to calibrated, honestly uncertain parameters
* [Hardware guide](/docs/HARDWARE) - sensors, firmware, live twin
* [Security notes](/docs/SECURITY) - server mode hardening
* [Limitations](/docs/LIMITATIONS)
* [Build stages](/docs/BUILD_STAGES)

Nothing here makes a network call: no CDN, telemetry, remote fonts, or third-party accounts.
