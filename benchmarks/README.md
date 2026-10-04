# Literature benchmark suite

Put published experimental datasets here as JSON files in `benchmarks/datasets/` (one file per curve set).
**No data ships with HYDRA** - nothing in this folder has been validated until *you* digitise and enter it, and
nothing here may be presented as literature data unless it carries a real citation.

Schema (`benchmarks/schema.json`):

```json
{
  "id": "smith2019_fig3a",
  "citation": "Author et al., Journal, Year, Fig. 3a (DOI ...)",
  "conditions": {"al_mass_g": 0.5, "form": "powder", "dim_um": 50, "c_naoh_M": 2.0, "v_liq_mL": 100,
                 "T0_C": 30, "T_amb_C": 25, "v_vessel_mL": 250, "alloy": "pure"},
  "t_s": [0, 60, 120],
  "h2_mol": [0.0, 0.001, 0.003],
  "digitization_sd_mol": 2e-4,
  "measurement": {"method": "water_displacement", "note": ""}
}
```

* `hydra.benchmarks.digitize` imports points from a plot image (axis calibration + colour picker).
* `hydra.benchmarks.leaderboard` runs every model level (reduced L1, full L1, L2, electrochemical) over every
  dataset: RMSE, max error, bias by condition, coverage of the 95 % band.
* `benchmarks/baseline.json` stores the accepted accuracy; a commit that makes it worse fails
  `tests/test_benchmarks.py` (use `hydra-benchmarks --update-baseline` after an intentional change).
