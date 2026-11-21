# Artifacts

| File | Purpose |
|---|---|
| `model_avg.joblib` | **Final model** (`rf-farmC-avg-v1`): Random Forest on 238 Farm C `_avg` features |
| `model_metadata_avg.json` | Feature list **and order**, threshold (0.1775), parameters, training turbines, library versions |
| `model_avg_selfref.joblib` | Secondary model (`rf-farmC-avg_selfref-v1`): ambient-corrected + per-turbine baseline features |
| `model_metadata_avg_selfref.json` | As above, plus transform definition and raw input feature list |
| `turbine_baselines_avg_selfref.csv` | Per-turbine median/IQR (from each turbine's own `train` period) required by the secondary model |

Created by `python -m wtpm.train`. Load through `wtpm.predict.load_model()` / `predict_turbine_status()`.
Requires scikit-learn 1.9.1 (pinned in `requirements.txt`); joblib models are not portable across sklearn versions.
