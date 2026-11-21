# Wind Turbine Predictive Maintenance — ML Module (Member 1)

Member 1 component of the MSc Big Data capstone *"Modernization of a Big Data Framework for Real-Time
Predictive Maintenance of Wind Turbines"*, following the Predictive Model Generator of Canizo et al. (2017):
historical SCADA data → preprocessing → Random Forest → saved model → prediction interface for the
streaming pipeline (Kafka / Spark, owned by the integration team).

**Status:** complete pipeline, trained and evaluated models, reusable prediction function, 25 passing tests.
All numbers below come from actual runs (`reports/`); nothing was tuned on the test set.

---

## 1. Problem and objective

**Objective — pre-failure anomaly-state classification.** Given one 10-minute SCADA record of a Farm C turbine
(sensor values only), decide whether the turbine is inside an expert-labelled anomaly window that leads up to a
documented failure.

- This **detects a developing fault from the current state. It does not forecast a future failure.**
- The *maintenance value* is reported as **event-level lead time**: hours between the first alarm
  (6 consecutive positive records forming one continuous hour) and `event_end` (documented failure/stop).
- Early-warning prediction before `event_start` was rejected: `event_start` is the expert-defined onset of visible
  anomaly, so no labelled evidence exists before it (see `docs/METHODOLOGY.md §1`).

## 2. Dataset

**CARE to Compare v6** (95 event datasets, 3 wind farms), kept **read-only** in `../CARE_To_Compare` and never
committed. Stage 1 inspection of all 95 files (`reports/inspection/STAGE1_FINDINGS.md`) selected **Wind Farm C**:
22 turbines, 27 anomaly + 31 normal events, 952 sensor columns, 10-minute data. Farm A's labels coincide with
downtime status; Farm B has only 6 anomaly events.

Key data issues found and handled:

| Issue (measured) | Handling |
|---|---|
| Same physical records repeated across a turbine's files, sometimes with timestamps shifted by ±365 days / +1 day | Fingerprint deduplication with offset-consistent matching: 3,187,136 rows → 1,807,406 unique records |
| Anomaly windows of other events hidden inside files' `train` parts | Conservative label merge: 14,012 records relabelled `assumed_normal → anomaly_window` |
| `status_type_id` reveals the outcome | Never a feature |
| Wind-speed failure code −14.9 m/s | Set to missing (964,322 cells) |
| ~2 % positives, autocorrelated rows, turbine effects | Turbine-grouped CV, PR-AUC, event-level metrics |

## 3. Method (details: `docs/METHODOLOGY.md`)

- **Labels:** 1 = inside anomaly window (60,897 records); 0 = expert normal window (59,045) or *assumed* normal
  (1,669,905 — not expert-confirmed); excluded = before `event_start` / after `event_end` in anomaly files (16,704).
- **Features:** the 238 `_avg` sensor columns (`final` model). Secondary variant `avg_selfref`: temperatures minus
  ambient, then per-turbine `(x − median)/IQR` from the turbine's own historical `train` period (235 features).
- **Validation:** 5 turbine-grouped folds (`StratifiedGroupKFold`, seed 2). Fold 0 = **test**, used once.
  Folds 1–4 = 4-fold CV for model, class-weight and threshold selection. 0 fingerprints shared between test and
  train/validation; every fold has both classes.

| Fold | Role | Turbines | Anomaly events | Normal events | Positive / negative records |
|---|---|---|---|---|---|
| 0 | **test** | 13, 23, 33, 34, 44 | 5 | 7 | 16,248 / 443,043 |
| 1 | CV | 14, 21, 32, 35, 56 | 5 | 6 | 12,354 / 338,058 |
| 2 | CV | 15, 38, 42, 50, 55 | 5 | 6 | 9,068 / 394,838 |
| 3 | CV | 2, 5, 43, 53 | 6 | 7 | 12,622 / 280,967 |
| 4 | CV | 12, 16, 52 | 6 | 5 | 10,605 / 272,044 |

- **Model:** `RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
  class_weight="balanced", random_state=42)`, selected from 5 RF configurations + 2 baselines by mean CV PR-AUC
  (rule fixed before testing). Negatives thinned to 1/hour in training folds only. Threshold 0.1775 = max F1 on
  pooled out-of-fold CV predictions.

## 4. Results

### 4.1 Cross-validation (folds 1–4: 17 turbines, 22 anomaly / 24 normal events)

| Model | PR-AUC mean ± sd | ROC-AUC | Events detected | Normal events with false alarm |
|---|---|---|---|---|
| Majority class | 0.035 ± 0.008 | 0.50 | 0 / 22 | 0 / 24 |
| Logistic regression (balanced) | 0.086 ± 0.083 | 0.59 | 11 / 22 | 6 / 24 |
| **RF `_avg` (final)** | **0.127 ± 0.146** | 0.56 | 16 / 22 | 4 / 24 |
| RF `avg_selfref` (secondary) | 0.075 ± 0.014 | 0.70 | 20 / 22 | 14 / 24 |

Per-fold PR-AUC of the final RF: 0.030, 0.022, 0.078, 0.377 — strongly turbine-dependent.
Full tables: `reports/metrics/cv_summary_*.csv`, `cv_results_*.json`.

### 4.2 Held-out test turbines (fold 0: 5 turbines, 5 anomaly / 7 normal events; prevalence 3.54 %)

| Metric | **RF `_avg` (final)** | RF `avg_selfref` (secondary) |
|---|---|---|
| PR-AUC (no-skill = 0.035) | **0.043** | 0.061 |
| ROC-AUC | 0.593 | 0.708 |
| Precision / Recall / F1 (anomaly class) | 0.042 / 0.022 / 0.029 | 0.061 / 0.167 / 0.090 |
| Accuracy (all-normal would score 0.965) | 0.947 | 0.880 |
| Confusion matrix (TN, FP, FN, TP) | 434,795 · 8,248 · 15,886 · 362 | 401,624 · 41,419 · 13,541 · 2,707 |
| Anomaly events detected | **1 / 5** (4 missed) | 4 / 5 (1 missed) |
| Normal events with false alarm | 1 / 7 | 5 / 7 |
| Median detection delay / lead time before `event_end` | 619 h / 953 h (one event) | 8.5 h / 461 h |

Figures (`reports/figures/`): `confusion_matrix_avg.png`, `pr_roc_curves_avg.png`,
`feature_importance_avg.png`, `test_anomaly_events_avg.png` (and `_avg_selfref` versions).
Metrics JSON: `reports/metrics/test_metrics_avg.json`; per-event table: `test_event_level_avg.csv`.

### 4.3 Error analysis (final model, test)

- **False negatives:** recall on anomaly records during *normal operation* (status 0, 93 % of positives) is
  0.01 %; recall is 36–39 % only when the turbine is already in service/downtime (status 3/4). The model mostly
  recognises non-production states, not the fault developing during production.
- **False positives:** 59 % occur in service status (3); false-positive rate 1.9 % on assumed-normal vs 0.2 % on
  expert-normal records — part of the "false" positives may be unrecorded faults in assumed-normal data.
- **Important features:** cooling-water conductivity and pressure (`sensor_225`, `sensor_227`), gearbox and
  hydraulic oil levels (`sensor_94`, `sensor_74`) — slow-moving levels that change with maintenance.

### 4.4 Interpretation

The final model, selected by the pre-specified CV rule, is **only marginally better than chance on unseen
turbines** (test PR-AUC 0.043 vs 0.035). Its high CV mean came mainly from one fold. The self-referenced variant
generalised more consistently (CV ROC-AUC 0.63–0.75 on every fold; test 0.71, 4/5 events detected) but raises
false alarms on 5/7 normal events. Neither is ready for unattended operational alarms. We report the
pre-specified model as final and do not re-select on test results; the secondary result motivates future work.

## 5. Limitations

- Only 27 anomaly events (5 in the test set; one event supplies 58 % of test positives) → metrics are unstable;
  per-event results are more informative than single scores.
- Labels are expert windows, not exact fault onsets; 96.6 % of negatives are *assumed* normal (not expert-confirmed).
- Anomaly-state classification, not failure forecasting; lead times are measured inside expert windows.
- Strong turbine and seasonal effects (anomaly months differ by fold); raw-feature models do not transfer.
- Only `_avg` features and a small RF grid were explored due to time; `_avg + _std` was not compared.
- Farm C only; the model does not apply to Farms A/B (different sensors).

## 6. Project structure

```
wind_turbine_ml/
├── wtpm/                  Python package
│   ├── config.py          all settings (paths, seeds, label rules, folds)
│   ├── inspect_dataset.py Stage 1 read-only inspection
│   ├── data_loader.py     reading files, fingerprints, sentinel cleaning, fold assignment
│   ├── preprocessing.py   labels, deduplication, leakage checks → data/processed/
│   ├── features.py        feature selection, ambient / per-turbine transforms, negative thinning
│   ├── metrics.py         record- and event-level metrics, alarm continuity
│   ├── train.py           baselines + RF grid, turbine-grouped CV, final model
│   ├── evaluate.py        one-time test evaluation, figures, error analysis
│   ├── predict.py         predict_turbine_status(), input validation, AlarmTracker
│   └── make_examples.py   sample request/response from real test-turbine records
├── tests/                 pytest (25 tests)
├── artifacts/             model_avg.joblib (final), model_avg_selfref.joblib, metadata, turbine baselines
├── reports/               inspection/, preprocessing/, metrics/, figures/
├── examples/              sample_request.json, sample_response.json, sample_batch.csv
├── docs/                  METHODOLOGY.md, INTEGRATION.md
└── data/processed/        generated (not committed)
```

## 7. How to run (Windows, Python 3.14)

```powershell
cd wind_turbine_ml
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# the dataset must be at ..\CARE_To_Compare (read-only)

.\.venv\Scripts\python.exe -m wtpm.inspect_dataset --data-dir ..\CARE_To_Compare --out reports\inspection  # ~20 min
.\.venv\Scripts\python.exe -m wtpm.preprocessing                       # ~10 min, writes data\processed
.\.venv\Scripts\python.exe -m wtpm.train --feature-set avg             # ~30 min (CV grid + final model)
.\.venv\Scripts\python.exe -m wtpm.train --feature-set avg_selfref --models dummy_majority rf_leaf5_sqrt_bal rf_leaf5_sqrt_none
.\.venv\Scripts\python.exe -m wtpm.evaluate --feature-set avg
.\.venv\Scripts\python.exe -m wtpm.evaluate --feature-set avg_selfref --perm-sample 0
.\.venv\Scripts\python.exe -m wtpm.make_examples
.\.venv\Scripts\python.exe -m pytest -q
```

## 8. Prediction usage

```python
from wtpm.predict import predict_turbine_status, AlarmTracker
result = predict_turbine_status({"asset_id": 34, "time_stamp": "2023-08-19 09:40:00", "sensor_0_avg": 88.5, ...})
# -> [{"prediction": "normal", "label": 0, "anomaly_probability": 0.105, "threshold": 0.1775,
#      "model_version": "rf-farmC-avg-v1", "validation": {...}, ...}]
```
Input/output schema, alarm logic and Spark notes: **`docs/INTEGRATION.md`**. Real example:
`examples/sample_request.json` → `examples/sample_response.json`.

## 9. Reproducibility

Seeds fixed (`RANDOM_STATE=42`, `FOLD_SEED=2`); deterministic negative thinning; settings saved in
`data/processed/preprocessing_config.json`, fold assignment in `reports/metrics/fold_assignment.csv`, model
parameters, feature order, threshold and library versions in `artifacts/model_metadata_*.json`.
Dependencies pinned in `requirements.txt` (no MLflow, no FastAPI).
