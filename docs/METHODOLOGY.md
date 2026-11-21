# Methodology — Member 1 ML Module

Project: *Modernization of a Big Data Framework for Real-Time Predictive Maintenance of Wind Turbines*
(based on Canizo et al., 2017). Dataset: **CARE to Compare v6, Wind Farm C** (read-only).

## 1. Prediction objective (what the model does and does not do)

| Term | Definition | Used here |
|---|---|---|
| **Anomaly-state classification** | Given one 10-minute SCADA record (sensor values only), is the turbine inside an expert-labelled *anomaly window that leads up to a documented failure*? | **Yes — this is the model.** |
| Early-warning prediction | Will an anomaly event *start* within the next H hours? | No — `event_start` is the expert-defined onset of visible anomaly, so no labelled evidence exists before it. |
| **Detection lead time** | For each held-out anomaly event: time from `event_start` to the first alarm (6 consecutive positive records forming one continuous hour) and remaining time from that alarm to `event_end` (documented failure/stop). | **Yes — event-level evaluation measure.** |

The model detects a developing fault from the current state; it is **not** a forecast of future failure.
Lead time before `event_end` quantifies the practical maintenance value of that detection.

## 2. Data and why Farm C

Stage 1 inspection (`reports/inspection/STAGE1_FINDINGS.md`) compared all three farms. Farm C was chosen
because it has the most independent anomaly events (27 on 19 of 22 turbines), the only farm where
turbine-grouped validation has several events per fold. Farm A labels coincide with downtime status
(100% of anomaly-window rows have abnormal status); Farm B has only 6 anomaly events.

## 3. Target construction (`wtpm/preprocessing.py`)

Each dataset file contains one event and roughly a year of data for one turbine. Per file, using that file's own
timestamps and `event_info.csv`:

| label_source | Rule | Label |
|---|---|---|
| `anomaly_window` | inside [`event_start`, `event_end`] of an anomaly event | 1 |
| `normal_window` | inside [`event_start`, `event_end`] of a normal event (expert-confirmed normal) | 0 |
| `assumed_normal` | outside any labelled window (mostly historical `train` part) — **assumed**, not confirmed | 0 |
| `excluded_pre_event` | prediction part before `event_start` of an anomaly file (possible early degradation) | excluded |
| `excluded_post_event` | prediction part after `event_end` of an anomaly file (repair/restart) | excluded |

## 4. Repeated physical records (deduplication)

Stage 1 found that files of the same turbine repeat the same records, sometimes with timestamps shifted by
exactly ±365 days or +1 day (per-file anonymisation), so timestamp-based deduplication fails.

- Every row gets two independent 64-bit fingerprints of **all 952 raw sensor values** (`fp1`, `fp2`);
  disagreement between them would reveal a hash collision (none found).
- For each pair of files of the same turbine, the time offset is estimated **only from fingerprints that are
  unique within both files** (repeated vectors such as frozen readings of a stopped turbine would otherwise
  dominate). A pair is accepted as copies only if ≥100 such matches exist and ≥90% share one offset.
- Only rows matching **at that offset** are merged. Identical values at other times are treated as distinct
  observations (identical sensor values are not assumed to be the same physical record).
- Conflicting labels across copies are resolved conservatively:
  `anomaly_window > excluded_post_event > excluded_pre_event > normal_window > assumed_normal`.
  This recovers anomaly windows hidden inside other files' `train` parts.
- Identical sensor vectors shared between different turbines are removed from the test fold.

## 5. Features

- Candidate set: the 238 `_avg` columns (every sensor has one; min/max/std describe the same signal and
  quadruple memory). `status_type_id`, `asset_id`, `id`, `event_id`, `time_stamp`, `train_test` are never features.
- Cleaning: negative wind speeds (`wind_speed_235/236/237`, failure code −14.9 m/s) → missing. Other
  negatives (°C, A, bar, deg …) are physically plausible and kept. Missing values are left as NaN and handled
  natively by scikit-learn's Random Forest (no imputer to fit → no imputation leakage).
- Columns constant in the training data are dropped (fitted on training folds only).
- No scaling (unnecessary for trees) and no PCA (would mix sensors and hide which component drives a detection).
- Feature-set variants evaluated (see results): raw `_avg`; ambient-corrected (`temperature − ambient`,
  outside-temperature columns dropped); self-referenced (ambient-corrected, then per-turbine
  `(value − median) / IQR` using only that turbine's own historical `train` period, no labels).

## 6. Validation

- `StratifiedGroupKFold` over turbines (5 folds, seed 2), stratified by anomaly events. No turbine appears in
  more than one fold; no fingerprint crosses test and train/validation (asserted).
- Fold 0 = final **test** set, used once. Folds 1–4 = turbine-grouped 4-fold CV for model and threshold selection.
- Class imbalance: in training folds only, negatives are thinned deterministically to one per hour (all
  positives kept) and `class_weight="balanced"` is compared with no weighting. Validation/test data are never resampled.
- Model selection by mean CV PR-AUC; decision threshold = max F1 on pooled out-of-fold CV predictions.
- Baselines: majority class (`DummyClassifier`) and class-balanced logistic regression (median imputation +
  standardisation fitted inside each training fold).

## 7. Metrics

Record level: PR-AUC, ROC-AUC, accuracy, precision, recall, F1, confusion matrix, classification report.
Event level: detected / missed anomaly events, false alarms on normal events, detection delay after
`event_start`, lead time before `event_end`. Alarm = 6 consecutive positive records **that are contiguous in
time** (each step exactly 10 minutes; a timestamp gap resets the run).

## 8. Results

See `README.md §4` (all values generated by `wtpm.train` / `wtpm.evaluate`; raw outputs in `reports/metrics/`).
