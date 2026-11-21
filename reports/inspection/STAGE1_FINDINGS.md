# Stage 1 — Dataset Inspection Findings (CARE to Compare v6)

All figures below come from `wtpm/inspect_dataset.py` (run 2026-11-21, exit code 0, 95/95 files)
and from the auto-generated `inspection_report.md`, `inspection_stats.json`, `per_dataset_summary.csv`
and `column_stats_farm_*.csv` in this folder. The original dataset was verified unchanged after the run
(size + modification time of all 103 files identical before/after).

## 1. Structure (verified)

- Three folders `Wind Farm A/B/C`, each with `datasets/<event_id>.csv`, `event_info.csv`, `feature_description.csv`.
- CSV delimiter `;`, CRLF line endings, timestamps `YYYY-MM-DD HH:MM:SS` at 10-minute intervals.
- Each dataset file = one turbine, ~1 year of data, one event, with a `train_test` column
  (`train` rows always strictly before `prediction` rows, one boundary per file).
- Metadata columns in every file: `time_stamp, asset_id, id, train_test, status_type_id`
  (the dataset README calls the last one `status_type`; the files use `status_type_id`).
- Farm A `event_info.csv` names the turbine column `asset`; B and C use `asset_id`.
- Farm A: 8 columns (`sensor_44`–`sensor_51`) lack the `_avg` suffix that `feature_description.csv` implies.

## 2. Farm comparison

| | Farm A | Farm B | Farm C |
|---|---|---|---|
| Dataset files | 22 | 15 | 58 |
| Anomaly / normal events | 12 / 10 | 6 / 9 | 27 / 31 |
| Turbines | 5 | 9 | 22 |
| Turbines with ≥1 anomaly event | 5 | 6 | 19 |
| Columns per file (sensor cols) | 86 (81) | 257 (252) | 957 (952) |
| Rows in files | 1,196,747 | 859,065 | 3,187,136 |
| Unique (turbine, timestamp) | 425,205 (35.5%) | 560,070 (65.2%) | 1,574,461 (49.4%) |
| Rows inside anomaly windows | 18,240 (1.52%) | 33,989 (3.96%) | 60,897 (1.91%) |
| Anomaly window length (days, median [min–max]) | 7.0 [7.0–31.5] | 36.5 [20–61] | 15.0 [1.7–65.5] |
| Status codes present | 0, 3, 4 | 0, 1, 4, 5 | 0, 3, 4, 5 |
| Abnormal status (≠0,2) inside anomaly windows, median | **100%** | 5.0% | 15.9% |
| Abnormal status inside normal windows, median | 0% | 2.9% | 1.1% |
| Missing sensor cells (median / max per file) | 0% / 0% | 0% / 0% | 0% / 0.33% |
| Timestamp gaps (missing 10-min slots) | 9,445 | 1,463 | 20,016 |
| Duplicate timestamps within a file | 0 | 0 | 0 |
| Constant-in-every-file columns | 2 | 18 | 1 |
| Negative values in m/s or rpm columns | none | yes (e.g. −15 sentinel) | yes (e.g. −14.9 sentinel) |
| Disk size (approx.) | 0.8 GB | 1.4 GB | 17.8 GB |

Event counts observed: 45 anomaly / 50 normal in total. The dataset README states 44 / 51; the files
contain one more `anomaly` label than documented. We use the labels as they appear in `event_info.csv`.

## 3. Event structure and existing labels

- `event_info.csv` gives, per dataset, `event_label` (`anomaly` / `normal`), `event_start`, `event_end`
  and the matching row `id`s (verified: start/end ids match the rows with those timestamps in 95/95 files).
- Every event window lies entirely in the `prediction` part of its own file (0 files have window rows in `train`).
- `anomaly` = expert-labelled period *leading up to* a documented failure (see `event_description`,
  e.g. Farm C event 49: "Failure 2023-04-05 03:30" with `event_end` = 2023-04-05 03:30).
  `normal` = a verified fault-free period of the same length type.
- `status_type_id` is a per-row operating state (0 normal, 1 derated, 2 idling, 3 service, 4 downtime, 5 other).
- There is **no row-level fault label**; any row label must be derived from the event windows.

## 4. Leakage and overlap risks (most important first)

1. **The same physical records are repeated across files of the same turbine.** Only 35–65% of rows are unique.
   In some file pairs the repeated records are **shifted by exactly one day** (anonymisation per file),
   e.g. Farm A datasets 0 vs 71: 23,640 identical sensor rows at offset +1 day. Deduplication by timestamp alone
   therefore fails. → *Any row-wise or file-wise split leaks. Splits must be by turbine (`asset_id`).*
2. **Other anomaly events sit inside a file's `train` part** (A: 16 cases, B: 2, C: 26). The per-file
   `train` part is *not* guaranteed fault-free. → *Labels must be assigned across all files of a turbine
   (matched by record content, not by timestamp).*
3. **`status_type_id` encodes the outcome.** In Farm A the anomaly windows are 100% downtime/service status,
   so status ≈ label. → *Exclude `status_type_id` from model features* (keep it for analysis only).
4. **Identifiers / bookkeeping columns** (`id`, `asset_id`, `time_stamp`, `train_test`) must not be features.
5. **Autocorrelation**: consecutive 10-minute rows are near-identical; random row splits would inflate metrics.
6. **Post-failure period**: rows after `event_end` in anomaly files are repair/restart periods of unknown state.

## 5. Prediction objectives compared

| Objective | Supported by data? | Main problems |
|---|---|---|
| **1. Anomaly-state (pre-failure) classification**: is this record inside an expert-labelled anomaly window that leads up to a failure? | **Yes** — labels come directly from `event_info.csv`. | It is detection of a developing fault, not forecasting of an unseen future. Few independent events. |
| **2. Early-warning (event starts within next H hours)** | Weakly. | `event_start` is the expert-chosen *onset* of visible anomaly; predicting before it is, by definition, predicting before evidence exists. Only 0–432 prediction rows precede each event. Positives would be a handful of short windows. |
| **3. Status forecasting (`status_type_id` at t+H)** | Technically yes. | Status is an operational controller state (service visits, curtailment) rather than a fault label; in A it only takes 0/3/4. Mostly predicts persistence. Not the failure question the paper targets. |

## 6. Recommendation

- **Farm C.** Most independent anomaly events (27 on 19 turbines, 22 turbines total), which is the only farm where
  turbine-grouped validation has enough events per fold. Farm A's labels coincide with downtime status (trivial/leaky);
  Farm B has only 6 anomaly events, one per turbine.
- **Objective 1, framed honestly as "pre-failure anomaly-state detection"**: classify each 10-minute record as
  *pre-failure anomalous* vs *normal* from sensor values only, and evaluate additionally at event level
  (does the model raise an alarm inside each held-out anomaly window, and how long before `event_end` = failure).
  The event-level lead time is the defensible "predictive maintenance" result; it is not future-horizon forecasting.
- **Validation**: GroupKFold by turbine; label assignment by record content across all files of a turbine;
  `status_type_id` excluded; post-failure rows excluded; fit everything inside training folds only.
