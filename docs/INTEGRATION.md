# Integration Guide (for the Monitoring Agent / Spark pipeline)

The ML component is a **Python library** (`wtpm`). It has no web server, message broker or database
dependency. The integration team calls one function from their Python consumer (e.g. a PySpark
`foreachBatch` / `mapInPandas` step, or a plain Kafka consumer).

> **What the model answers:** "Does this 10-minute SCADA record look like the turbine is inside a
> pre-failure anomaly period?" It classifies the *current state*. It does **not** forecast a future failure.

## 1. Install

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Required artifacts (in `artifacts/`): `model_avg.joblib`, `model_metadata_avg.json`.

## 2. Call

```python
from wtpm.predict import predict_turbine_status, AlarmTracker, InputValidationError

results = predict_turbine_status(record_or_records)   # dict | list[dict] | pandas.DataFrame
```

## 3. Input schema

| Field | Type | Required | Notes |
|---|---|---|---|
| `asset_id` | int | recommended | echoed back; required only for the `avg_selfref` model |
| `time_stamp` | str `YYYY-MM-DD HH:MM:SS` | recommended | echoed back; needed for `AlarmTracker` |
| 238 sensor fields `sensor_*_avg`, `power_*_avg`, `wind_speed_*_avg`, … | float or null | **yes (all keys)** | exact names and order in `model_metadata_avg.json → "features"`; Farm C 10-minute averages |

- Column order in the input does not matter; the module reorders to the training order.
- `null`/NaN values are allowed (the Random Forest handles them); a record with >20 % nulls is flagged
  `high_missingness: true`.
- Negative wind speeds (`wind_speed_235/236/237_avg`, sensor-failure code −14.9) are converted to missing,
  exactly as in training.
- Missing feature keys, non-numeric values or infinities raise `InputValidationError` (nothing is silently guessed).
- Extra fields (e.g. `status_type_id`, `id`) are ignored and listed in `validation.batch_notes.ignored_fields`.
  `status_type_id` is deliberately **not** a model input (it would leak the outcome).

Examples: `examples/sample_request.json`, `examples/sample_batch.csv` (real records from held-out test turbines).

## 4. Output schema (one object per input record)

Format only (values below are placeholders, not results):

```json
{
  "asset_id": 0,
  "time_stamp": "YYYY-MM-DD HH:MM:SS",
  "prediction": "anomaly | normal",
  "label": 1,
  "anomaly_probability": 0.0,
  "threshold": 0.0,
  "model_version": "rf-farmC-avg-v1",
  "validation": {"null_features": 0, "high_missingness": false}
}
```
`threshold` was chosen on cross-validation predictions (max F1); `label = anomaly_probability >= threshold`.
See `examples/sample_response.json` for a real response.

## 5. From record predictions to alarms

Single records are noisy. The evaluation uses an **alarm = 6 consecutive "anomaly" records that are
contiguous in time (one continuous hour)**. The same rule is available for streaming:

```python
tracker = AlarmTracker()                     # keeps state per asset_id
for r in results:
    state = tracker.update(r["asset_id"], r["time_stamp"], r["label"])
    if state["alarm"]:
        ...  # publish alarm to dashboard
```
A missing 10-minute record or an out-of-order timestamp resets the run for that turbine.

## 6. Spark sketch (illustrative, not part of Member 1 deliverable)

```python
def score(pdf_iter):
    from wtpm.predict import predict_turbine_status
    import pandas as pd
    for pdf in pdf_iter:
        yield pd.DataFrame(predict_turbine_status(pdf))[["asset_id", "time_stamp", "prediction",
                                                          "anomaly_probability", "model_version"]]
# df.mapInPandas(score, schema="asset_id long, time_stamp string, prediction string, "
#                              "anomaly_probability double, model_version string")
```
The model is a scikit-learn object: it runs in **PySpark (Python workers)**, not in Scala/Java Spark MLlib.

## 7. Responsibilities

| Member 1 (this repo) | Integration team |
|---|---|
| model, preprocessing rules, feature schema, `predict_turbine_status`, `AlarmTracker`, tests, docs | Kafka topics, Spark job, mapping raw SCADA messages to the input schema, dashboard, optional REST API |
