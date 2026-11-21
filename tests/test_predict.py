"""Tests for the prediction interface. Uses the real saved artifact; skipped if not trained yet."""

import math

import numpy as np
import pandas as pd
import pytest

from wtpm import config
from wtpm.predict import AlarmTracker, InputValidationError, load_model, predict_turbine_status

ART = config.ARTIFACTS_DIR / "model_avg.joblib"
needs_model = pytest.mark.skipif(not ART.exists(), reason="model artifact not trained yet")


@pytest.fixture(scope="module")
def meta():
    return load_model("avg")[1]


def record(meta, value=1.0):
    return {"asset_id": 99, "time_stamp": "2023-01-01 00:00:00", **{f: value for f in meta["features"]}}


@needs_model
def test_model_loads_and_feature_order_matches(meta):
    model, m = load_model("avg")
    assert list(model.feature_names_in_) == m["features"]
    assert 0 < m["threshold"] < 1 and "status_type_id" not in m["features"]


@needs_model
def test_single_record_output_schema(meta):
    out = predict_turbine_status(record(meta))
    assert len(out) == 1
    r = out[0]
    assert set(r) >= {"asset_id", "time_stamp", "prediction", "label", "anomaly_probability", "threshold",
                      "model_version", "validation"}
    assert r["prediction"] in {"anomaly", "normal"} and r["label"] in {0, 1}
    assert 0.0 <= r["anomaly_probability"] <= 1.0 and r["asset_id"] == 99


@needs_model
def test_column_order_does_not_matter(meta):
    rec = record(meta)
    rec[meta["features"][0]] = 50.0
    df = pd.DataFrame([rec])
    shuffled = df[list(reversed(df.columns))]
    a = predict_turbine_status(df)[0]["anomaly_probability"]
    b = predict_turbine_status(shuffled)[0]["anomaly_probability"]
    assert a == b


@needs_model
def test_missing_feature_raises(meta):
    rec = record(meta)
    del rec[meta["features"][3]]
    with pytest.raises(InputValidationError, match="missing"):
        predict_turbine_status(rec)


@needs_model
def test_non_numeric_raises(meta):
    rec = record(meta)
    rec[meta["features"][0]] = "abc"
    with pytest.raises(InputValidationError, match="Non-numeric"):
        predict_turbine_status(rec)


@needs_model
def test_null_values_allowed_and_reported(meta):
    rec = record(meta)
    for f in meta["features"][:100]:
        rec[f] = None
    r = predict_turbine_status(rec)[0]
    assert r["validation"]["null_features"] == 100 and r["validation"]["high_missingness"] is True


@needs_model
def test_sentinel_wind_speed_is_cleaned(meta):
    ws = [f for f in meta["features"] if f.startswith("wind_speed_235")]
    if not ws:
        pytest.skip("wind speed feature not in model")
    rec = record(meta)
    rec[ws[0]] = -14.9
    r = predict_turbine_status(rec)[0]
    assert r["validation"]["batch_notes"]["sentinel_values_set_missing"] == 1


@needs_model
def test_bad_input_type():
    with pytest.raises(InputValidationError):
        predict_turbine_status("not a record")


@needs_model
def test_predicts_real_unseen_test_records(meta):
    """Records from held-out TEST turbines (never used in training)."""
    p = config.PROCESSED_DIR / "event_windows.pkl"
    if not p.exists():
        pytest.skip("processed data not available")
    win = pd.read_pickle(p)
    t = win[win["fold"] == config.TEST_FOLD].groupby("event_id").head(2)
    assert not set(t["asset_id"]) & set(meta["train_turbines"])
    out = predict_turbine_status(t)
    assert len(out) == len(t) and all(not math.isnan(r["anomaly_probability"]) for r in out)


def test_alarm_tracker_continuity():
    tr = AlarmTracker(k=3, freq_minutes=10)
    ts = pd.date_range("2023-01-01", periods=3, freq="10min")
    assert [tr.update(1, t, 1)["alarm"] for t in ts] == [False, False, True]
    tr2 = AlarmTracker(k=3, freq_minutes=10)
    tr2.update(1, "2023-01-01 00:00", 1)
    tr2.update(1, "2023-01-01 00:10", 1)
    assert tr2.update(1, "2023-01-01 00:40", 1)["alarm"] is False  # gap resets the run
    assert tr2.update(2, "2023-01-01 00:40", 0)["consecutive_anomalies"] == 0  # turbines independent
