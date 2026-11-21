"""Unit tests for labelling, deduplication and feature helpers (small synthetic frames - logic only)."""

import numpy as np
import pandas as pd
import pytest

from wtpm import config
from wtpm.data_loader import HASH_KEY_1, HASH_KEY_2, assign_folds, clean_sentinels
from wtpm.features import candidate_features, fit_non_constant, thin_negatives
from wtpm.preprocessing import dedup_turbine, label_file_rows


def make_file(event_id, start, n, values, train_until, label_sources=None):
    ts = pd.date_range(start, periods=n, freq="10min")
    df = pd.DataFrame({"ts": ts, "s_avg": values, "train_test": ["train"] * train_until + ["prediction"] * (n - train_until)})
    raw = df[["s_avg"]]
    df["fp1"] = pd.util.hash_pandas_object(raw, index=False, hash_key=HASH_KEY_1).to_numpy()
    df["fp2"] = pd.util.hash_pandas_object(raw, index=False, hash_key=HASH_KEY_2).to_numpy()
    df["event_id"] = event_id
    df["label_source"] = label_sources if label_sources is not None else "assumed_normal"
    return df


def test_label_file_rows_anomaly():
    df = make_file(1, "2023-01-01", 10, np.arange(10.0), train_until=4)
    ev = pd.Series({"event_label": "anomaly", "event_start": df.ts[6], "event_end": df.ts[7]})
    src = label_file_rows(df, ev).tolist()
    assert src[:4] == ["assumed_normal"] * 4             # train part
    assert src[4:6] == ["excluded_pre_event"] * 2       # prediction, before start
    assert src[6:8] == ["anomaly_window"] * 2
    assert src[8:] == ["excluded_post_event"] * 2       # after failure


def test_label_file_rows_normal():
    df = make_file(1, "2023-01-01", 6, np.arange(6.0), train_until=2)
    ev = pd.Series({"event_label": "normal", "event_start": df.ts[3], "event_end": df.ts[4]})
    assert label_file_rows(df, ev).tolist() == ["assumed_normal"] * 3 + ["normal_window"] * 2 + ["assumed_normal"]


def test_dedup_shifted_copy_and_relabel():
    rng = np.random.default_rng(0)
    vals = rng.normal(size=300)
    # file A: anomaly window on rows 250-299 (own timeline)
    src_a = ["assumed_normal"] * 250 + ["anomaly_window"] * 50
    a = make_file(10, "2023-01-01", 300, vals, 200, src_a)
    # file B: SAME physical records, timestamps shifted by +1 day, all marked assumed_normal
    b = make_file(11, "2023-01-02", 300, vals, 300)
    rec, diag = dedup_turbine([a, b])
    assert diag["unique_records"] == 300 and diag["duplicates_removed"] == 300
    assert (rec["label_source"] == "anomaly_window").sum() == 50  # hidden fault correctly kept as anomaly
    assert diag["file_pairs"][0]["dominant_offset"] == str(pd.Timedelta("1D"))  # ts_new - ts_prev


def test_dedup_does_not_merge_coincidental_repeats():
    # constant readings (stopped turbine) repeat inside and across files at inconsistent times
    a = make_file(1, "2023-01-01", 200, np.zeros(200), 200)
    b = make_file(2, "2023-03-01", 200, np.zeros(200), 200)
    rec, diag = dedup_turbine([a, b])
    assert diag["unique_records"] == 400  # identical values are NOT assumed to be the same record
    assert diag["within_file_repeated_fingerprints_kept"] == 398


def test_assign_folds_groups_turbines():
    ev = pd.DataFrame({"asset_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                       "event_label": ["anomaly", "normal"] * 5})
    f = assign_folds(ev, n_folds=5, seed=0)
    assert f["asset_id"].is_unique and set(f["fold"]) == set(range(5))


def test_clean_sentinels():
    df = pd.DataFrame({"wind_speed_235_avg": [5.0, -14.9], "wind_speed_235_std": [0.5, 0.0], "sensor_1_avg": [-3.0, -3.0]})
    out, n = clean_sentinels(df)
    assert np.isnan(out.loc[1, "wind_speed_235_avg"]) and np.isnan(out.loc[1, "wind_speed_235_std"])
    assert out["sensor_1_avg"].tolist() == [-3.0, -3.0]  # legitimate negatives untouched
    assert n == 2


def test_features_exclude_identifiers_and_status():
    cols = ["asset_id", "status_type_id", "ts", "label", "a_avg", "a_std", "b_avg"]
    assert candidate_features(cols) == ["a_avg", "b_avg"]
    assert candidate_features(cols, ("_avg", "_std")) == ["a_avg", "a_std", "b_avg"]


def test_fit_non_constant_and_thinning():
    X = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [np.nan] * 3})
    assert fit_non_constant(X) == ["b"]
    df = pd.DataFrame({"ts": pd.date_range("2023-01-01", periods=12, freq="10min"), "label": [0] * 11 + [1]})
    kept = thin_negatives(df)
    assert kept["label"].sum() == 1 and (kept["label"] == 0).sum() == 2  # minutes 00 at 00:00 and 01:00


def test_ambient_delta_removes_ambient_level():
    from wtpm.features import ambient_delta
    X = pd.DataFrame({"sensor_7_avg": [5.0, 25.0], "sensor_41_avg": [4.0, 24.0], "sensor_186_avg": [45.0, 65.0],
                      "power_2_avg": [100.0, 100.0]})
    out = ambient_delta(X, ["sensor_7_avg", "sensor_41_avg", "sensor_186_avg"])
    assert out["sensor_186_avg"].tolist() == [40.0, 40.0]       # same gearbox oil rise in winter and summer
    assert "sensor_7_avg" not in out and "sensor_41_avg" not in out
    assert out["power_2_avg"].tolist() == [100.0, 100.0]         # non-temperature untouched


def test_turbine_baselines_use_only_own_reference_period():
    from wtpm.features import apply_turbine_baselines, fit_turbine_baselines
    df = pd.DataFrame({"asset_id": [1, 1, 1, 1, 2, 2, 2, 2],
                       "x": [10.0, 12.0, 14.0, 99.0, 100.0, 102.0, 104.0, -50.0],
                       "train_test": ["train", "train", "train", "prediction"] * 2})
    bl = fit_turbine_baselines(df, ["x"], df["train_test"].eq("train"))
    assert bl.set_index("asset_id").loc[1, "median"] == 12.0   # prediction value 99 not used
    assert bl.set_index("asset_id").loc[2, "median"] == 102.0  # other turbine not mixed in
    out = apply_turbine_baselines(df, ["x"], bl)
    assert out["x"].iloc[1] == 0.0 and out["x"].iloc[5] == 0.0  # same relative level on both turbines
    with pytest.raises(KeyError):
        apply_turbine_baselines(df.assign(asset_id=3), ["x"], bl)
