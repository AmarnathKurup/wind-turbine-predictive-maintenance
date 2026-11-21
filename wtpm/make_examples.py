"""Create integration examples from REAL held-out test-turbine records (never seen in training).

Usage::  .\\.venv\\Scripts\\python.exe -m wtpm.make_examples
Writes examples/sample_request.json, examples/sample_response.json and
examples/sample_batch.csv (a few records from one test anomaly window and one normal period).
"""

from __future__ import annotations

import json

import pandas as pd

from wtpm import config
from wtpm.predict import load_model, predict_turbine_status


def main() -> int:
    out = config.PROJECT_ROOT / "examples"
    out.mkdir(exist_ok=True)
    _, meta = load_model("avg")
    win = pd.read_pickle(config.PROCESSED_DIR / "event_windows.pkl")
    test = win[win["fold"] == config.TEST_FOLD].sort_values(["event_id", "ts"])
    anom = test[test["event_label"] == "anomaly"].groupby("event_id").tail(3).head(3)
    norm = test[test["event_label"] == "normal"].groupby("event_id").head(3).head(3)
    batch = pd.concat([anom, norm])
    batch = batch.assign(time_stamp=batch["ts"].dt.strftime(config.TS_FORMAT))[["asset_id", "time_stamp"] + meta["features"]]

    one = batch.iloc[0]
    request = {"asset_id": int(one["asset_id"]), "time_stamp": one["time_stamp"],
               **{c: (None if pd.isna(one[c]) else float(one[c])) for c in meta["features"]}}
    (out / "sample_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    resp = predict_turbine_status(request)
    (out / "sample_response.json").write_text(json.dumps(resp, indent=2), encoding="utf-8")
    batch.to_csv(out / "sample_batch.csv", index=False)
    batch_resp = predict_turbine_status(batch)
    (out / "sample_batch_response.json").write_text(json.dumps(batch_resp, indent=2), encoding="utf-8")
    print(json.dumps(resp, indent=2))
    print("batch:", [(r["asset_id"], r["time_stamp"], r["prediction"], r["anomaly_probability"]) for r in batch_resp])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
