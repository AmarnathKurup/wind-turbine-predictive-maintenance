"""Unit tests for record- and event-level metric helpers."""

import numpy as np
import pandas as pd

from wtpm.metrics import best_f1_threshold, event_level, first_alarm, record_metrics


def test_first_alarm_requires_six_contiguous():
    ts = pd.Series(pd.date_range("2023-01-01", periods=10, freq="10min"))
    pred = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    alarm, gaps = first_alarm(ts, pred)
    assert alarm == ts[5] and gaps == 0


def test_first_alarm_gap_breaks_run():
    ts = pd.Series(list(pd.date_range("2023-01-01 00:00", periods=3, freq="10min")) +
                   list(pd.date_range("2023-01-01 02:00", periods=3, freq="10min")))
    alarm, gaps = first_alarm(ts, np.ones(6))
    assert alarm is None and gaps == 1  # 6 positives but not one continuous hour


def test_record_metrics_confusion():
    m = record_metrics([0, 0, 1, 1], [0.1, 0.9, 0.8, 0.2], 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_best_f1_threshold():
    thr = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < thr <= 0.8


def test_event_level_lead_time():
    ts = pd.date_range("2023-01-01", periods=12, freq="10min")
    win = pd.DataFrame({"event_id": 7, "ts": ts})
    events = pd.DataFrame({"event_id": [7], "asset_id": [1], "event_label": ["anomaly"],
                           "event_start": [ts[0]], "event_end": [ts[-1]]})
    proba = np.array([0.9] * 12)
    r = event_level(win, proba, 0.5, events).iloc[0]
    assert r["alarm"] and r["detection_delay_h"] == round(50 / 60, 1)
    assert r["lead_time_before_event_end_h"] == round(60 / 60, 1)
