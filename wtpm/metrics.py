"""Record-level and event-level evaluation helpers (pure functions, unit-tested)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from wtpm import config


def record_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    both = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)), "positives": int(y_true.sum()), "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)) if both else None,
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1],
                                                       target_names=["normal", "anomaly"],
                                                       output_dict=True, zero_division=0),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximising F1 - must be chosen on training/CV predictions, never on test."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


def first_alarm(ts: pd.Series, pred: np.ndarray, k: int = config.ALARM_CONSECUTIVE,
                freq_min: int = config.FREQ_MINUTES):
    """Time of the first alarm = k consecutive positive records that are also contiguous in time
    (each step exactly ``freq_min`` minutes). A timestamp gap breaks the run.
    Returns (alarm_timestamp or None, n_gaps_in_series)."""
    order = np.argsort(ts.to_numpy())
    t = ts.to_numpy()[order]
    p = np.asarray(pred)[order].astype(bool)
    step = np.diff(t) == np.timedelta64(freq_min, "m")
    n_gaps = int((~step).sum())
    run = 0
    for i in range(len(p)):
        contiguous = i > 0 and step[i - 1]
        run = (run + 1 if contiguous else 1) if p[i] else 0
        if run >= k:
            return pd.Timestamp(t[i]), n_gaps
    return None, n_gaps


def event_level(windows: pd.DataFrame, proba: np.ndarray, threshold: float, events: pd.DataFrame) -> pd.DataFrame:
    """One row per event window: detected?, detection delay after event_start, lead time before event_end.
    For normal events, a detection is a false alarm."""
    w = windows.assign(pred=(proba >= threshold).astype(int), proba=proba)
    ev = events.set_index("event_id")
    out = []
    for eid, g in w.groupby("event_id"):
        e = ev.loc[eid]
        alarm, gaps = first_alarm(g["ts"], g["pred"].to_numpy())
        row = {"event_id": int(eid), "asset_id": int(e["asset_id"]), "event_label": e["event_label"],
               "window_rows": int(len(g)), "window_gaps": gaps,
               "share_records_flagged": round(float(g["pred"].mean()), 4),
               "mean_probability": round(float(g["proba"].mean()), 4),
               "alarm": alarm is not None, "first_alarm": None if alarm is None else str(alarm),
               "detection_delay_h": None, "lead_time_before_event_end_h": None,
               "event_duration_h": round((e["event_end"] - e["event_start"]) / pd.Timedelta("1h"), 1)}
        if alarm is not None:
            row["detection_delay_h"] = round((alarm - e["event_start"]) / pd.Timedelta("1h"), 1)
            row["lead_time_before_event_end_h"] = round((e["event_end"] - alarm) / pd.Timedelta("1h"), 1)
        out.append(row)
    return pd.DataFrame(out)


def summarise_events(ev_df: pd.DataFrame) -> dict:
    a = ev_df[ev_df["event_label"] == "anomaly"]
    n = ev_df[ev_df["event_label"] == "normal"]
    return {
        "anomaly_events": int(len(a)), "anomaly_events_detected": int(a["alarm"].sum()),
        "anomaly_events_missed": int((~a["alarm"]).sum()),
        "normal_events": int(len(n)), "normal_events_with_false_alarm": int(n["alarm"].sum()),
        "median_detection_delay_h": None if not a["alarm"].any() else float(a.loc[a["alarm"], "detection_delay_h"].median()),
        "median_lead_time_before_event_end_h": None if not a["alarm"].any() else float(a.loc[a["alarm"], "lead_time_before_event_end_h"].median()),
    }
