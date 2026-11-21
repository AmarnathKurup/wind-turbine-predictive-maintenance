"""Stage 4 - one-time evaluation of the saved final model on the held-out TEST turbines.

Usage (from project root)::

    .\\.venv\\Scripts\\python.exe -m wtpm.evaluate --feature-set avg
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay  # noqa: E402

from wtpm import config  # noqa: E402
from wtpm.data_loader import load_event_info  # noqa: E402
from wtpm.features import ambient_delta, apply_turbine_baselines  # noqa: E402
from wtpm.metrics import event_level, record_metrics, summarise_events  # noqa: E402

# Colour-blind-safe pair (blue / orange) used consistently across figures.
C_NORMAL, C_ANOM = "#2E6FBA", "#E07B1F"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def describe_features(cols):
    fd = pd.read_csv(config.FARM_DIR / "feature_description.csv", sep=";")
    d = dict(zip(fd["sensor_name"], fd["description"]))
    return [f"{c} ({d.get(c.rsplit('_', 1)[0], '?')})" for c in cols]  # ambient-delta cols keep sensor name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="avg")
    ap.add_argument("--perm-sample", type=int, default=20000, help="test rows for permutation importance (0=skip)")
    args = ap.parse_args(argv)
    tag = args.feature_set
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    meta = json.loads((config.ARTIFACTS_DIR / f"model_metadata_{tag}.json").read_text(encoding="utf-8"))
    model = joblib.load(config.ARTIFACTS_DIR / f"model_{tag}.joblib")
    cols, thr = meta["features"], meta["threshold"]
    events = load_event_info()

    rec = pd.read_pickle(config.PROCESSED_DIR / "records.pkl")
    test = rec[(rec["fold"] == config.TEST_FOLD) & (rec["label"] != config.LABEL_EXCLUDED)]
    del rec
    win = pd.read_pickle(config.PROCESSED_DIR / "event_windows.pkl")
    twin = win[win["fold"] == config.TEST_FOLD]
    tf = meta.get("transform")
    if tf and tf["type"] == "ambient_delta":
        raw = meta["raw_input_features"]
        test = pd.concat([test.drop(columns=raw), ambient_delta(test[raw], tf["temperature_columns"])], axis=1)
        twin = pd.concat([twin.drop(columns=raw), ambient_delta(twin[raw], tf["temperature_columns"])], axis=1)
    if meta.get("turbine_baselines_file"):
        bl = pd.read_csv(config.ARTIFACTS_DIR / meta["turbine_baselines_file"])
        test = apply_turbine_baselines(test, cols, bl)
        twin = apply_turbine_baselines(twin, cols, bl)
    assert not set(test["asset_id"]) & set(meta["train_turbines"]), "test turbine seen in training!"

    proba = model.predict_proba(test[cols])[:, 1]
    m_thr = record_metrics(test["label"], proba, thr)
    m_05 = record_metrics(test["label"], proba, 0.5)
    evdf = event_level(twin, model.predict_proba(twin[cols])[:, 1], thr, events)
    evsum = summarise_events(evdf)

    # ---- error analysis
    pred = (proba >= thr).astype(int)
    t = test.assign(pred=pred, proba=proba)
    fp = t[(t["label"] == 0) & (t["pred"] == 1)]
    fn = t[(t["label"] == 1) & (t["pred"] == 0)]
    errors = {
        "false_positives_by_label_source": fp["label_source"].value_counts().to_dict(),
        "false_positive_rate_by_label_source": (t[t["label"] == 0].groupby("label_source")["pred"].mean().round(4).to_dict()),
        "false_positives_by_status": fp["status_type_id"].value_counts().to_dict(),
        "false_positives_by_turbine": fp["asset_id"].value_counts().to_dict(),
        "false_negatives_by_event": fn["event_id"].value_counts().to_dict(),
        "recall_by_status_inside_anomaly_windows": t[t["label"] == 1].groupby("status_type_id")["pred"].mean().round(4).to_dict(),
        "positives_by_status": t[t["label"] == 1]["status_type_id"].value_counts().to_dict(),
    }

    # ---- importances
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
    perm = None
    if args.perm_sample:
        s = test.sample(min(args.perm_sample, len(test)), random_state=config.RANDOM_STATE)
        pi = permutation_importance(model, s[cols], s["label"], scoring="average_precision",
                                    n_repeats=3, random_state=config.RANDOM_STATE, n_jobs=1)
        perm = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False)

    out = {
        "model_version": meta["model_version"], "evaluated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_fold": config.TEST_FOLD, "test_turbines": sorted(int(a) for a in test["asset_id"].unique()),
        "test_events": {"anomaly": int((evdf["event_label"] == "anomaly").sum()), "normal": int((evdf["event_label"] == "normal").sum())},
        "record_level_at_cv_threshold": m_thr, "record_level_at_0.5": m_05,
        "event_level_summary": evsum, "event_level": evdf.to_dict(orient="records"),
        "error_analysis": errors,
        "top20_impurity_importance": imp.head(20).round(5).to_dict(),
        "top20_permutation_importance_pr_auc": None if perm is None else perm.head(20).round(5).to_dict(),
    }
    (config.METRICS_DIR / f"test_metrics_{tag}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    evdf.to_csv(config.METRICS_DIR / f"test_event_level_{tag}.csv", index=False)

    # ---- figures
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(test["label"], pred, display_labels=["normal", "anomaly"],
                                            cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Test confusion matrix (threshold {thr:.2f})")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / f"confusion_matrix_{tag}.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    PrecisionRecallDisplay.from_predictions(test["label"], proba, ax=axes[0], color=C_ANOM, name="Random Forest")
    axes[0].axhline(test["label"].mean(), ls="--", color="grey", lw=1, label="no-skill (prevalence)")
    axes[0].legend(); axes[0].set_title("Precision-Recall (test turbines)")
    RocCurveDisplay.from_predictions(test["label"], proba, ax=axes[1], name="Random Forest",
                                     curve_kwargs={"color": C_NORMAL})  # sklearn >= 1.7 API
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey", lw=1); axes[1].set_title("ROC (test turbines)")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / f"pr_roc_curves_{tag}.png", dpi=150); plt.close(fig)

    top = (perm if perm is not None else imp).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(describe_features(top.index), top.values, color=C_NORMAL)
    ax.set_xlabel("Mean decrease in PR-AUC (permutation, test sample)" if perm is not None else "Impurity importance")
    ax.set_title("Top 15 features")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / f"feature_importance_{tag}.png", dpi=150); plt.close(fig)

    an = evdf[evdf["event_label"] == "anomaly"]
    if len(an):
        fig, axes = plt.subplots(len(an), 1, figsize=(10, 1.9 * len(an)), squeeze=False)
        twp = twin.assign(proba=model.predict_proba(twin[cols])[:, 1])
        for ax, (_, e) in zip(axes[:, 0], an.iterrows()):
            g = twp[twp["event_id"] == e["event_id"]].sort_values("ts")
            ax.plot(g["ts"], g["proba"], color=C_ANOM, lw=0.8)
            ax.axhline(thr, ls="--", color="grey", lw=1)
            if e["alarm"]:
                ax.axvline(pd.Timestamp(e["first_alarm"]), color=C_NORMAL, lw=1.5)
            ax.set_ylim(0, 1); ax.set_ylabel("P(anomaly)")
            ax.set_title(f"Event {e['event_id']} (turbine {e['asset_id']}): "
                         f"{'alarm after ' + str(e['detection_delay_h']) + ' h' if e['alarm'] else 'MISSED'}", fontsize=9)
        fig.tight_layout(); fig.savefig(config.FIGURES_DIR / f"test_anomaly_events_{tag}.png", dpi=130); plt.close(fig)

    log(f"TEST @thr={thr:.3f}: PR-AUC={m_thr['pr_auc']:.4f} ROC-AUC={m_thr['roc_auc']:.4f} "
        f"P={m_thr['precision']:.4f} R={m_thr['recall']:.4f} F1={m_thr['f1']:.4f} CM={m_thr['confusion_matrix']}")
    log(f"Event level: {evsum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
