"""Stage 3 - baselines + Random Forest with turbine-grouped cross-validation, then final model.

The TEST fold (config.TEST_FOLD) is never touched here. Everything (feature filtering, negative
thinning, model fitting, threshold choice) is decided on folds 1-4 only.

Usage (from project root)::

    .\\.venv\\Scripts\\python.exe -m wtpm.train                       # _avg baseline feature set
    .\\.venv\\Scripts\\python.exe -m wtpm.train --feature-set avg_std  # optional comparison
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wtpm import config
from wtpm.data_loader import load_event_info
from wtpm.features import (AMBIENT_COL, OUTSIDE_TEMP_COLS, ambient_delta, apply_turbine_baselines,
                           candidate_features, fit_non_constant, fit_turbine_baselines,
                           temperature_columns, thin_negatives)
from wtpm.metrics import best_f1_threshold, event_level, record_metrics, summarise_events

FEATURE_SETS = {"avg": ("_avg",), "avg_std": ("_avg", "_std"), "avg_ambdelta": ("_avg",),
                "avg_selfref": ("_avg",)}
TRANSFORMS = {"avg_ambdelta": "ambient_delta", "avg_selfref": "ambient_delta"}
TURBINE_NORM = {"avg_selfref"}  # + per-turbine robust baseline from the turbine's own train period

RF_GRID = {
    "rf_leaf1_sqrt_bal": dict(min_samples_leaf=1, max_features="sqrt", class_weight="balanced"),
    "rf_leaf5_sqrt_bal": dict(min_samples_leaf=5, max_features="sqrt", class_weight="balanced"),
    "rf_leaf5_sqrt_none": dict(min_samples_leaf=5, max_features="sqrt", class_weight=None),
    "rf_depth20_leaf5_bal": dict(max_depth=20, min_samples_leaf=5, max_features="sqrt", class_weight="balanced"),
    "rf_leaf5_f0.1_bal": dict(min_samples_leaf=5, max_features=0.1, class_weight="balanced"),
}
RF_FIXED = dict(n_estimators=200, n_jobs=-1, random_state=config.RANDOM_STATE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_model(name: str):
    if name == "dummy_majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg_balanced":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))
    return RandomForestClassifier(**RF_FIXED, **RF_GRID[name])


def apply_transform(df, feats, transform):
    if transform != "ambient_delta":
        return df, feats
    tcols = temperature_columns(feats)
    X = ambient_delta(df[feats], tcols)
    meta_cols = [c for c in df.columns if c not in feats]
    return pd.concat([df[meta_cols], X], axis=1), list(X.columns)


def load_data(suffixes, transform=None, turbine_norm=False):
    rec = pd.read_pickle(config.PROCESSED_DIR / "records.pkl")
    feats = candidate_features(rec.columns, suffixes)
    # baselines use ALL of a turbine's own historical `train` rows (incl. later-excluded ones), no labels
    rec = rec[["asset_id", "event_id", "ts", "label", "label_source", "fold", "status_type_id", "train_test"] + feats]
    win = pd.read_pickle(config.PROCESSED_DIR / "event_windows.pkl")
    win = win[[c for c in win.columns if c not in set(candidate_features(win.columns, ("_avg", "_std"))) or c in feats]]
    rec, new_feats = apply_transform(rec, feats, transform)
    win, _ = apply_transform(win, feats, transform)
    baselines = None
    if turbine_norm:
        baselines = fit_turbine_baselines(rec, new_feats, rec["train_test"].eq("train"))
        rec = apply_turbine_baselines(rec, new_feats, baselines)
        win = apply_turbine_baselines(win, new_feats, baselines)
    rec = rec[rec["label"] != config.LABEL_EXCLUDED]
    return rec, win, new_feats, feats, baselines


def run_cv(rec, win, feats, events, models):
    cv = rec[rec["fold"] != config.TEST_FOLD]
    cv_win = win[win["fold"] != config.TEST_FOLD]
    folds = sorted(cv["fold"].unique())
    results, oof = {m: [] for m in models}, {m: [] for m in models}
    ev_rows = {m: [] for m in models}
    for k in folds:
        tr, va = cv[cv["fold"] != k], cv[cv["fold"] == k]
        tr = thin_negatives(tr)
        cols = fit_non_constant(tr[feats])  # fitted on the training part only
        for m in models:
            t0 = time.time()
            model = make_model(m).fit(tr[cols], tr["label"])
            p = model.predict_proba(va[cols])[:, 1]
            r = record_metrics(va["label"], p, 0.5)
            r.update(fold=int(k), train_rows=int(len(tr)), train_pos=int(tr["label"].sum()),
                     n_features=len(cols), fit_seconds=round(time.time() - t0, 1))
            results[m].append(r)
            oof[m].append(pd.DataFrame({"fold": k, "label": va["label"].to_numpy(), "proba": p}))
            wk = cv_win[cv_win["fold"] == k]
            ev_rows[m].append((wk, model.predict_proba(wk[cols])[:, 1]))
            log(f"fold {k} {m}: PR-AUC={r['pr_auc']:.4f} ROC-AUC={r['roc_auc']:.4f} "
                f"F1@0.5={r['f1']:.4f} ({r['fit_seconds']}s, {len(tr)} train rows)")
    return results, oof, ev_rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="avg")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to run")
    args = ap.parse_args(argv)
    tag = args.feature_set
    np.random.seed(config.RANDOM_STATE)

    for d in (config.ARTIFACTS_DIR, config.METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    events = load_event_info()
    transform = TRANSFORMS.get(tag)
    rec, win, feats, raw_feats, baselines = load_data(FEATURE_SETS[tag], transform, tag in TURBINE_NORM)
    log(f"Loaded {len(rec)} labelled records, {len(feats)} candidate features ({tag})")

    models = args.models or ["dummy_majority", "logreg_balanced", *RF_GRID]
    results, oof, ev_rows = run_cv(rec, win, feats, events, models)

    # ---- summarise CV; select best RF by mean PR-AUC
    summary = []
    cv_detail = {}
    for m in models:
        o = pd.concat(oof[m])
        thr = best_f1_threshold(o["label"], o["proba"]) if m != "dummy_majority" else 0.5
        pooled = record_metrics(o["label"], o["proba"], thr)
        evdf = pd.concat([event_level(w, p, thr, events) for w, p in ev_rows[m]], ignore_index=True)
        prs = [r["pr_auc"] for r in results[m]]
        summary.append({"model": m, "pr_auc_mean": np.mean(prs), "pr_auc_std": np.std(prs),
                        "pr_auc_min": np.min(prs), "pr_auc_max": np.max(prs),
                        "roc_auc_mean": np.mean([r["roc_auc"] for r in results[m]]),
                        "cv_threshold": thr, "pooled_precision": pooled["precision"],
                        "pooled_recall": pooled["recall"], "pooled_f1": pooled["f1"],
                        **{f"events_{k}": v for k, v in summarise_events(evdf).items()}})
        cv_detail[m] = {"per_fold": results[m], "pooled_oof_at_cv_threshold": pooled,
                        "event_level": evdf.to_dict(orient="records")}
    summary = pd.DataFrame(summary).sort_values("pr_auc_mean", ascending=False)
    summary.to_csv(config.METRICS_DIR / f"cv_summary_{tag}.csv", index=False)
    (config.METRICS_DIR / f"cv_results_{tag}.json").write_text(json.dumps(cv_detail, indent=2, default=str), encoding="utf-8")
    log("CV summary:\n" + summary[["model", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "cv_threshold",
                                   "pooled_f1", "events_anomaly_events_detected", "events_anomaly_events",
                                   "events_normal_events_with_false_alarm"]].to_string(index=False))

    rf_rows = summary[summary["model"].str.startswith("rf_")]
    if rf_rows.empty:
        log("No RF model run; skipping final fit.")
        return 0
    best = rf_rows.iloc[0]
    log(f"Selected {best['model']} (mean CV PR-AUC {best['pr_auc_mean']:.4f}), CV threshold {best['cv_threshold']:.4f}")

    # ---- final model on ALL non-test folds
    trv = thin_negatives(rec[rec["fold"] != config.TEST_FOLD])
    cols = fit_non_constant(trv[feats])
    t0 = time.time()
    final = make_model(best["model"]).fit(trv[cols], trv["label"])
    log(f"Final model fitted on {len(trv)} rows x {len(cols)} features in {time.time() - t0:.0f}s")

    version = f"rf-farm{config.FARM}-{tag}-v1"
    joblib.dump(final, config.ARTIFACTS_DIR / f"model_{tag}.joblib", compress=3)
    meta = {
        "model_version": version, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "objective": "Pre-failure anomaly-state classification of a single 10-minute SCADA record "
                     "(NOT a forecast of future failure).",
        "farm": config.FARM, "feature_set": tag, "features": cols, "n_features": len(cols),
        "threshold": float(best["cv_threshold"]), "threshold_source": "max F1 on pooled out-of-fold CV predictions",
        "model": best["model"], "params": {**RF_FIXED, **RF_GRID[best["model"]]},
        "classes": {"0": "normal", "1": "anomaly (inside expert pre-failure anomaly window)"},
        "train_folds": sorted(int(f) for f in trv["fold"].unique()), "test_fold": config.TEST_FOLD,
        "train_rows": int(len(trv)), "train_positives": int(trv["label"].sum()),
        "train_turbines": sorted(int(a) for a in trv["asset_id"].unique()),
        "cleaning": {"negative_is_missing_prefixes": list(config.NEGATIVE_IS_MISSING_PREFIXES)},
        "transform": None if transform is None else {
            "type": transform, "ambient_column": AMBIENT_COL, "dropped_columns": list(OUTSIDE_TEMP_COLS),
            "temperature_columns": [c for c in temperature_columns(raw_feats) if c not in OUTSIDE_TEMP_COLS]},
        "raw_input_features": None if transform is None else raw_feats,
        "turbine_baselines_file": None if baselines is None else f"turbine_baselines_{tag}.csv",
        "turbine_baselines_reference": None if baselines is None else
            "per-turbine median/IQR of the turbine's own historical `train` period (no labels used)",
        "alarm_rule": {"consecutive_records": config.ALARM_CONSECUTIVE, "freq_minutes": config.FREQ_MINUTES},
        "missing_values": "NaN allowed (handled natively by scikit-learn RandomForest)",
        "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
    }
    (config.ARTIFACTS_DIR / f"model_metadata_{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if baselines is not None:
        baselines.to_csv(config.ARTIFACTS_DIR / f"turbine_baselines_{tag}.csv", index=False)
    pd.read_csv(config.PROCESSED_DIR / "fold_assignment.csv").to_csv(config.METRICS_DIR / "fold_assignment.csv", index=False)
    log(f"Saved artifacts/model_{tag}.joblib and model_metadata_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
