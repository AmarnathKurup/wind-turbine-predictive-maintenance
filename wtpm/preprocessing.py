"""Stage 2 - build the processed Farm C dataset (labels, deduplication, folds, checks).

Reads the original CSVs read-only and writes ONLY to ``data/processed`` and ``reports/``.

Usage (from project root)::

    .\\.venv\\Scripts\\python.exe -m wtpm.preprocessing                 # all turbines
    .\\.venv\\Scripts\\python.exe -m wtpm.preprocessing --turbines 5 13  # quick trial

Outputs (data/processed/):
    records.pkl          deduplicated records: features, label, label_source, fold, asset_id, ts, status
    event_windows.pkl    every event window on its own file timeline (for event-level evaluation)
    fold_assignment.csv  turbine -> fold
    preprocessing_config.json
Reports (reports/preprocessing/):
    preprocessing_report.json, label_counts_per_event.csv, label_counts_per_turbine.csv
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from wtpm import config
from wtpm.data_loader import (assign_folds, clean_sentinels, dataset_path, load_event_info,
                              read_dataset_file)

PRIORITY = {s: i for i, s in enumerate(config.SOURCE_PRIORITY)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- labelling (one file)
def label_file_rows(df: pd.DataFrame, ev: pd.Series) -> pd.Series:
    """label_source for every row of one dataset file, using ONLY that file's own event and timestamps."""
    ts, is_pred = df["ts"], df["train_test"].eq("prediction")
    in_win = (ts >= ev["event_start"]) & (ts <= ev["event_end"])
    src = pd.Series("assumed_normal", index=df.index, dtype=object)
    if ev["event_label"] == "anomaly":
        src[is_pred & (ts < ev["event_start"])] = "excluded_pre_event"
        src[is_pred & (ts > ev["event_end"])] = "excluded_post_event"
        src[in_win] = "anomaly_window"
    else:
        src[in_win] = "normal_window"
    return src


# --------------------------------------------------------------------------- dedup (one turbine)
def match_same_records(prev: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Find rows of ``new`` that are the same physical records as rows of ``prev``.

    A fingerprint match counts only if it agrees with the dominant time offset of this file pair
    (identical sensor values at an inconsistent offset are treated as coincidence, not a copy).
    Returns a Series (index = new row index) of matched prev record_key, plus diagnostics.
    """
    m = new[["fp1", "fp2", "ts"]].reset_index().merge(
        prev[["fp1", "fp2", "ts", "record_key"]], on=["fp1", "fp2"], suffixes=("_new", "_prev"))
    diag = {"fingerprint_matches": int(len(m)), "unique_fp_matches": 0, "dominant_offset": None,
            "offset_share": None, "accepted": 0, "coincidental": int(len(m))}
    # Offset is estimated ONLY from fingerprints that occur once in each file: repeated vectors
    # (e.g. a stopped turbine with frozen readings) match many rows and would hide the true offset.
    uniq_new = ~new.duplicated(["fp1", "fp2"], keep=False)
    uniq_prev = ~prev.duplicated(["fp1", "fp2"], keep=False)
    fps_u_new = set(map(tuple, new.loc[uniq_new, ["fp1", "fp2"]].to_numpy()))
    fps_u_prev = set(map(tuple, prev.loc[uniq_prev, ["fp1", "fp2"]].to_numpy()))
    is_u = pd.Series(list(map(tuple, m[["fp1", "fp2"]].to_numpy()))).isin(fps_u_new & fps_u_prev).to_numpy()
    mu = m[is_u]
    diag["unique_fp_matches"] = int(len(mu))
    if len(mu) < config.DEDUP_MIN_MATCHES:
        return pd.Series(dtype="int64"), diag
    counts = (mu["ts_new"] - mu["ts_prev"]).value_counts()
    dom, share = counts.index[0], counts.iloc[0] / len(mu)
    diag.update(dominant_offset=str(dom), offset_share=round(float(share), 4))
    if share < config.DEDUP_MIN_OFFSET_SHARE:
        return pd.Series(dtype="int64"), diag
    off = m["ts_new"] - m["ts_prev"]
    acc = m[off == dom].drop_duplicates("index")
    diag.update(accepted=int(len(acc)), coincidental=int(len(m) - (off == dom).sum()))
    return pd.Series(acc["record_key"].to_numpy(), index=acc["index"].to_numpy()), diag


def dedup_turbine(files: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Merge all files of one turbine into unique physical records with a conservative label."""
    next_key = 0
    seen: list[pd.DataFrame] = []  # per processed file: fp1, fp2, ts, record_key
    occurrences = []               # (record_key, label_source, file order)
    kept_rows = []
    pair_diags = []
    within_file_dup_fps = 0
    for order, df in enumerate(files):
        within_file_dup_fps += int(df.duplicated(["fp1", "fp2"]).sum())
        key = pd.Series(-1, index=df.index, dtype="int64")
        for prev in seen:
            todo = key[key < 0].index
            if not len(todo):
                break
            matched, diag = match_same_records(prev, df.loc[todo])
            diag.update(file_prev=int(prev["event_id"].iloc[0]), file_new=int(df["event_id"].iloc[0]))
            pair_diags.append(diag)
            key.loc[matched.index] = matched.to_numpy()
        new = key[key < 0].index
        key.loc[new] = np.arange(next_key, next_key + len(new))
        next_key += len(new)
        df = df.assign(record_key=key.to_numpy())
        seen.append(df[["fp1", "fp2", "ts", "record_key", "event_id"]])
        occurrences.append(df[["record_key", "label_source"]].assign(order=order))
        kept_rows.append(df.loc[new])

    occ = pd.concat(occurrences, ignore_index=True)
    occ["prio"] = occ["label_source"].map(PRIORITY)
    first_src = occ.sort_values("order").drop_duplicates("record_key").set_index("record_key")["label_source"]
    final_src = occ.sort_values("prio").drop_duplicates("record_key").set_index("record_key")["label_source"]
    n_sources = occ.groupby("record_key")["label_source"].nunique()

    rec = pd.concat(kept_rows, ignore_index=True).set_index("record_key")
    rec["label_source"] = final_src.reindex(rec.index)
    rec["n_copies"] = occ.groupby("record_key").size().reindex(rec.index)
    rec = rec.reset_index()
    relabelled = (final_src != first_src.reindex(final_src.index))
    diag = {
        "rows_in_files": int(len(occ)),
        "unique_records": int(len(rec)),
        "duplicates_removed": int(len(occ) - len(rec)),
        "records_with_conflicting_labels": int((n_sources > 1).sum()),
        "records_relabelled_by_priority": int(relabelled.sum()),
        "relabel_transitions": {f"{a}->{b}": int(n) for (a, b), n in
                                pd.DataFrame({"a": first_src[relabelled.index[relabelled]],
                                              "b": final_src[relabelled]}).value_counts().items()},
        "within_file_repeated_fingerprints_kept": within_file_dup_fps,
        "file_pairs": pair_diags,
    }
    return rec, diag


# --------------------------------------------------------------------------- main build
def build(turbines: list[int] | None = None) -> dict:
    t0 = time.time()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rep_dir = config.REPORTS_DIR / "preprocessing"
    rep_dir.mkdir(parents=True, exist_ok=True)

    events = load_event_info()
    folds = assign_folds(events)
    fold_of = dict(zip(folds["asset_id"], folds["fold"]))
    assets = sorted(events["asset_id"].unique()) if not turbines else sorted(turbines)

    all_records, all_windows, per_event, turb_diag = [], [], [], {}
    sentinel_cells = 0
    feature_cols: list[str] | None = None

    for a in assets:
        evs = events[events["asset_id"] == a].sort_values("event_id")
        files = []
        for _, ev in evs.iterrows():
            df = read_dataset_file(dataset_path(int(ev["event_id"])))
            if feature_cols is None:
                feature_cols = [c for c in df.columns if c.endswith(config.FEATURE_SUFFIXES_STORED)]
            df, n = clean_sentinels(df)
            sentinel_cells += n
            df["event_id"] = int(ev["event_id"])
            df["label_source"] = label_file_rows(df, ev)
            # per-event counts on the file's own timeline (before dedup)
            vc = df["label_source"].value_counts()
            per_event.append({"asset_id": int(a), "event_id": int(ev["event_id"]), "event_label": ev["event_label"],
                              "fold": fold_of[a], "rows": len(df), **{s: int(vc.get(s, 0)) for s in config.SOURCE_PRIORITY}})
            # event window on own timeline, for event-level evaluation
            win = (df["ts"] >= ev["event_start"]) & (df["ts"] <= ev["event_end"])
            all_windows.append(df.loc[win, ["asset_id", "event_id", "ts", "status_type_id"] + feature_cols]
                               .assign(event_label=ev["event_label"], fold=fold_of[a]))
            files.append(df)
            log(f"turbine {a}: read event {int(ev['event_id'])} ({len(df)} rows)")
        rec, diag = dedup_turbine(files)
        rec["fold"] = fold_of[a]
        turb_diag[int(a)] = diag
        all_records.append(rec)
        log(f"turbine {a}: {diag['rows_in_files']} rows -> {diag['unique_records']} unique records, "
            f"{diag['records_relabelled_by_priority']} relabelled")
        del files

    records = pd.concat(all_records, ignore_index=True)
    records["label"] = records["label_source"].map(config.SOURCE_TO_LABEL).astype("int8")
    windows = pd.concat(all_windows, ignore_index=True)

    # ---- fingerprint collision check (same fp1, different fp2) across everything kept
    collisions = int(records.groupby("fp1")["fp2"].nunique().gt(1).sum())

    # ---- cross-turbine identical sensor vectors
    fp_turb = records.groupby(["fp1", "fp2"])["asset_id"].nunique()
    cross_fps = fp_turb[fp_turb > 1]
    cross_rows = records.set_index(["fp1", "fp2"]).index.isin(cross_fps.index)
    # identical vectors shared by a TEST turbine and a non-test turbine -> drop from test (leakage guard)
    test_mask = records["fold"].eq(config.TEST_FOLD)
    fps_nontest = set(map(tuple, records.loc[~test_mask, ["fp1", "fp2"]].to_numpy()))
    test_cross = test_mask & pd.Series(list(map(tuple, records[["fp1", "fp2"]].to_numpy()))).isin(fps_nontest)
    cross_labels = records.loc[cross_rows, "label_source"].value_counts().to_dict()
    records = records.loc[~test_cross.to_numpy()].reset_index(drop=True)

    # ---- per-turbine label counts (after dedup)
    per_turbine = (records.groupby(["asset_id", "fold", "label_source"]).size().unstack(fill_value=0)
                   .reindex(columns=config.SOURCE_PRIORITY, fill_value=0).reset_index())
    ev_counts = events.assign(anom=events["event_label"].eq("anomaly")).groupby("asset_id")["anom"].agg(["sum", "count"])
    per_turbine["anomaly_events"] = per_turbine["asset_id"].map(ev_counts["sum"])
    per_turbine["normal_events"] = per_turbine["asset_id"].map(ev_counts["count"] - ev_counts["sum"])

    # ---- fold summary + checks
    fold_summary = []
    for k, g in records.groupby("fold"):
        evk = events[events["asset_id"].isin(g["asset_id"].unique())]
        fold_summary.append({"fold": int(k), "role": "test" if k == config.TEST_FOLD else "cv",
                             "turbines": sorted(int(x) for x in g["asset_id"].unique()),
                             "anomaly_events": int((evk["event_label"] == "anomaly").sum()),
                             "normal_events": int((evk["event_label"] == "normal").sum()),
                             "positives": int((g["label"] == 1).sum()), "negatives": int((g["label"] == 0).sum()),
                             "negatives_confirmed": int((g["label_source"] == "normal_window").sum()),
                             "negatives_assumed": int((g["label_source"] == "assumed_normal").sum()),
                             "excluded": int((g["label"] == -1).sum())})
    fp_test = set(map(tuple, records.loc[records["fold"] == config.TEST_FOLD, ["fp1", "fp2"]].to_numpy()))
    fp_other = set(map(tuple, records.loc[records["fold"] != config.TEST_FOLD, ["fp1", "fp2"]].to_numpy()))
    checks = {
        "every_fold_has_pos_and_neg": all(f["positives"] > 0 and f["negatives"] > 0 for f in fold_summary),
        "turbine_in_single_fold": bool(records.groupby("asset_id")["fold"].nunique().max() == 1),
        "fingerprints_shared_test_vs_trainval_after_guard": len(fp_test & fp_other),
        "fingerprint_collisions_fp1_same_fp2_differs": collisions,
        "status_type_id_is_feature": "status_type_id" in feature_cols,
    }

    # ---- save
    keep_cols = ["asset_id", "event_id", "ts", "status_type_id", "train_test", "label", "label_source",
                 "fold", "n_copies", "fp1", "fp2"] + feature_cols
    records[keep_cols].to_pickle(config.PROCESSED_DIR / "records.pkl")
    windows.to_pickle(config.PROCESSED_DIR / "event_windows.pkl")
    folds.to_csv(config.PROCESSED_DIR / "fold_assignment.csv", index=False)
    pd.DataFrame(per_event).to_csv(rep_dir / "label_counts_per_event.csv", index=False)
    per_turbine.to_csv(rep_dir / "label_counts_per_turbine.csv", index=False)
    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(config).items()
           if k.isupper() and isinstance(v, (int, float, str, list, tuple, dict))}
    (config.PROCESSED_DIR / "preprocessing_config.json").write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "farm": config.FARM, "turbines": [int(a) for a in assets],
        "n_feature_columns_stored": len(feature_cols), "sentinel_cells_set_missing": sentinel_cells,
        "records_total": int(len(records)),
        "label_counts": records["label_source"].value_counts().to_dict(),
        "dedup_totals": {k: int(sum(d[k] for d in turb_diag.values())) for k in
                         ("rows_in_files", "unique_records", "duplicates_removed", "records_with_conflicting_labels",
                          "records_relabelled_by_priority", "within_file_repeated_fingerprints_kept")},
        "cross_turbine_identical_vectors": {"fingerprints": int(len(cross_fps)), "records": int(cross_rows.sum()),
                                            "label_sources": cross_labels,
                                            "dropped_from_test_fold": int(test_cross.sum())},
        "folds": fold_summary, "checks": checks, "per_turbine_dedup": turb_diag,
        "seconds": round(time.time() - t0, 1),
    }
    (rep_dir / "preprocessing_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"Saved {len(records)} records, {len(windows)} event-window rows. Checks: {checks}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turbines", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)
    build(args.turbines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
