"""Stage 1 - read-only inspection of the CARE to Compare dataset.

This module ONLY READS the dataset. It never writes, moves, renames or extracts
anything inside the dataset folder; all outputs go to ``--out`` (which must be
outside the dataset folder).

It does NOT preprocess data, create labels or train models. It measures what
is in the files so that later decisions are based on observed facts.

Usage (from the wind_turbine_ml project root)::

    .\\.venv\\Scripts\\python.exe -m wtpm.inspect_dataset --data-dir ..\\CARE_To_Compare --out reports\\inspection
    # quick trial on one farm:
    .\\.venv\\Scripts\\python.exe -m wtpm.inspect_dataset --data-dir ..\\CARE_To_Compare --out reports\\inspection_trial --farms A

Outputs (in ``--out``):
    inspection_stats.json        all raw statistics (machine readable)
    per_dataset_summary.csv      one row per dataset file
    column_stats_farm_<X>.csv    per-column missing / range / constant stats per farm
    inspection_report.md         human-readable tables generated from the stats
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Column names documented in the dataset README (verified against files at runtime).
META_COLS = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]
# README: "status 0 and 2 are considered normal operation".
NORMAL_STATUS = {0, 2}
STATUS_NAMES = {0: "normal", 1: "derated", 2: "idling", 3: "service", 4: "downtime", 5: "other"}
STAT_SUFFIX = {"average": "avg", "minimum": "min", "maximum": "max", "std_dev": "std"}
TS_FORMAT = "%Y-%m-%d %H:%M:%S"
EXPECTED_FREQ = pd.Timedelta("10min")
POST_EVENT_HORIZON = pd.Timedelta("1D")


# --------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_small_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Read a small metadata CSV whose encoding is unknown (tries UTF-8, then cp1252)."""
    raw = path.read_bytes()  # read-only
    for enc in ("utf-8", "cp1252"):
        try:
            text = raw.decode(enc)
            return pd.read_csv(io.StringIO(text), sep=";"), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as utf-8 or cp1252")


def to_jsonable(obj):
    """Convert numpy / pandas scalars and containers to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def ts_str(x) -> str | None:
    return None if pd.isna(x) else str(x)


def status_runs(status: pd.Series) -> pd.DataFrame:
    """Contiguous runs of identical status codes: code, start position, length (rows)."""
    change = status.ne(status.shift())
    run_id = change.cumsum()
    runs = status.groupby(run_id).agg(code="first", length="size")
    runs["start_pos"] = np.flatnonzero(change.to_numpy())
    return runs.reset_index(drop=True)


# --------------------------------------------------------------------------- metadata
def inspect_metadata(farm_dir: Path) -> dict:
    events, ev_enc = read_small_csv(farm_dir / "event_info.csv")
    feats, fd_enc = read_small_csv(farm_dir / "feature_description.csv")
    raw_event_cols = list(events.columns)
    # Farm files differ ("asset" vs "asset_id"); normalise in memory only.
    events = events.rename(columns={"asset": "asset_id"})

    # Expected data columns implied by feature_description.
    stat_col = "statistics_type" if "statistics_type" in feats.columns else "statistic_type"
    expected_cols = []
    sensor_info = {}
    for _, r in feats.iterrows():
        stats = [s.strip() for s in str(r[stat_col]).split(",")]
        for s in stats:
            expected_cols.append(f"{r['sensor_name']}_{STAT_SUFFIX.get(s, s)}")
        sensor_info[r["sensor_name"]] = {
            "description": r.get("description"),
            "unit": r.get("unit"),
            "is_angle": r.get("is_angle"),
            "is_counter": r.get("is_counter"),
            "statistics": stats,
        }

    return {
        "event_info_encoding": ev_enc,
        "event_info_columns_raw": raw_event_cols,
        "events": events,
        "feature_description_encoding": fd_enc,
        "feature_description_columns": list(feats.columns),
        "n_sensors_described": len(feats),
        "n_expected_feature_columns": len(expected_cols),
        "expected_cols": expected_cols,
        "sensor_info": sensor_info,
        "n_angle_sensors": int(feats["is_angle"].astype(str).str.lower().eq("true").sum()),
        "n_counter_sensors": int(feats["is_counter"].astype(str).str.lower().eq("true").sum()),
    }


# --------------------------------------------------------------------------- one dataset
def inspect_dataset(path: Path, farm: str, ev: pd.Series | None, all_events: pd.DataFrame):
    t0 = time.time()
    df = pd.read_csv(path, sep=";")  # read-only
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    res: dict = {
        "farm": farm,
        "event_id": int(path.stem),
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "meta_cols_missing": [c for c in META_COLS if c not in cols],
    }
    sensor_cols = [c for c in cols if c not in META_COLS]
    num = df[sensor_cols].select_dtypes(include="number")
    res["n_sensor_cols"] = len(sensor_cols)
    res["non_numeric_sensor_cols"] = [c for c in sensor_cols if c not in num.columns]
    res["dtype_counts"] = {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()}

    # ---- timestamps
    ts = pd.to_datetime(df["time_stamp"], format=TS_FORMAT, errors="coerce")
    res["ts_unparseable"] = int(ts.isna().sum())
    res["ts_min"], res["ts_max"] = ts_str(ts.min()), ts_str(ts.max())
    res["span_days"] = round((ts.max() - ts.min()) / pd.Timedelta("1D"), 1)
    res["ts_monotonic_increasing"] = bool(ts.is_monotonic_increasing)
    res["ts_duplicates"] = int(ts.duplicated().sum())
    d = ts.diff().dropna()
    res["interval_top"] = {str(k): int(v) for k, v in d.value_counts().head(5).items()}
    gaps = d[d > EXPECTED_FREQ]
    res["n_gaps"] = int(len(gaps))
    res["missing_slots"] = int(((gaps / EXPECTED_FREQ).round() - 1).sum()) if len(gaps) else 0
    res["max_gap"] = str(gaps.max()) if len(gaps) else "0"
    res["n_nonpositive_intervals"] = int((d <= pd.Timedelta(0)).sum())

    # ---- id / asset
    res["id_is_0_to_n"] = bool(np.array_equal(df["id"].to_numpy(), np.arange(n_rows)))
    res["asset_ids_in_file"] = sorted(pd.unique(df["asset_id"]).tolist())

    # ---- train / prediction split
    tt = df["train_test"].astype(str)
    res["train_test_counts"] = tt.value_counts().to_dict()
    is_train = tt.eq("train")
    is_pred = tt.eq("prediction")
    tr, pr = ts[is_train], ts[is_pred]
    res["train_range"] = [ts_str(tr.min()), ts_str(tr.max())]
    res["pred_range"] = [ts_str(pr.min()), ts_str(pr.max())]
    res["train_strictly_before_pred"] = bool(len(tr) and len(pr) and tr.max() < pr.min())
    res["split_label_changes"] = int(tt.ne(tt.shift()).sum() - 1)  # 1 = one clean boundary

    # ---- status
    st = df["status_type_id"]
    res["status_counts"] = {int(k): int(v) for k, v in st.value_counts().sort_index().items()}
    res["status_counts_train"] = {int(k): int(v) for k, v in st[is_train].value_counts().sort_index().items()}
    res["status_counts_pred"] = {int(k): int(v) for k, v in st[is_pred].value_counts().sort_index().items()}
    res["abnormal_status_frac"] = float((~st.isin(NORMAL_STATUS)).mean())
    runs = status_runs(st)
    abn_flag = ~st.isin(NORMAL_STATUS)
    abn_runs = status_runs(abn_flag.astype(int))
    abn_runs = abn_runs[abn_runs["code"] == 1]
    res["abnormal_status_episodes"] = int(len(abn_runs))
    res["abnormal_episode_len_rows"] = {
        "median": float(abn_runs["length"].median()) if len(abn_runs) else 0,
        "max": int(abn_runs["length"].max()) if len(abn_runs) else 0,
        "n_ge_6rows_1h": int((abn_runs["length"] >= 6).sum()),
    }
    res["status_runs_by_code"] = {
        int(c): {"n_runs": int(len(g)), "median_len": float(g["length"].median()), "max_len": int(g["length"].max())}
        for c, g in runs.groupby("code")
    }

    # ---- this file's event window
    if ev is not None:
        es = pd.Timestamp(ev["event_start"])
        ee = pd.Timestamp(ev["event_end"])
        in_win = (ts >= es) & (ts <= ee)
        res["event_label"] = ev["event_label"]
        res["event_asset_id"] = int(ev["asset_id"])
        res["asset_matches_event_info"] = res["asset_ids_in_file"] == [int(ev["asset_id"])]
        res["event_start"], res["event_end"] = str(es), str(ee)
        res["event_duration_days"] = round((ee - es) / pd.Timedelta("1D"), 2)
        res["event_window_rows"] = int(in_win.sum())
        res["event_window_split"] = tt[in_win].value_counts().to_dict()
        res["event_window_status"] = {int(k): int(v) for k, v in st[in_win].value_counts().sort_index().items()}
        res["event_window_abnormal_frac"] = float((~st[in_win].isin(NORMAL_STATUS)).mean()) if in_win.any() else None
        # Do the *_id columns in event_info point to the rows with those timestamps?
        id_at_start = df.loc[ts.eq(es), "id"]
        id_at_end = df.loc[ts.eq(ee), "id"]
        res["event_start_ts_present"] = bool(len(id_at_start))
        res["event_end_ts_present"] = bool(len(id_at_end))
        res["event_start_id_matches_row_id"] = bool(len(id_at_start) and int(id_at_start.iloc[0]) == int(ev["event_start_id"]))
        res["event_end_id_matches_row_id"] = bool(len(id_at_end) and int(id_at_end.iloc[0]) == int(ev["event_end_id"]))
        res["row_id_at_event_start"] = int(id_at_start.iloc[0]) if len(id_at_start) else None
        res["pred_rows_before_event"] = int((is_pred & (ts < es)).sum())
        res["pred_rows_after_event"] = int((is_pred & (ts > ee)).sum())
        res["train_rows_after_event_start"] = int((is_train & (ts >= es)).sum())
        post = (ts > ee) & (ts <= ee + POST_EVENT_HORIZON)
        res["status_1d_after_event_end"] = {int(k): int(v) for k, v in st[post].value_counts().sort_index().items()}
        pre = (ts < es) & (ts >= es - POST_EVENT_HORIZON)
        res["status_1d_before_event_start"] = {int(k): int(v) for k, v in st[pre].value_counts().sort_index().items()}
        win_down = ts[in_win & st.eq(4)]
        res["first_downtime_in_window_offset_h"] = (
            round((win_down.min() - es) / pd.Timedelta("1h"), 2) if len(win_down) else None)
        win_abn = ts[in_win & ~st.isin(NORMAL_STATUS)]
        res["first_abnormal_in_window_offset_h"] = (
            round((win_abn.min() - es) / pd.Timedelta("1h"), 2) if len(win_abn) else None)

    # ---- other anomaly events of the same turbine that fall inside this file's time span
    same_asset = all_events[(all_events["asset_id"].isin(res["asset_ids_in_file"])) & (all_events["event_id"] != res["event_id"])]
    contained = []
    for _, oe in same_asset.iterrows():
        oes, oee = pd.Timestamp(oe["event_start"]), pd.Timestamp(oe["event_end"])
        m = (ts >= oes) & (ts <= oee)
        if m.any():
            contained.append({
                "other_event_id": int(oe["event_id"]),
                "other_label": oe["event_label"],
                "rows": int(m.sum()),
                "in_split": tt[m].value_counts().to_dict(),
            })
    res["other_events_inside_file"] = contained

    # ---- missing values
    na = df[sensor_cols].isna()
    col_na = na.mean()
    res["missing_cell_frac"] = float(na.to_numpy().mean())
    res["missing_cell_frac_train"] = float(na[is_train].to_numpy().mean()) if is_train.any() else None
    res["missing_cell_frac_pred"] = float(na[is_pred].to_numpy().mean()) if is_pred.any() else None
    res["rows_with_any_missing"] = int(na.any(axis=1).sum())
    res["rows_all_sensors_missing"] = int(na.all(axis=1).sum())
    res["cols_all_missing"] = col_na[col_na == 1].index.tolist()
    res["n_cols_gt50pct_missing"] = int((col_na > 0.5).sum())
    res["meta_missing"] = {c: int(df[c].isna().sum()) for c in META_COLS if c in cols}

    # ---- invalid / degenerate values
    arr = num.to_numpy(dtype="float64", copy=False)
    res["inf_cells"] = int(np.isinf(arr).sum())
    nun = num.nunique(dropna=True)
    res["constant_cols"] = nun[nun <= 1].index.tolist()

    # ---- duplicates
    res["duplicate_rows_excluding_id"] = int(df.drop(columns=["id"]).duplicated().sum())
    res["duplicate_sensor_vectors"] = int(df[sensor_cols].duplicated().sum())

    colstats = pd.DataFrame({
        "n_rows": n_rows,
        "n_missing": na.sum(),
        "min": num.min().reindex(sensor_cols),
        "max": num.max().reindex(sensor_cols),
        "sum": num.sum().reindex(sensor_cols),
        "n_valid": num.count().reindex(sensor_cols),
        "is_constant": (nun <= 1).reindex(sensor_cols),
        "all_missing": (col_na == 1),
    })
    colstats["event_id"] = res["event_id"]

    # Row fingerprints (sensor values only) keyed by timestamp, for cross-file comparison.
    hashes = pd.Series(pd.util.hash_pandas_object(df[sensor_cols], index=False).to_numpy(), index=ts)
    hashes = hashes[~hashes.index.duplicated()]

    res["seconds"] = round(time.time() - t0, 1)
    return res, colstats, cols, hashes


# --------------------------------------------------------------------------- farm aggregation
def aggregate_columns(colstats: list[pd.DataFrame], sensor_info: dict) -> pd.DataFrame:
    cs = pd.concat([c.rename_axis("column").reset_index() for c in colstats], ignore_index=True)
    g = cs.groupby("column", sort=False)
    out = pd.DataFrame({
        "n_files": g["event_id"].nunique(),
        "missing_frac_overall": g["n_missing"].sum() / g["n_rows"].sum(),
        "files_all_missing": g["all_missing"].sum(),
        "files_constant": g["is_constant"].sum(),
        "min": g["min"].min(),
        "max": g["max"].max(),
        "mean": g["sum"].sum() / g["n_valid"].sum(),
    })
    base = out.index.str.replace(r"_(avg|min|max|std)$", "", regex=True)
    out["sensor"] = base
    out["description"] = [sensor_info.get(b, {}).get("description") for b in base]
    out["unit"] = [sensor_info.get(b, {}).get("unit") for b in base]
    out["is_angle"] = [sensor_info.get(b, {}).get("is_angle") for b in base]
    out["is_counter"] = [sensor_info.get(b, {}).get("is_counter") for b in base]
    return out.reset_index()


def time_overlaps(rows: list[dict], hashes: dict[int, pd.Series]) -> list[dict]:
    """Pairs of datasets from the same turbine whose time spans overlap, and whether
    the sensor values at shared timestamps are identical (same physical records)."""
    out = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["asset_ids_in_file"] != b["asset_ids_in_file"]:
                continue
            s = max(pd.Timestamp(a["ts_min"]), pd.Timestamp(b["ts_min"]))
            e = min(pd.Timestamp(a["ts_max"]), pd.Timestamp(b["ts_max"]))
            if s < e:
                ha, hb = hashes[a["event_id"]], hashes[b["event_id"]]
                common = ha.index.intersection(hb.index)
                same = float((ha.loc[common].to_numpy() == hb.loc[common].to_numpy()).mean()) if len(common) else None
                out.append({
                    "asset_id": int(a["asset_ids_in_file"][0]),
                    "dataset_a": int(a["event_id"]), "dataset_b": int(b["event_id"]),
                    "overlap_days": round((e - s) / pd.Timedelta("1D"), 1),
                    "shared_timestamps": int(len(common)),
                    "identical_values_pct": None if same is None else round(100 * same, 2),
                })
    return out


def unique_turbine_rows(rows: list[dict], hashes: dict[int, pd.Series]) -> dict:
    """Total rows vs unique (turbine, timestamp) pairs across all files of a farm."""
    per_asset: dict[int, set] = {}
    for r in rows:
        per_asset.setdefault(int(r["asset_ids_in_file"][0]), set()).update(hashes[r["event_id"]].index)
    uniq = sum(len(v) for v in per_asset.values())
    return {"rows_total": int(sum(r["n_rows"] for r in rows)), "unique_turbine_timestamps": int(uniq),
            "unique_per_asset": {a: len(v) for a, v in per_asset.items()}}


# --------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def write_report(out: Path, stats: dict, summary: pd.DataFrame) -> None:
    L = ["# CARE to Compare — Stage 1 Inspection Report (auto-generated)", "",
         f"Generated: {stats['generated']}  ", f"Dataset folder: `{stats['data_dir']}`  ",
         "All numbers below were computed by `wtpm/inspect_dataset.py` from the actual files.", ""]
    L += ["## 1. Overview per farm", ""]
    ov = pd.DataFrame([{
        "farm": f, "datasets": s["n_dataset_files"], "anomaly events": s["n_anomaly_events"],
        "normal events": s["n_normal_events"], "turbines": s["n_assets"],
        "columns per file": s["n_cols_set"], "rows total": s["rows_total"],
        "column set identical across files": s["identical_column_sets"],
    } for f, s in stats["farms"].items()])
    L += [md_table(ov), ""]

    L += ["## 2. Metadata consistency", ""]
    for f, s in stats["farms"].items():
        L += [f"**Farm {f}**: event_info columns `{s['event_info_columns_raw']}` (encoding {s['event_info_encoding']}); "
              f"feature_description encoding {s['feature_description_encoding']}; "
              f"{s['n_sensors_described']} sensors described → {s['n_expected_feature_columns']} expected columns; "
              f"columns in data but not described: {len(s['cols_not_in_description'])} {s['cols_not_in_description'][:15]}; "
              f"described but not in data: {len(s['described_not_in_data'])} {s['described_not_in_data'][:15]}; "
              f"angle sensors: {s['n_angle_sensors']}, counter sensors: {s['n_counter_sensors']}.", ""]

    L += ["## 3. Per-dataset summary", ""]
    show = summary[["farm", "event_id", "event_label", "asset", "n_rows", "ts_min", "ts_max", "train_rows", "pred_rows",
                    "event_window_rows", "event_duration_days", "pred_rows_before_event", "window_abnormal_status_pct",
                    "first_abnormal_offset_h", "status_1d_before_start",
                    "abnormal_status_pct", "abnormal_status_episodes", "missing_cell_pct", "n_gaps", "ts_duplicates",
                    "duplicate_rows_excluding_id", "n_constant_cols", "other_events_inside"]]
    L += [md_table(show), ""]

    L += ["## 4. Events per turbine", ""]
    for f, s in stats["farms"].items():
        L += [f"**Farm {f}**: " + ", ".join(f"asset {a}: {v['anomaly']}A/{v['normal']}N" for a, v in s["events_per_asset"].items()), ""]

    L += ["## 5. Same-turbine datasets with overlapping time spans", ""]
    for f, s in stats["farms"].items():
        ov = s["time_overlaps"]
        L += [f"**Farm {f}**: {len(ov)} overlapping pairs"]
        if ov:
            L += ["", md_table(pd.DataFrame(ov))]
        L += [""]

    L += ["## 5b. Row uniqueness (same turbine appears in several files)", ""]
    for f, s in stats["farms"].items():
        u = s["row_uniqueness"]
        L += [f"**Farm {f}**: {u['rows_total']} rows in files → {u['unique_turbine_timestamps']} unique (turbine, timestamp) pairs "
              f"({100 * u['unique_turbine_timestamps'] / u['rows_total']:.1f}%)", ""]

    L += ["## 6. Status code distribution (rows)", ""]
    for f, s in stats["farms"].items():
        L += [f"**Farm {f}**: overall {s['status_counts']}; train {s['status_counts_train']}; prediction {s['status_counts_pred']}; "
              f"inside anomaly windows {s['status_in_anomaly_windows']}; inside normal-event windows {s['status_in_normal_windows']}", ""]

    L += ["## 7. Row counts by event-window membership (counting only — no labels created)", ""]
    for f, s in stats["farms"].items():
        L += [f"**Farm {f}**: {s['rows_in_anomaly_windows']} rows inside anomaly windows of {s['rows_total']} total "
              f"({100 * s['rows_in_anomaly_windows'] / s['rows_total']:.2f}%); "
              f"{s['rows_in_anomaly_windows_pred']} of them in the prediction split.", ""]

    L += ["## 8. Data quality flags", ""]
    for f, s in stats["farms"].items():
        L += [f"**Farm {f}**: unparseable timestamps {s['ts_unparseable']}; non-monotonic files {s['files_non_monotonic']}; "
              f"duplicate timestamps {s['ts_duplicates']}; inf cells {s['inf_cells']}; non-numeric sensor columns {s['non_numeric_sensor_cols']}; "
              f"columns all-missing in ≥1 file: {s['n_cols_all_missing_somewhere']}; columns constant in every file: {s['n_cols_constant_everywhere']}; "
              f"columns >50% missing overall: {s['n_cols_gt50_missing_overall']}; ids are 0..n-1 in all files: {s['all_ids_0_to_n']}; "
              f"train strictly before prediction in all files: {s['all_train_before_pred']}; single split boundary in all files: {s['all_single_split_boundary']}; "
              f"event_start_id matches row id: {s['event_start_id_matches']}/{s['n_dataset_files']}; "
              f"asset matches event_info: {s['asset_matches']}/{s['n_dataset_files']}.", ""]
        if s["negative_in_nonneg_units"]:
            L += [f"  - Negative minimum in columns whose unit is m/s or rpm: {s['negative_in_nonneg_units']}", ""]
    (out / "inspection_report.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only inspection of CARE to Compare.")
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--farms", nargs="*", default=None, help="e.g. A B C (default: all found)")
    args = ap.parse_args(argv)

    data_dir = args.data_dir.resolve()
    out = args.out.resolve()
    if not data_dir.is_dir():
        print(f"Dataset folder not found: {data_dir}", file=sys.stderr)
        return 2
    if out == data_dir or data_dir in out.parents:
        print("Refusing to write inside the dataset folder.", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    farm_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("Wind Farm"))
    if args.farms:
        farm_dirs = [p for p in farm_dirs if p.name.split()[-1] in args.farms]
    log(f"Dataset: {data_dir} | farms: {[p.name for p in farm_dirs]}")

    stats = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "data_dir": str(data_dir),
             "top_level_files": sorted(p.name for p in data_dir.iterdir() if p.is_file()), "farms": {}}
    all_rows = []

    for fdir in farm_dirs:
        farm = fdir.name.split()[-1]
        meta = inspect_metadata(fdir)
        events = meta.pop("events")
        expected_cols = meta.pop("expected_cols")
        sensor_info = meta.pop("sensor_info")
        files = sorted((fdir / "datasets").glob("*.csv"), key=lambda p: int(p.stem))
        log(f"Farm {farm}: {len(files)} dataset files, {len(events)} rows in event_info")

        rows, colstats, col_sets, hashes = [], [], [], {}
        agg = {k: {} for k in ("status_counts", "status_counts_train", "status_counts_pred",
                               "status_in_anomaly_windows", "status_in_normal_windows")}

        def add(d: dict, src: dict) -> None:
            for k, v in src.items():
                d[k] = d.get(k, 0) + v

        for i, f in enumerate(files, 1):
            ev_match = events[events["event_id"] == int(f.stem)]
            ev = ev_match.iloc[0] if len(ev_match) else None
            res, cs, cols, h = inspect_dataset(f, farm, ev, events)
            rows.append(res)
            hashes[res["event_id"]] = h
            colstats.append(cs)
            col_sets.append(tuple(cols))
            add(agg["status_counts"], res["status_counts"])
            add(agg["status_counts_train"], res["status_counts_train"])
            add(agg["status_counts_pred"], res["status_counts_pred"])
            if ev is not None:
                key = "status_in_anomaly_windows" if ev["event_label"] == "anomaly" else "status_in_normal_windows"
                add(agg[key], res["event_window_status"])
            log(f"  [{i}/{len(files)}] {f.name}: {res['n_rows']} rows x {res['n_cols']} cols ({res['seconds']}s)")

        data_cols = [c for c in col_sets[0] if c not in META_COLS] if col_sets else []
        colagg = aggregate_columns(colstats, sensor_info)
        colagg.to_csv(out / f"column_stats_farm_{farm}.csv", index=False)

        neg = colagg[(colagg["unit"].astype(str).isin(["m/s", "rpm"])) & (colagg["min"] < 0)]
        events_per_asset = {}
        for a, g in events.groupby("asset_id"):
            events_per_asset[int(a)] = {"anomaly": int((g["event_label"] == "anomaly").sum()),
                                        "normal": int((g["event_label"] == "normal").sum())}
        anom = [r for r in rows if r.get("event_label") == "anomaly"]
        stats["farms"][farm] = {
            **meta,
            "n_dataset_files": len(files),
            "n_event_info_rows": len(events),
            "files_without_event_info": [r["event_id"] for r in rows if "event_label" not in r],
            "events_without_file": sorted(set(events["event_id"]) - {r["event_id"] for r in rows}),
            "n_anomaly_events": int((events["event_label"] == "anomaly").sum()),
            "n_normal_events": int((events["event_label"] == "normal").sum()),
            "event_label_values": sorted(events["event_label"].unique().tolist()),
            "n_assets": int(events["asset_id"].nunique()),
            "assets": sorted(events["asset_id"].unique().tolist()),
            "events_per_asset": events_per_asset,
            "identical_column_sets": len(set(col_sets)) == 1,
            "n_cols_set": sorted({len(c) for c in col_sets}),
            "cols_not_in_description": [c for c in data_cols if c not in set(expected_cols)],
            "described_not_in_data": [c for c in expected_cols if c not in set(data_cols)],
            "rows_total": int(sum(r["n_rows"] for r in rows)),
            "rows_in_anomaly_windows": int(sum(r["event_window_rows"] for r in anom)),
            "rows_in_anomaly_windows_pred": int(sum(r["event_window_split"].get("prediction", 0) for r in anom)),
            **agg,
            "ts_unparseable": int(sum(r["ts_unparseable"] for r in rows)),
            "files_non_monotonic": [r["event_id"] for r in rows if not r["ts_monotonic_increasing"]],
            "ts_duplicates": int(sum(r["ts_duplicates"] for r in rows)),
            "inf_cells": int(sum(r["inf_cells"] for r in rows)),
            "non_numeric_sensor_cols": sorted({c for r in rows for c in r["non_numeric_sensor_cols"]}),
            "n_cols_all_missing_somewhere": int((colagg["files_all_missing"] > 0).sum()),
            "n_cols_constant_everywhere": int((colagg["files_constant"] == colagg["n_files"]).sum()),
            "n_cols_gt50_missing_overall": int((colagg["missing_frac_overall"] > 0.5).sum()),
            "all_ids_0_to_n": all(r["id_is_0_to_n"] for r in rows),
            "all_train_before_pred": all(r["train_strictly_before_pred"] for r in rows),
            "all_single_split_boundary": all(r["split_label_changes"] == 1 for r in rows),
            "event_start_id_matches": int(sum(r.get("event_start_id_matches_row_id", False) for r in rows)),
            "asset_matches": int(sum(r.get("asset_matches_event_info", False) for r in rows)),
            "negative_in_nonneg_units": {c: float(m) for c, m in zip(neg["column"], neg["min"])},
            "time_overlaps": time_overlaps(rows, hashes),
            "row_uniqueness": unique_turbine_rows(rows, hashes),
        }
        all_rows.extend(rows)

    # per-dataset flat summary
    summary = pd.DataFrame([{
        "farm": r["farm"], "event_id": r["event_id"], "event_label": r.get("event_label"),
        "asset": r["asset_ids_in_file"][0] if len(r["asset_ids_in_file"]) == 1 else r["asset_ids_in_file"],
        "n_rows": r["n_rows"], "n_cols": r["n_cols"], "ts_min": r["ts_min"], "ts_max": r["ts_max"],
        "span_days": r["span_days"],
        "train_rows": r["train_test_counts"].get("train", 0), "pred_rows": r["train_test_counts"].get("prediction", 0),
        "event_start": r.get("event_start"), "event_end": r.get("event_end"),
        "event_duration_days": r.get("event_duration_days"), "event_window_rows": r.get("event_window_rows"),
        "event_window_in_train": r.get("event_window_split", {}).get("train", 0),
        "pred_rows_before_event": r.get("pred_rows_before_event"), "pred_rows_after_event": r.get("pred_rows_after_event"),
        "first_abnormal_offset_h": r.get("first_abnormal_in_window_offset_h"),
        "first_downtime_offset_h": r.get("first_downtime_in_window_offset_h"),
        "status_1d_before_start": r.get("status_1d_before_event_start"),
        "event_start_id_matches": r.get("event_start_id_matches_row_id"),
        "window_abnormal_status_pct": None if r.get("event_window_abnormal_frac") is None else round(100 * r["event_window_abnormal_frac"], 1),
        "abnormal_status_pct": round(100 * r["abnormal_status_frac"], 1),
        "abnormal_status_episodes": r["abnormal_status_episodes"],
        "abnormal_episodes_ge_1h": r["abnormal_episode_len_rows"]["n_ge_6rows_1h"],
        "missing_cell_pct": round(100 * r["missing_cell_frac"], 2),
        "rows_all_sensors_missing": r["rows_all_sensors_missing"],
        "n_gaps": r["n_gaps"], "missing_slots": r["missing_slots"], "max_gap": r["max_gap"],
        "ts_duplicates": r["ts_duplicates"], "duplicate_rows_excluding_id": r["duplicate_rows_excluding_id"],
        "n_constant_cols": len(r["constant_cols"]),
        "other_events_inside": ";".join(f"{o['other_event_id']}({o['other_label'][0]}:{o['in_split']})" for o in r["other_events_inside_file"]),
    } for r in all_rows])
    summary.to_csv(out / "per_dataset_summary.csv", index=False)

    stats["datasets"] = all_rows
    (out / "inspection_stats.json").write_text(json.dumps(to_jsonable(stats), indent=2), encoding="utf-8")
    write_report(out, stats, summary)
    log(f"Done. Outputs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
