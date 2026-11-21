"""Read-only access to the CARE to Compare files (one farm)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wtpm import config

HASH_KEY_1 = "0123456789123456"  # pandas default key
HASH_KEY_2 = "wtpm-second-key!"  # independent second key -> detects 64-bit hash collisions


def load_event_info(farm_dir: Path = config.FARM_DIR) -> pd.DataFrame:
    ev = pd.read_csv(farm_dir / "event_info.csv", sep=";")
    ev = ev.rename(columns={"asset": "asset_id"})  # Farm A uses "asset"
    ev["event_start"] = pd.to_datetime(ev["event_start"], format=config.TS_FORMAT)
    ev["event_end"] = pd.to_datetime(ev["event_end"], format=config.TS_FORMAT)
    return ev


def assign_folds(events: pd.DataFrame, n_folds: int = config.N_FOLDS, seed: int = config.FOLD_SEED) -> pd.DataFrame:
    """Turbine-grouped, anomaly-stratified fold per turbine. Returns columns asset_id, fold."""
    y = (events["event_label"] == "anomaly").astype(int)
    sgk = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=events.index)
    for k, (_, te) in enumerate(sgk.split(events, y, events["asset_id"])):
        fold.iloc[te] = k
    out = pd.DataFrame({"asset_id": events["asset_id"], "fold": fold}).drop_duplicates()
    if out["asset_id"].duplicated().any():
        raise AssertionError("A turbine was assigned to more than one fold")
    return out.sort_values("asset_id").reset_index(drop=True)


def dataset_path(event_id: int, farm_dir: Path = config.FARM_DIR) -> Path:
    return farm_dir / "datasets" / f"{event_id}.csv"


def read_dataset_file(path: Path, keep_suffixes=config.FEATURE_SUFFIXES_STORED) -> pd.DataFrame:
    """Read one dataset file (read-only).

    Returns metadata + parsed timestamp ``ts`` + two independent 64-bit fingerprints (``fp1``,
    ``fp2``) computed on ALL raw sensor columns + the selected feature columns as float32.
    """
    df = pd.read_csv(path, sep=";")
    sensor_cols = [c for c in df.columns if c not in config.META_COLS]
    raw = df[sensor_cols]
    fp1 = pd.util.hash_pandas_object(raw, index=False, hash_key=HASH_KEY_1).to_numpy()
    fp2 = pd.util.hash_pandas_object(raw, index=False, hash_key=HASH_KEY_2).to_numpy()
    keep = [c for c in sensor_cols if c.endswith(tuple(keep_suffixes))]
    out = df[config.META_COLS].copy()
    out["ts"] = pd.to_datetime(out["time_stamp"], format=config.TS_FORMAT)
    out["fp1"], out["fp2"] = fp1, fp2
    feats = df[keep].astype("float32")
    out = pd.concat([out.drop(columns=["time_stamp"]), feats], axis=1)
    out["train_test"] = out["train_test"].astype(str)
    return out


def clean_sentinels(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Negative wind speeds (sensor failure code, e.g. -14.9 m/s) -> NaN. Returns (df, n_cells_changed)."""
    n = 0
    cols = [c for c in df.columns if c.startswith(config.NEGATIVE_IS_MISSING_PREFIXES)]
    for c in cols:
        if c.endswith("_std"):
            continue  # std of a signal is >= 0 by definition; negative std only arises with the sentinel avg
        m = df[c] < 0
        n += int(m.sum())
        df.loc[m, c] = np.nan
    # std columns of the same sensors: invalid when their avg was a sentinel
    for c in cols:
        if c.endswith("_std"):
            avg = c[:-4] + "_avg"
            if avg in df.columns:
                m = df[avg].isna() & df[c].notna()
                n += int(m.sum())
                df.loc[m, c] = np.nan
    return df, n
