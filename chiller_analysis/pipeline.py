# Pipeline: read archives -> per-minute dataset -> short-cycle analysis -> statistics.

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import find_archives, load_archive
from .physics import (
    NOMINAL_TONS,
    SHORT_CYCLE_THRESHOLD_MIN,
    WATLOW_STAGES,
    chiller_power_kw,
    classify_operating_state,
    derive_features,
    nameplate_levels,
)

REQUIRED = ["chiller_a", "analyzer_a", "switcher_a", "terminal_mv"]


def save_table(df, base_):
    base_ = Path(base_)
    try:
        df.to_parquet(base_.with_suffix(".parquet"))
        return base_.with_suffix(".parquet")
    except (ImportError, ValueError):
        df.to_pickle(base_.with_suffix(".pkl"))
        return base_.with_suffix(".pkl")


def load_table(base_):
    base_ = Path(base_)
    pq = base_.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(base_.with_suffix(".pkl"))


def analyze_cycles(minute_df, idle_amp_threshold=15.0):
    """
    Extracts continuous ON blocks (compressors active) and quantifies cycle durations,
    short-cycling frequency (< 10 mins), and average active thermal loads.
    """
    df = minute_df.copy()
    is_on = df["chiller_a"] >= idle_amp_threshold

    # Group contiguous ON blocks
    blocks = (is_on != is_on.shift()).cumsum()
    on_blocks = df[is_on].groupby(blocks[is_on])

    cycles = []
    for _, block in on_blocks:
        start_time = block.index.min()
        end_time = block.index.max()
        duration_min = (end_time - start_time).total_seconds() / 60.0 + 1.0  # include inclusive minute
        mean_kw = block["chiller_kw"].mean()
        
        cycles.append({
            "start": start_time,
            "end": end_time,
            "duration_min": duration_min,
            "mean_kw": mean_kw,
            "is_short_cycle": duration_min < SHORT_CYCLE_THRESHOLD_MIN,
        })

    cycles_df = pd.DataFrame(cycles)
    if cycles_df.empty:
        return {}, cycles_df

    short_cycles = cycles_df[cycles_df["is_short_cycle"]]
    stats = {
        "total_cycles": len(cycles_df),
        "short_cycles_count": len(short_cycles),
        "short_cycle_pct": float(len(short_cycles) / len(cycles_df) * 100.0),
        "median_run_duration_min": float(cycles_df["duration_min"].median()),
        "p10_run_duration_min": float(cycles_df["duration_min"].quantile(0.10)),
        "max_run_duration_min": float(cycles_df["duration_min"].max()),
    }
    return stats, cycles_df


def _get_cached_archive(path, cache_dir, quiet=False):
    """Loads an archive from Parquet/Pickle cache if it exists, otherwise parses it."""
    path = Path(path)
    cache_dir = Path(cache_dir)
    pq = cache_dir / f"{path.stem}.parquet"
    pkl = cache_dir / f"{path.stem}.pkl"
    
    if pq.exists():
        return pd.read_parquet(pq)
    if pkl.exists():
        return pd.read_pickle(pkl)
        
    df = load_archive(path, quiet=quiet)
    
    # Save parsed df to cache for future runs
    if not df.empty:
        try:
            df.to_parquet(pq)
        except (ImportError, ValueError):
            df.to_pickle(pkl)
    return df


def build(data_dir, out_dir, start_year=2011, resample="1min", clear_cache=False):
    """Process every archive; returns (minutely DataFrame, levels payload dict)."""
    out_dir = Path(out_dir)
    cache_dir = out_dir / "cache"
    if clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = find_archives(data_dir, start_year)
    if not archives:
        raise SystemExit(f"No archives found in {data_dir} for year >= {start_year}")

    minute_frames = []
    for k, path in enumerate(archives, 1):
        raw_df = _get_cached_archive(path, cache_dir)
        feat = derive_features(raw_df).dropna(subset=REQUIRED)
        if feat.empty:
            continue
        
        # Resample features
        numeric = feat.drop(columns=["accel_on"]).resample(resample).mean(numeric_only=True)
        on_frac = feat["accel_on"].astype("float64").resample(resample).mean().rename("accel_on_frac")
        m = numeric.join(on_frac)
        m["accel_on"] = m["accel_on_frac"] >= 0.5
        minute_frames.append(m)

    minute = pd.concat(minute_frames).sort_index()
    minute = minute[~minute.index.duplicated(keep="first")]

    # Classify operating states and malfunctions
    minute["operating_state"] = classify_operating_state(minute["chiller_a"])
    save_table(minute, out_dir / "minutely")

    # Cycle and short-cycling analysis
    cycle_stats, cycles_df = analyze_cycles(minute)
    save_table(cycles_df, out_dir / "cycles")

    # Watlow stage duty cycle breakdown
    stage_counts = minute["operating_state"].value_counts(normalize=True).to_dict()

    payload = {
        "nominal_tons": NOMINAL_TONS,
        "watlow_stages": WATLOW_STAGES,
        "stage_occupancy_pct": {k: round(float(v * 100), 2) for k, v in stage_counts.items()},
        "short_cycling_analysis": cycle_stats,
    }
    (out_dir / "levels.json").write_text(json.dumps(payload, indent=2))
    return minute, payload
