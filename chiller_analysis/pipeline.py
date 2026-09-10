# Pipeline: read archives -> per-minute dataset -> short-cycle & empirical stage analysis.

import json
import shutil
import requests
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import find_archives, load_archive
from .physics import (
    NOMINAL_TONS,
    SHORT_CYCLE_THRESHOLD_MIN,
    chiller_power_kw,
    derive_features,
    discover_empirical_stages,
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


def _get_cached_archive(path, cache_dir, quiet=False):
    path = Path(path)
    cache_dir = Path(cache_dir)
    pq = cache_dir / f"{path.stem}.parquet"
    pkl = cache_dir / f"{path.stem}.pkl"
    
    if pq.exists():
        return pd.read_parquet(pq)
    if pkl.exists():
        return pd.read_pickle(pkl)
        
    df = load_archive(path, quiet=quiet)
    if not df.empty:
        try:
            df.to_parquet(pq)
        except (ImportError, ValueError):
            df.to_pickle(pkl)
    return df


def analyze_cycles(minute_df, idle_amp_threshold=15.0):
    df = minute_df.copy()
    is_on = df["chiller_a"] >= idle_amp_threshold

    blocks = (is_on != is_on.shift()).cumsum()
    on_blocks = df[is_on].groupby(blocks[is_on])

    cycles = []
    for _, block in on_blocks:
        start_time = block.index.min()
        end_time = block.index.max()
        duration_min = (end_time - start_time).total_seconds() / 60.0 + 1.0
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


def fetch_ambient_temperature(df):
    """
    Fetches historical hourly temperature data from Open-Meteo for Athens, OH,
    and interpolates it to the minute-level dataframe index.
    """
    if df.empty:
        return df
        
    # Coordinates for Athens, OH
    lat = 39.3292
    lon = -82.1013
    
    start_date = df.index.min().strftime('%Y-%m-%d')
    # Open-Meteo archive requires end date to be up to 1-2 days ago
    end_date = df.index.max().strftime('%Y-%m-%d')
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m"
        f"&timezone=America%2FNew_York"
    )
    
    print(f"Fetching ambient temperature data from Open-Meteo for {start_date} to {end_date}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Create a dataframe from the hourly weather data
        weather_df = pd.DataFrame({
            "time": pd.to_datetime(data["hourly"]["time"]),
            "ambient_temp_c": data["hourly"]["temperature_2m"]
        }).set_index("time")
        
        # Strip timezone awareness to match naive lab time index
        weather_df.index = weather_df.index.tz_localize(None)
        
        print("Merging and interpolating weather data...")
        df = df.join(weather_df, how="left")
        # Interpolate the hourly temperatures down to minute resolution
        df["ambient_temp_c"] = df["ambient_temp_c"].interpolate(method="time")
        
    except Exception as e:
        print(f"Warning: Failed to fetch ambient temperature data: {e}")
        df["ambient_temp_c"] = np.nan
        
    return df


def build(data_dir, out_dir, start_year=2011, resample="1min", seed=0, clear_cache=False):
    out_dir = Path(out_dir)
    cache_dir = out_dir / "cache"
    if clear_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    archives = find_archives(data_dir, start_year)
    if not archives:
        raise SystemExit(f"No archives found in {data_dir} for year >= {start_year}")

    rng = np.random.default_rng(seed)
    minute_frames = []
    chiller_samples = []

    for k, path in enumerate(archives, 1):
        raw_df = _get_cached_archive(path, cache_dir)
        feat = derive_features(raw_df).dropna(subset=REQUIRED)
        if feat.empty:
            continue
        
        numeric = feat.drop(columns=["accel_on"]).resample(resample).mean(numeric_only=True)
        on_frac = feat["accel_on"].astype("float64").resample(resample).mean().rename("accel_on_frac")
        m = numeric.join(on_frac)
        m["accel_on"] = m["accel_on_frac"] >= 0.5
        minute_frames.append(m)

        vals = feat["chiller_a"].to_numpy()
        take = min(vals.size, 100_000)
        chiller_samples.append(rng.choice(vals, size=take, replace=False))
        print(f"[{k}/{len(archives)}] {path.name}: processed")

    minute = pd.concat(minute_frames).sort_index()
    minute = minute[~minute.index.duplicated(keep="first")]

    # Attach ambient temps via API
    minute = fetch_ambient_temperature(minute)

    save_table(minute, out_dir / "minutely")

    cycle_stats, cycles_df = analyze_cycles(minute)
    save_table(cycles_df, out_dir / "cycles")

    # Empirical stage discovery across all sampled data
    all_amps = np.concatenate(chiller_samples) if chiller_samples else np.array([])
    discovered_stages = discover_empirical_stages(all_amps)

    payload = {
        "nominal_tons": NOMINAL_TONS,
        "discovered_empirical_stages": discovered_stages,
        "short_cycling_analysis": cycle_stats,
    }
    (out_dir / "levels.json").write_text(json.dumps(payload, indent=2))
    return minute, payload
    