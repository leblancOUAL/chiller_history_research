# Pipeline: read archives -> per-minute dataset -> summaries and statistics.

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import find_archives, load_archive
from .physics import (
    ASSUMPTIONS,
    chiller_power_kw,
    derive_features,
    detect_current_levels,
    nearest_level_counts,
    stage_fraction,
)

REQUIRED = ["chiller_a", "analyzer_a", "switcher_a", "terminal_mv"]


def save_table(df, base):
    base = Path(base)
    try:
        df.to_parquet(base.with_suffix(".parquet"))
        return base.with_suffix(".parquet")
    except (ImportError, ValueError):
        df.to_pickle(base.with_suffix(".pkl"))
        return base.with_suffix(".pkl")


def load_table(base):
    base = Path(base)
    pq = base.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_pickle(base.with_suffix(".pkl"))


def resample_feats(feat, rule):
    """Per-archive raw features -> regular bins (means), keeping the on-fraction."""
    numeric = feat.drop(columns=["accel_on"]).resample(rule).mean(numeric_only=True)
    on_frac = feat["accel_on"].astype("float64").resample(rule).mean().rename("accel_on_frac")
    out = numeric.join(on_frac)
    out["accel_on"] = out["accel_on_frac"] >= 0.5
    return out


def build(data_dir, out_dir, start_year=2011, resample="1min", seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = find_archives(data_dir, start_year)
    if not archives:
        raise SystemExit(f"No archives found in {data_dir} for year >= {start_year}")
    rng = np.random.default_rng(seed)

    # ---- pass 1: per-archive binned means + a raw chiller sample for level detection
    minute_frames = []
    chiller_samples = []
    for k, path in enumerate(archives, 1):
        feat = derive_features(load_archive(path)).dropna(subset=REQUIRED)
        if feat.empty:
            print(f"[{k}/{len(archives)}] {path.name}: no matched rows, skipping")
            continue
        m = resample_feats(feat, resample)
        minute_frames.append(m)
        vals = feat["chiller_a"].to_numpy()
        take = min(vals.size, 200_000)
        chiller_samples.append(rng.choice(vals, size=take, replace=False))
        print(f"[{k}/{len(archives)}] {path.name}: {len(feat)} rows -> {len(m)} bins")

    minute = pd.concat(minute_frames).sort_index()
    minute = minute[~minute.index.duplicated(keep="first")]
    for c in minute.columns:
        if c != "accel_on":
            minute[c] = minute[c].astype("float32")
    save_table(minute, out_dir / "minutely")
    print(f"combined: {len(minute)} bins, {minute.index[0]} -> {minute.index[-1]}")

    summary = monthly_summary(minute)
    summary.to_csv(out_dir / "monthly_summary.csv", index=False)

    # ---- chiller level detection from the raw sample
    levels = detect_current_levels(np.concatenate(chiller_samples))
    if levels.size == 0:
        level_info = []
        print("warning: no chiller current levels detected")
    else:
        print(f"detected {len(levels)} chiller current levels (A): {np.round(levels, 2)}")

        # ---- pass 2: exact fraction-of-time at each level (re-reads the archives)
        counts = np.zeros(levels.size, dtype=np.int64)
        total = 0
        for path in archives:
            i = load_archive(path, only=("chiller_a",), quiet=True)["chiller_a"].dropna().to_numpy()
            counts += nearest_level_counts(i, levels)
            total += i.size
        occupancy = counts / max(total, 1)
        level_info = [
            {
                "level_a": round(float(l), 3),
                "level_kw": round(float(chiller_power_kw(l)), 2),
                "fraction_of_time": round(float(f), 6),
            }
            for l, f in zip(levels, occupancy)
        ]
    (out_dir / "levels.json").write_text(json.dumps(level_info, indent=2))
    return minute, level_info


def _bin_hours(index):
    d = index.to_series().diff().median()
    return pd.Timedelta(d).total_seconds() / 3600.0 if pd.notna(d) else np.nan


def monthly_summary(minute):
    dt_h = _bin_hours(minute.index)
    rows = []
    for period, chunk in minute.groupby(minute.index.to_period("M")):
        rows.append(
            {
                "month": str(period),
                "bins": len(chunk),
                "accel_on_frac": float(chunk["accel_on"].mean()),
                "chiller_kw_mean": float(chunk["chiller_kw"].mean()),
                "chiller_kw_p50": float(chunk["chiller_kw"].median()),
                "chiller_kw_p95": float(chunk["chiller_kw"].quantile(0.95)),
                "chiller_kw_p99": float(chunk["chiller_kw"].quantile(0.99)),
                "chiller_kw_max": float(chunk["chiller_kw"].max()),
                "magnet_kw_mean": float(chunk["magnet_kw"].mean()),
                "kwh": float(chunk["chiller_kw"].sum()) * dt_h,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_median_diff(a, b, n_boot=2000, seed=0):
    """95% bootstrap CI for median(a) - median(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return [None, None, None]
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        diffs[k] = np.median(rng.choice(a, size=a.size, replace=True)) - np.median(
            rng.choice(b, size=b.size, replace=True)
        )
    lo, med, hi = np.percentile(diffs, [2.5, 50, 97.5])
    return [float(lo), float(med), float(hi)]


def compute_stats(minute, level_info):
    from scipy.stats import mannwhitneyu, spearmanr

    dt_h = _bin_hours(minute.index)
    on = minute[minute["accel_on"]]
    off = minute[~minute["accel_on"]]
    on_kw = on["chiller_kw"].to_numpy(dtype=float)
    off_kw = off["chiller_kw"].to_numpy(dtype=float)

    levels = np.array([lv["level_a"] for lv in level_info], dtype=float)
    on_stage = stage_fraction(on["chiller_a"].to_numpy(dtype=float), levels)
    off_stage = stage_fraction(off["chiller_a"].to_numpy(dtype=float), levels)

    valid = minute.dropna(subset=["magnet_kw", "chiller_kw"])
    rho, p_rho = spearmanr(valid["magnet_kw"], valid["chiller_kw"])
    try:
        _, p_mw = mannwhitneyu(on_kw, off_kw, alternative="two-sided")
    except ValueError:
        p_mw = None

    def pct(a, q):
        return float(np.nanpercentile(a, q))

    ci = bootstrap_median_diff(on_kw, off_kw)
    stats = {
        "coverage_start": str(minute.index.min()),
        "coverage_end": str(minute.index.max()),
        "n_minutes": int(len(minute)),
        "accel_on_fraction": float(minute["accel_on"].mean()),
        "total_kwh": float(minute["chiller_kw"].sum()) * dt_h,
        "chiller_kw_percentiles": {
            str(q): pct(minute["chiller_kw"].to_numpy(dtype=float), q)
            for q in (50, 90, 95, 99, 99.9, 100)
        },
        "on": {
            "n": int(len(on)),
            "median_kw": float(np.nanmedian(on_kw)) if on_kw.size else None,
            "p95_kw": pct(on_kw, 95) if on_kw.size else None,
            "mean_stage_fraction": float(np.nanmean(on_stage)) if on_stage.size else None,
        },
        "off": {
            "n": int(len(off)),
            "median_kw": float(np.nanmedian(off_kw)) if off_kw.size else None,
            "p95_kw": pct(off_kw, 95) if off_kw.size else None,
            "mean_stage_fraction": float(np.nanmean(off_stage)) if off_stage.size else None,
        },
        "median_diff_kw": ci[1],
        "median_diff_ci95": [ci[0], ci[2]],
        "mannwhitney_p": p_mw,
        "spearman_magnet_chiller": {"rho": float(rho), "p": float(p_rho)},
        "levels": level_info,
        "assumptions": ASSUMPTIONS,
    }
    return stats
