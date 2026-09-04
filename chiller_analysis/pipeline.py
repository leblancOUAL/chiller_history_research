# Pipeline: read archives -> per-minute dataset -> summaries and statistics.

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import find_archives, load_archive
from .physics import (
    ASSUMPTIONS,
    NOMINAL_TONS,
    chiller_power_kw,
    derive_features,
    detect_current_levels,
    nameplate_levels,
    nearest_level_counts,
    stage_fraction,
    validate_against_nameplate,
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


def resample_feats(feat, rule):
    """Per-archive raw features -> regular bins (means), keeping the on-fraction."""
    numeric = feat.drop(columns=["accel_on"]).resample(rule).mean(numeric_only=True)
    on_frac = feat["accel_on"].astype("float64").resample(rule).mean().rename("accel_on_frac")
    out = numeric.join(on_frac)
    out["accel_on"] = out["accel_on_frac"] >= 0.5
    return out


def build(data_dir, out_dir, start_year=2011, resample="1min", seed=0):
    """Process every archive; returns (minutely DataFrame, levels payload dict)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archives = find_archives(data_dir, start_year)
    if not archives:
        raise SystemExit(f"No archives found in {data_dir} for year >= {start_year}")
    rng = np.random.default_rng(seed)

    # ---- pass 1: per-archive binned means + a raw chiller sample
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

    # ---- pass 2: exact fraction-of-time at each level (re-reads the archives)
    # Duty-cycle levels come from the nameplate; data-driven detection is kept
    # as a cross-check that measured currents land on the nameplate levels.
    nameplate = nameplate_levels()
    np_amps = np.array([r["amps"] for r in nameplate], dtype=float)
    det_amps = detect_current_levels(np.concatenate(chiller_samples))
    if det_amps.size:
        print(f"data-detected current levels (A): {np.round(det_amps, 2)}")

    counts_np = np.zeros(np_amps.size, dtype=np.int64)
    counts_det = np.zeros(max(det_amps.size, 1), dtype=np.int64)
    total = 0
    for path in archives:
        i = load_archive(path, only=("chiller_a",), quiet=True)["chiller_a"].dropna().to_numpy()
        counts_np += nearest_level_counts(i, np_amps)
        if det_amps.size:
            counts_det += nearest_level_counts(i, det_amps)
        total += i.size

    occ = counts_np / max(total, 1)
    cum = np.cumsum(occ)
    for row, f, c in zip(nameplate, occ, cum):
        row["fraction_of_time"] = round(float(f), 6)
        row["cum_fraction_of_time"] = round(float(c), 6)

    detected = [
        {"level_a": round(float(l), 3), "fraction_of_time": round(float(f), 6)}
        for l, f in zip(det_amps, counts_det / max(total, 1))
    ] if det_amps.size else []

    payload = {
        "nominal_tons": NOMINAL_TONS,
        "nameplate_levels": nameplate,
        "detected_levels": detected,
        "validation": validate_against_nameplate(det_amps) if det_amps.size else [],
    }
    (out_dir / "levels.json").write_text(json.dumps(payload, indent=2))
    return minute, payload


def nameplate_amps(payload):
    return np.array([r["amps"] for r in payload["nameplate_levels"]], dtype=float)


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


def compute_stats(minute, payload):
    from scipy.stats import mannwhitneyu, spearmanr

    dt_h = _bin_hours(minute.index)
    on = minute[minute["accel_on"]]
    off = minute[~minute["accel_on"]]
    on_kw = on["chiller_kw"].to_numpy(dtype=float)
    off_kw = off["chiller_kw"].to_numpy(dtype=float)

    levels = nameplate_amps(payload)
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
        "levels": payload,
        "assumptions": ASSUMPTIONS,
    }
    return stats
