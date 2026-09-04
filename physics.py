# Physics constants and chiller state inference.
#
# Conventions: currents in amps, terminal voltage in MV, powers in kW.
# Edit the constants below to match nameplate / measured values.

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

V_LL = 480.0                 # 3-phase line-to-line voltage at the disconnect
POWER_FACTOR = 0.85          # assumed motor power factor (cancels in on/off comparisons)
R_ANALYZER_OHM = 0.16        # analyzer magnet winding resistance
R_SWITCHER_OHM = 0.20        # switcher magnet winding resistance
TERMINAL_ON_MV = 0.25        # terminal voltage above which the accelerator counts as on
COP_ASSUMED = 3.0            # rough chiller COP for cooling-capacity estimates
KW_PER_TON = 3.517           # kW of cooling per ton of refrigeration

ASSUMPTIONS = {
    "V_LL": V_LL,
    "POWER_FACTOR": POWER_FACTOR,
    "COP": COP_ASSUMED,
    "KW_PER_TON": KW_PER_TON,
    "TERMINAL_ON_MV": TERMINAL_ON_MV,
    "R_ANALYZER_OHM": R_ANALYZER_OHM,
    "R_SWITCHER_OHM": R_SWITCHER_OHM,
}


def chiller_power_kw(leg_amps):
    """Balanced 3-phase power from one leg current: P = sqrt(3) * V_LL * I * pf."""
    return np.sqrt(3.0) * V_LL * np.asarray(leg_amps, dtype=float) * POWER_FACTOR / 1000.0


def derive_features(df):
    """Raw aligned channels -> derived power columns and the accel_on flag."""
    out = pd.DataFrame(index=df.index)
    out["analyzer_a"] = df["analyzer_a"]
    out["switcher_a"] = df["switcher_a"]
    out["chiller_a"] = df["chiller_a"]
    out["terminal_mv"] = df["terminal_mv"].clip(lower=0.0)  # negative terminal = 0
    out["chiller_kw"] = chiller_power_kw(out["chiller_a"])
    out["analyzer_kw"] = out["analyzer_a"] ** 2 * R_ANALYZER_OHM / 1000.0
    out["switcher_kw"] = out["switcher_a"] ** 2 * R_SWITCHER_OHM / 1000.0
    out["magnet_kw"] = out["analyzer_kw"] + out["switcher_kw"]
    out["accel_on"] = out["terminal_mv"] > TERMINAL_ON_MV
    return out


def detect_current_levels(current, max_levels=16):
    """Detect the discrete chiller current levels (fans / compressor combos).

    The chiller has no VFDs, so current is a step signal; levels appear as
    peaks of the histogram. Returns sorted amps array (may be empty).
    """
    i = np.asarray(current, dtype=float)
    i = i[np.isfinite(i)]
    if i.size < 1000:
        return np.array([], dtype=float)
    hist, edges = np.histogram(i, bins="auto")
    centers = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0]
    distance = max(1, int(round(1.5 / width)))  # merge peaks closer than ~1.5 A
    prominence = max(hist.max() * 0.02, 5)
    peaks, _ = find_peaks(hist, prominence=prominence, distance=distance)
    if peaks.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(hist[peaks])[::-1][:max_levels]
    return np.sort(centers[peaks][order])


def nearest_level_counts(current, levels):
    """Exact per-level sample counts (nearest level wins)."""
    i = np.asarray(current, dtype=float)
    i = i[np.isfinite(i)]
    levels = np.asarray(levels, dtype=float)
    if levels.size < 2 or i.size == 0:
        return np.zeros(levels.size, dtype=np.int64)
    idx = np.clip(np.searchsorted(levels, i), 1, levels.size - 1)
    lo = levels[idx - 1]
    hi = levels[idx]
    nearest = np.where(np.abs(i - lo) <= np.abs(i - hi), idx - 1, idx)
    return np.bincount(nearest, minlength=levels.size).astype(np.int64)


def stage_fraction(chiller_a, levels):
    """Effective compressor utilization in [0, 1]: 0 = idle, 1 = full load."""
    a = np.asarray(chiller_a, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if levels.size < 2:
        return np.full(a.shape, np.nan)
    idle, full = levels[0], levels[-1]
    return np.clip((a - idle) / (full - idle), 0.0, 1.0)
