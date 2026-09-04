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

# ---- Chiller nameplate data ----------------------------------------------
# 4 fans at 1.5 hp / 3.5 FLA each; compressors given as (label, RLA amps, tons).
N_FANS = 4
FAN_HP_EACH = 1.5
FAN_FLA_EACH_A = 3.5
FAN_TOTAL_A = N_FANS * FAN_FLA_EACH_A          # 14.0 A with all fans running
COMPRESSORS = (
    {"label": "5 hp", "rla_a": 12.8, "tons": 4.0},
    {"label": "10 hp", "rla_a": 21.2, "tons": 8.0},
    {"label": "20 hp", "rla_a": 37.8, "tons": 18.0},
)
NOMINAL_TONS = sum(c["tons"] for c in COMPRESSORS)   # 30 tons


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


# ---- Chiller levels --------------------------------------------------------
# With no VFDs the current is a step signal. The expected levels are every
# subset of compressors (by RLA), with and without the fan load.

def nameplate_levels(with_fans=True):
    """Expected current levels from the nameplate: list of dicts sorted by amps.

    Each row: compressors, fans, amps, kw (electrical), tons (nominal cooling).
    """
    rlas = np.array([c["rla_a"] for c in COMPRESSORS])
    tons = np.array([c["tons"] for c in COMPRESSORS])
    n = len(COMPRESSORS)
    rows = []
    for k in range(2 ** n):
        mask = np.array([(k >> j) & 1 for j in range(n)], dtype=bool)
        label = "+".join(COMPRESSORS[j]["label"] for j in range(n) if mask[j]) or "none"
        base_a = float(rlas[mask].sum())
        base_tons = float(tons[mask].sum())
        if with_fans:
            amps_list = (base_a, base_a + FAN_TOTAL_A)
            fans_list = ("off", "on")
        else:
            amps_list = (base_a,)
            fans_list = ("off",)
        for amps, fans in zip(amps_list, fans_list):
            rows.append(
                {
                    "compressors": label,
                    "fans": fans,
                    "amps": round(amps, 2),
                    "kw": round(float(chiller_power_kw(amps)), 2),
                    "tons": base_tons,
                }
            )
    rows.sort(key=lambda r: r["amps"])
    return rows


def validate_against_nameplate(detected_levels):
    """Pair each data-detected current level with the nearest nameplate level."""
    expected = np.array([r["amps"] for r in nameplate_levels()], dtype=float)
    out = []
    for d in detected_levels:
        j = int(np.argmin(np.abs(expected - d)))
        out.append(
            {
                "detected_a": round(float(d), 2),
                "nearest_nameplate_a": round(float(expected[j]), 2),
                "diff_a": round(float(d - expected[j]), 2),
            }
        )
    return out


def detect_current_levels(current, max_levels=16):
    """Data-driven detection of discrete chiller current levels (validation use).

    Levels appear as peaks of the histogram. Returns sorted amps array
    (may be empty). Prefer nameplate_levels() for the duty cycle; use this
    to confirm the measured currents actually land on the nameplate levels.
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
