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

SHORT_CYCLE_THRESHOLD_MIN = 10.0  # short cycle defined as < 10 minutes run time
BUFFER_GALLONS = 400.0            # 350-gal tank + 50-gal main piping

# ---- Nameplate Data ----
N_FANS = 4
FAN_FLA_EACH_A = 3.5
FAN_TOTAL_A = N_FANS * FAN_FLA_EACH_A  # 14.0 A total

COMPRESSORS = {
    "5hp": {"label": "5 hp", "rla_a": 12.8, "tons": 4.0},
    "10hp": {"label": "10 hp", "rla_a": 21.2, "tons": 8.0},
    "20hp": {"label": "20 hp", "rla_a": 37.8, "tons": 18.0},
}
NOMINAL_TONS = sum(c["tons"] for c in COMPRESSORS.values())  # 30 tons

# Sequential Watlow Controller Stages
WATLOW_STAGES = [
    {"stage": 0, "label": "Off", "compressors": "none", "fans": "off", "amps": 0.0, "tons": 0.0},
    {"stage": 1, "label": "Stage 1 (5hp)", "compressors": "5 hp", "fans": "on", "amps": 26.8, "tons": 4.0},
    {"stage": 2, "label": "Stage 2 (5+10hp)", "compressors": "5 hp + 10 hp", "fans": "on", "amps": 48.0, "tons": 12.0},
    {"stage": 3, "label": "Stage 3 (5+10+20hp)", "compressors": "5 hp + 10 hp + 20 hp", "fans": "on", "amps": 85.8, "tons": 30.0},
]


def chiller_power_kw(leg_amps):
    """Balanced 3-phase power from leg current."""
    return np.sqrt(3.0) * V_LL * np.asarray(leg_amps, dtype=float) * POWER_FACTOR / 1000.0


def derive_features(df):
    """Raw aligned channels -> derived power columns and accelerator state."""
    out = pd.DataFrame(index=df.index)
    out["analyzer_a"] = df["analyzer_a"]
    out["switcher_a"] = df["switcher_a"]
    out["chiller_a"] = df["chiller_a"]
    out["terminal_mv"] = df["terminal_mv"].clip(lower=0.0)
    out["chiller_kw"] = chiller_power_kw(out["chiller_a"])
    out["analyzer_kw"] = out["analyzer_a"] ** 2 * R_ANALYZER_OHM / 1000.0
    out["switcher_kw"] = out["switcher_a"] ** 2 * R_SWITCHER_OHM / 1000.0
    out["magnet_kw"] = out["analyzer_kw"] + out["switcher_kw"]
    out["accel_on"] = out["terminal_mv"] > TERMINAL_ON_MV
    return out


# ---- Chiller levels --------------------------------------------------------
# With no VFDs the current is a step signal. The expected levels are every
# subset of compressors (by RLA), with and without the fan load.

def nameplate_levels():
    """Returns sequential Watlow stage expectations."""
    rows = []
    for s in WATLOW_STAGES:
        rows.append({
            "stage": s["stage"],
            "compressors": s["compressors"],
            "fans": s["fans"],
            "amps": s["amps"],
            "kw": round(float(chiller_power_kw(s["amps"])), 2),
            "tons": s["tons"],
        })
    return rows


def classify_operating_state(amps_series):
    """
    Classifies currents into:
      - Sequential Watlow Stages (0, 1, 2, 3)
      - Out-of-Order / Malfunctioning states (e.g. 10hp or 20hp running without 5hp)
    """
    a = np.asarray(amps_series, dtype=float)
    out = np.full(a.shape, "Unknown", dtype=object)

    # Thresholds around expected sequential stages (+/- 3.5 A)
    out[a < 5.0] = "Stage 0 (Off)"
    out[(a >= 23.0) & (a <= 31.0)] = "Stage 1 (5hp)"
    out[(a >= 44.0) & (a <= 52.0)] = "Stage 2 (5+10hp)"
    out[(a >= 81.0) & (a <= 91.0)] = "Stage 3 (5+10+20hp)"

    # Malfunctions / Out-of-order combinations
    out[(a >= 33.0) & (a <= 39.0)] = "Malfunction: 10hp only"
    out[(a >= 50.0) & (a <= 56.0)] = "Malfunction: 20hp only"
    out[(a >= 71.0) & (a <= 77.0)] = "Malfunction: 10+20hp (5hp down)"

    return out
