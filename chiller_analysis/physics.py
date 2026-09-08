# Physics constants and chiller state inference.
#
# Conventions: currents in amps, terminal voltage in MV, powers in kW.
# Edit the constants below to match nameplate / measured values.

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

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


def discover_empirical_stages(chiller_amps_array, min_peak_distance_amps=3.0):
    """
    Identifies true operating current clusters (KDE density peaks) from 
    chiller_a telemetry to map real compressor + fan staging modes.
    """
    clean_amps = np.asarray(chiller_amps_array, dtype=float)
    clean_amps = clean_amps[np.isfinite(clean_amps) & (clean_amps > 1.0)]

    if clean_amps.size < 100:
        return []

    amp_grid = np.linspace(0, max(clean_amps.max() + 5.0, 100.0), 1000)
    kde = gaussian_kde(clean_amps, bw_method=0.04)
    density = kde(amp_grid)

    bin_width = amp_grid[1] - amp_grid[0]
    distance_bins = max(int(min_peak_distance_amps / bin_width), 1)
    
    peaks, _ = find_peaks(density, distance=distance_bins, prominence=max(density) * 0.03)
    
    discovered_modes = []
    for p in peaks:
        amp_val = round(float(amp_grid[p]), 2)
        discovered_modes.append({
            "peak_amps": amp_val,
            "estimated_kw": round(float(chiller_power_kw(amp_val)), 2),
            "probable_equipment": interpret_amps(amp_val)
        })

    return discovered_modes


def interpret_amps(amps):
    """Maps an empirical current peak to physical compressor and fan states."""
    if amps < 3.0:
        return "Idle / Off"
    if 3.0 <= amps < 15.0:
        n_fans = round(amps / FAN_FLA_EACH_A)
        return f"Fans only ({n_fans} fan{'s' if n_fans > 1 else ''})"
    
    # Sequential Stage 1 (5 hp = 12.8 A + 0-4 fans)
    if 15.0 <= amps < 32.0:
        fan_amps = amps - 12.8
        n_fans = max(0, min(4, round(fan_amps / FAN_FLA_EACH_A)))
        return f"Stage 1 (5 hp + {n_fans} fan{'s' if n_fans != 1 else ''})"
    
    # Sequential Stage 2 (5 hp + 10 hp = 34.0 A + 0-4 fans)
    if 32.0 <= amps < 54.0:
        fan_amps = amps - 34.0
        n_fans = max(0, min(4, round(fan_amps / FAN_FLA_EACH_A)))
        return f"Stage 2 (5+10 hp + {n_fans} fan{'s' if n_fans != 1 else ''})"

    # Sequential Stage 3 (5 hp + 10 hp + 20 hp = 71.8 A + 0-4 fans)
    if 70.0 <= amps < 95.0:
        fan_amps = amps - 71.8
        n_fans = max(0, min(4, round(fan_amps / FAN_FLA_EACH_A)))
        return f"Stage 3 (5+10+20 hp + {n_fans} fan{'s' if n_fans != 1 else ''})"

    # Anomaly / Unclassified
    if 54.0 <= amps < 70.0:
        return "Unusual mode (Possible 20 hp running without 10 hp)"

    return "High load / Over-current"
