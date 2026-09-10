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

# ---- Empirical Hardware Data ----
IDLE_BASELINE_A = 1.2          # off/idle control state
N_FANS = 4
FAN_FLA_EACH_A = 2.8          # empirically measured ~2.8 A per fan
FAN_TOTAL_A = N_FANS * FAN_FLA_EACH_A  # 11.2 A total

COMPRESSORS = {
    "5hp": {"label": "5 hp", "rla_a": 12.8, "tons": 4.0},
    "10hp": {"label": "10 hp", "rla_a": 21.2, "tons": 8.0},
    "20hp": {"label": "20hp", "rla_a": 37.8, "tons": 18.0},
}
NOMINAL_TONS = sum(c["tons"] for c in COMPRESSORS.values())  # 30 tons

# Sequential Watlow Controller Stages (Updated with empirical operating draw)
WATLOW_STAGES = [
    {"stage": 0, "label": "Off / Control Base", "compressors": "none", "fans": "off", "amps": 1.2, "tons": 0.0},
    {"stage": 1, "label": "Stage 1 (5hp)", "compressors": "5 hp", "fans": "on (4 fans)", "amps": 25.4, "tons": 4.0},
    {"stage": 2, "label": "Stage 2 (5+10hp)", "compressors": "5 hp + 10 hp", "fans": "on (4 fans)", "amps": 40.0, "tons": 12.0},
    {"stage": 3, "label": "Stage 3 (20hp Solo / Full)", "compressors": "20 hp Solo / All", "fans": "on (4 fans)", "amps": 58.2, "tons": 18.0},
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
    
    # Pass through water temperature if present
    if "water_temp_c" in df:
        out["water_temp_c"] = df["water_temp_c"]
        
    out["chiller_kw"] = chiller_power_kw(out["chiller_a"])
    out["analyzer_kw"] = out["analyzer_a"] ** 2 * R_ANALYZER_OHM / 1000.0
    out["switcher_kw"] = out["switcher_a"] ** 2 * R_SWITCHER_OHM / 1000.0
    out["magnet_kw"] = out["analyzer_kw"] + out["switcher_kw"]
    out["accel_on"] = out["terminal_mv"] > TERMINAL_ON_MV
    return out


def discover_empirical_stages(chiller_amps_array, min_peak_distance_amps=2.0):
    """
    Histogram-based mode detection using logarithmic density scaling to catch
    both steady-state stages and short high-current pulses (e.g. 58–68 A).
    """
    clean_amps = np.asarray(chiller_amps_array, dtype=float)
    clean_amps = clean_amps[np.isfinite(clean_amps) & (clean_amps > IDLE_BASELINE_A + 1.0)]

    if clean_amps.size < 100:
        return []

    # Bin into 0.5 A steps from 0 to 80 A
    bins = np.arange(0, 80.5, 0.5)
    counts, bin_edges = np.histogram(clean_amps, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Smooth histogram across a 2.5 A window
    kernel = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
    smoothed = np.convolve(counts, kernel, mode="same")

    # Logarithmic scaling preserves short-duration high-load spikes
    log_counts = np.log1p(smoothed)
    distance_bins = max(int(min_peak_distance_amps / 0.5), 1)

    peaks, _ = find_peaks(log_counts, distance=distance_bins, prominence=0.4, height=np.log1p(5))

    discovered_modes = []
    for p in peaks:
        amp_val = round(float(bin_centers[p]), 1)
        discovered_modes.append({
            "peak_amps": amp_val,
            "estimated_kw": round(float(chiller_power_kw(amp_val)), 2),
            "probable_equipment": interpret_amps(amp_val),
        })

    return discovered_modes


def interpret_amps(amps):
    """
    Maps an empirical current level to physical compressor and fan combinations
    using measured fan steps (~2.8 A) and empirical stage baselines.
    """
    if amps < 2.7:
        return "Idle / Control Board"

    # Pure Fan Operation (No Compressors Active)
    if 2.7 <= amps < 11.5:
        n_fans = max(1, min(4, round(amps / FAN_FLA_EACH_A)))
        return f"Fans only ({n_fans} fan{'s' if n_fans > 1 else ''} @ ~{round(n_fans * FAN_FLA_EACH_A, 1)} A)"

    # Stage 1 (5 hp compressor ~14.2 A base + fans)
    # Peak runtime cluster at ~25.4 A (5 hp + 4 fans)
    if 11.5 <= amps < 27.5:
        comp_amps = max(0.0, amps - (N_FANS * FAN_FLA_EACH_A))
        n_fans = max(0, min(4, round((amps - 14.2) / FAN_FLA_EACH_A)))
        return f"Stage 1 (5 hp + {n_fans} fan{'s' if n_fans != 1 else ''})"

    # Stage 1 / Stage 2 Transition Region (5 hp + 10 hp ramping)
    # Dominant peak at ~29.9 A
    if 27.5 <= amps < 33.0:
        return "Stage 1 High / Stage 2 Light Load"

    # Stage 2 (5 hp + 10 hp ~28.8 A base + fans)
    # Steady operating range ~33 A – 45 A
    if 33.0 <= amps < 46.0:
        fan_amps = max(0.0, amps - 28.8)
        n_fans = max(0, min(4, round(fan_amps / FAN_FLA_EACH_A)))
        return f"Stage 2 (5+10 hp + {n_fans} fan{'s' if n_fans != 1 else ''})"

    # Heavy Load / Stage 2 Maximum / 20 hp Stage Transition
    if 46.0 <= amps < 56.0:
        return "Stage 2 Heavy / High Thermal Head"

    # 20 hp Solo / Short-Cycling Spikes / Stage 3 Peak
    # Short-duration occurrences up to 68.96 A max
    if 56.0 <= amps <= 70.0:
        return "20 hp Solo / High-Head Spike / Short Cycle State"

    return "Over-current / Transient Spike"
    