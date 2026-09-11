# Physics constants and chiller state inference.
#
# Conventions: currents in amps, terminal voltage in MV, powers in kW.
# Edit the constants below to match nameplate / measured values.

import numpy as np
import pandas as pd

V_LL = 480.0                 # 3-phase line-to-line voltage at the disconnect
POWER_FACTOR = 0.85          # assumed motor power factor (cancels in on/off comparisons)
R_ANALYZER_OHM = 0.16        # analyzer magnet winding resistance
R_SWITCHER_OHM = 0.20        # switcher magnet winding resistance
TERMINAL_ON_MV = 0.25        # terminal voltage above which the accelerator counts as on
COP_ASSUMED = 3.0            # rough chiller COP for cooling-capacity estimates
KW_PER_TON = 3.517           # kW of cooling per ton of refrigeration

SHORT_CYCLE_THRESHOLD_MIN = 10.0  # short cycle defined as < 10 minutes run time

# Assumptions dict required for report.py
ASSUMPTIONS = {
    "COP": COP_ASSUMED,
    "KW_PER_TON": KW_PER_TON
}

COMPRESSORS = {
    "5hp": {"label": "5 hp", "rla_a": 12.8, "tons": 4.0},
    "10hp": {"label": "10 hp", "rla_a": 21.2, "tons": 8.0},
    "20hp": {"label": "20hp", "rla_a": 37.8, "tons": 18.0},
}
NOMINAL_TONS = sum(c["tons"] for c in COMPRESSORS.values())  # 30 tons


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


def interpret_amps(amps, ambient_temp_c=None):
    """
    Maps current level to physical compressor combinations using broad,
    empirically-derived bands, accounting for unmonitored heater loads in winter.
    """
    if np.isnan(amps) or amps < 2.7:
        return "Idle"
        
    # Heaters draw ~4A combined when active. If it's cold, we shift the bucket logic 
    # to absorb that unmonitored baseline bump.
    heater_offset = 4.0 if (ambient_temp_c is not None and ambient_temp_c < 10.0) else 0.0
    effective_amps = amps - heater_offset

    if effective_amps < 11.5:
        return "Fans Only"
    elif 11.5 <= effective_amps < 18.0:
        return "5hp Compressor"
    elif 18.0 <= effective_amps < 27.5:
        return "10hp Compressor"
    elif 27.5 <= effective_amps < 42.0:
        return "5hp + 10hp Compressors"
    elif 42.0 <= effective_amps < 57.0:
        return "20hp Compressor"
    elif 57.0 <= effective_amps <= 70.0:
        return "All Compressors Peak Load"

    return "Over-current"
    