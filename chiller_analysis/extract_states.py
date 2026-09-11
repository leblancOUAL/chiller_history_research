import sys
from pathlib import Path
import pandas as pd
import numpy as np


def extract_chronological_states(minutely_path, min_amp_threshold=2.7):
    """
    Traverses the minute-by-minute data chronologically, using step-change
    deltas to track the actual physical state (compressors and fans) rather 
    than relying on absolute amperage buckets.
    """
    path = Path(minutely_path)
    if path.is_dir():
        path = path / "minutely.parquet"
        
    if not path.exists():
        path = path.with_suffix(".pkl")
        if not path.exists():
            raise FileNotFoundError(f"Could not find minutely data at {minutely_path}")

    print(f"Loading data from {path}...")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
    
    # Calculate time gaps and current deltas
    df["time_gap_min"] = df.index.to_series().diff().dt.total_seconds() / 60.0
    df["amp_delta"] = df["chiller_a"].diff()
    df["prev_chiller_a"] = df["chiller_a"].shift(1)
    
    amps = df["chiller_a"].to_numpy()
    deltas = df["amp_delta"].to_numpy()
    gaps = df["time_gap_min"].to_numpy()
    times = df.index.to_numpy()
    
    if "ambient_temp_c" in df.columns:
        ambients = df["ambient_temp_c"].to_numpy()
    else:
        ambients = np.full(len(df), np.nan)
        
    events = []
    
    # State Machine Variables
    current_base_state = "Idle"
    current_fan_offset = 0
    
    seg_start_time = times[0]
    seg_sum_amps = 0.0
    seg_count = 0
    seg_sum_ambient = 0.0
    seg_ambient_count = 0
    
    for i in range(1, len(amps)):
        gap = gaps[i]
        step = deltas[i]
        current_amp = amps[i]
        
        # 1. Check for Telemetry Drops (Desync)
        if gap > 1.5:
            current_base_state = "Desynced (Gap)"
            current_fan_offset = 0
            
        # 2. Check for State Transitions
        # We only trigger a new state if the jump is significant (>1.5A) OR we lost sync
        if abs(step) >= 1.5 or gap > 1.5:
            
            # Save the PREVIOUS plateau before we transition
            if seg_count > 0:
                duration = (times[i-1] - seg_start_time) / np.timedelta64(1, "m") + 1.0
                if duration >= 1.0 and current_base_state != "Idle":
                    events.append({
                        "start_time": seg_start_time,
                        "end_time": times[i-1],
                        "duration_min": duration,
                        "mean_amps": round(float(seg_sum_amps / seg_count), 2),
                        "mean_ambient_c": round(float(seg_sum_ambient / seg_ambient_count), 2) if seg_ambient_count > 0 else np.nan,
                        "min_ambient_c": round(float(np.nanmin(ambients[i-seg_count:i])), 2) if seg_ambient_count > 0 else np.nan,
                        "max_ambient_c": round(float(np.nanmax(ambients[i-seg_count:i])), 2) if seg_ambient_count > 0 else np.nan,
                        "inferred_compressor": current_base_state,
                        "inferred_fans": current_fan_offset
                    })
            
            # Update the State Machine for the NEW plateau
            seg_start_time = times[i]
            seg_sum_amps = current_amp
            seg_count = 1
            
            if not np.isnan(ambients[i]):
                seg_sum_ambient = ambients[i]
                seg_ambient_count = 1
            else:
                seg_sum_ambient = 0.0
                seg_ambient_count = 0
            
            # Logic: Startup from Idle (or recovering from desync)
            if current_amp >= min_amp_threshold and (amps[i-1] < min_amp_threshold or gap > 1.5):
                # We use the absolute current to anchor the initial state, absorbing heater variance
                if 8.0 <= current_amp <= 17.5:
                    current_base_state = "5hp"
                elif 17.5 < current_amp <= 27.5:
                    current_base_state = "10hp"
                elif 27.5 < current_amp <= 42.0:
                    current_base_state = "5hp+10hp"
                elif 42.0 < current_amp <= 57.0:
                    current_base_state = "20hp"
                elif current_amp > 57.0:
                    current_base_state = "All Compressors"
                else:
                    current_base_state = "Unknown"
                
                current_fan_offset = 0 # Reset fan counting on new startup
                
            # Logic: Shutting Down to Idle
            elif current_amp < min_amp_threshold:
                current_base_state = "Idle"
                current_fan_offset = 0
                
            # Logic: Stepping while already Operating
            elif current_base_state not in ["Idle", "Desynced (Gap)"] and current_amp >= min_amp_threshold:
                # Is it a Fan? (~2.8A step)
                if 2.0 <= step <= 3.8:
                    current_fan_offset += 1
                elif -3.8 <= step <= -2.0:
                    current_fan_offset -= 1
                # If it's a massive jump (e.g., >8A), another compressor fired while running
                elif step >= 8.0:
                    if current_base_state == "5hp" and step > 15.0:
                        current_base_state = "5hp+10hp"
                    elif current_base_state == "10hp" and 8.0 <= step <= 15.0:
                        current_base_state = "5hp+10hp"
                    else:
                        current_base_state = "State Shifted (Re-evaluating)"
                elif step <= -8.0:
                    current_base_state = "State Shifted (Re-evaluating)"
                    
        # 3. Accumulate data if stable
        else:
            seg_sum_amps += current_amp
            seg_count += 1
            if not np.isnan(ambients[i]):
                seg_sum_ambient += ambients[i]
                seg_ambient_count += 1

    return pd.DataFrame(events)


def summarize_inferred_states(events_df):
    if events_df.empty:
        return pd.DataFrame()
    
    # Combine compressor state and fan offset for the final label
    events_df["final_state"] = events_df["inferred_compressor"] + " (+" + events_df["inferred_fans"].astype(str) + " fan toggles)"
    
    summary = events_df.groupby("final_state").agg(
        total_occurrences=("duration_min", "count"),
        total_runtime_hours=("duration_min", lambda x: round(x.sum() / 60.0, 2)),
        avg_amps=("mean_amps", "mean"),
        amp_std_dev=("mean_amps", "std"), # Shows how wide the variance is (heaters/weather)
        avg_ambient_c=("mean_ambient_c", "mean"),
        min_ambient_c=("min_ambient_c", "min"),
        max_ambient_c=("max_ambient_c", "max")
    ).reset_index()

    summary["avg_amps"] = summary["avg_amps"].round(1)
    summary["amp_std_dev"] = summary["amp_std_dev"].round(2)
    summary["avg_ambient_c"] = summary["avg_ambient_c"].round(1)
    
    summary = summary.sort_values(by="total_runtime_hours", ascending=False)
    return summary


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "chiller_out"
    
    events = extract_chronological_states(out_dir)
    print(f"\nExtracted {len(events):,} chronologically tracked active plateaus.")
    
    summary = summarize_inferred_states(events)
    print("\n=== Chronological State Machine Summary ===")
    print("Buckets are built by tracking chronological transitions, absorbing heater/weather variance.")
    print(summary.to_string(index=False))
    
    events.to_csv(f"{out_dir}/extracted_state_events.csv", index=False)
    