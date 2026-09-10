import sys
from pathlib import Path
import pandas as pd
import numpy as np


def extract_current_plateaus(minutely_path, min_amp_threshold=2.7, delta_threshold=2.2, min_duration_min=1):
    """
    Extracts distinct continuous current levels from minute-level data using
    instantaneous point-to-point step changes (|I_t - I_{t-1}| >= delta_threshold).

    Parameters:
    - min_amp_threshold: Cutoff above baseline (e.g. 2.7 A).
    - delta_threshold: Instantaneous jump in Amps (>= 2.2 A) that signals a contactor event.
    - min_duration_min: Minimum continuous duration (in minutes) to be recorded as a stable state.
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
    
    # Filter for active chiller operation above baseline
    active = df[df["chiller_a"] >= min_amp_threshold].copy()
    if active.empty:
        print("No active state data found above threshold.")
        return pd.DataFrame()

    amps = active["chiller_a"].to_numpy()
    times = active.index.to_numpy()

    events = []
    
    # Initialize first segment
    seg_start_idx = 0
    seg_sum_amps = amps[0]
    seg_count = 1
    last_step_delta = 0.0  # Initial jump from off/idle

    for i in range(1, len(amps)):
        time_gap_min = (times[i] - times[i - 1]) / np.timedelta64(1, "m")
        instantaneous_delta = amps[i] - amps[i - 1]

        # Trigger state boundary on missing minute (chiller off) or instantaneous contactor switch
        if time_gap_min > 2.0 or abs(instantaneous_delta) >= delta_threshold:
            duration_min = (times[i - 1] - times[seg_start_idx]) / np.timedelta64(1, "m") + 1.0
            
            if duration_min >= min_duration_min:
                events.append({
                    "start_time": times[seg_start_idx],
                    "end_time": times[i - 1],
                    "duration_min": duration_min,
                    "mean_amps": round(float(seg_sum_amps / seg_count), 2),
                    "min_amps": round(float(amps[seg_start_idx:i].min()), 2),
                    "max_amps": round(float(amps[seg_start_idx:i].max()), 2),
                    "initial_step_delta_a": round(float(last_step_delta), 2),
                })
            
            # Start new segment
            seg_start_idx = i
            seg_sum_amps = amps[i]
            seg_count = 1
            last_step_delta = instantaneous_delta
        else:
            seg_sum_amps += amps[i]
            seg_count += 1

    # Close final segment
    duration_min = (times[-1] - times[seg_start_idx]) / np.timedelta64(1, "m") + 1.0
    if duration_min >= min_duration_min:
        events.append({
            "start_time": times[seg_start_idx],
            "end_time": times[-1],
            "duration_min": duration_min,
            "mean_amps": round(float(seg_sum_amps / seg_count), 2),
            "min_amps": round(float(amps[seg_start_idx:].min()), 2),
            "max_amps": round(float(amps[seg_start_idx:].max()), 2),
            "initial_step_delta_a": round(float(last_step_delta), 2),
        })

    return pd.DataFrame(events)


def summarize_distinct_levels(events_df, level_bin_width=2.2):
    """Groups extracted plateaus into distinct current buckets based on 2.2 A steps."""
    if events_df.empty:
        return pd.DataFrame()
        
    events_df["amp_bucket"] = (events_df["mean_amps"] // level_bin_width) * level_bin_width
    
    summary = events_df.groupby("amp_bucket").agg(
        total_occurrences=("duration_min", "count"),
        total_runtime_hours=("duration_min", lambda x: round(x.sum() / 60.0, 2)),
        avg_amps=("mean_amps", "mean"),
        min_duration_min=("duration_min", "min"),
        median_duration_min=("duration_min", "median"),
        max_duration_min=("duration_min", "max"),
    ).reset_index()

    summary["avg_amps"] = summary["avg_amps"].round(2)
    summary = summary.sort_values(by="total_runtime_hours", ascending=False)
    return summary


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "chiller_out"
    
    events = extract_current_plateaus(out_dir, delta_threshold=2.2)
    print(f"\nExtracted {len(events):,} continuous active state plateaus (> 2.7 A).")
    
    summary = summarize_distinct_levels(events, level_bin_width=2.2)
    print("\n=== Discovered Distinct Operating Levels (Grouped by ~2.2 A steps) ===")
    print(summary.to_string(index=False))
    
    events.to_csv(f"{out_dir}/extracted_state_events.csv", index=False)
    summary.to_csv(f"{out_dir}/extracted_state_summary.csv", index=False)
    print(f"\nDetailed event log saved to: {out_dir}/extracted_state_events.csv")
    