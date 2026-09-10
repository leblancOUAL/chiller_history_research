import sys
from pathlib import Path
import pandas as pd
import numpy as np


def extract_current_plateaus(minutely_path, min_amp_threshold=2.7, delta_threshold=2.2, min_duration_min=1):
    """
    Extracts distinct continuous current levels from minute-level data.
    Calculates the exact transition deltas to isolate equipment signatures.
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
    
    # Calculate step-changes on the FULL dataset to accurately catch the jump from Idle.
    df["amp_delta"] = df["chiller_a"].diff()
    
    active = df[df["chiller_a"] >= min_amp_threshold].copy()
    if active.empty:
        print("No active state data found above threshold.")
        return pd.DataFrame()

    # Calculate time gaps on the ACTIVE slice to detect when it dipped below threshold (to Idle).
    active["time_gap_min"] = active.index.to_series().diff().dt.total_seconds() / 60.0

    amps = active["chiller_a"].to_numpy()
    times = active.index.to_numpy()
    deltas = active["amp_delta"].to_numpy()
    time_gaps = active["time_gap_min"].to_numpy()

    events = []
    seg_start_idx = 0
    seg_sum_amps = amps[0]
    seg_count = 1
    last_step_delta = deltas[0]

    for i in range(1, len(amps)):
        gap = time_gaps[i]
        
        # Trigger boundary if chiller went idle (gap > 2 min) OR a contactor switched (delta >= threshold)
        if gap > 2.0 or abs(deltas[i]) >= delta_threshold:
            duration_min = (times[i - 1] - times[seg_start_idx]) / np.timedelta64(1, "m") + 1.0
            
            if duration_min >= min_duration_min:
                events.append({
                    "start_time": times[seg_start_idx],
                    "end_time": times[i - 1],
                    "duration_min": duration_min,
                    "mean_amps": round(float(seg_sum_amps / seg_count), 2),
                    "initial_step_delta_a": round(float(last_step_delta), 2),
                    "is_startup": bool(gap > 2.0 or seg_start_idx == 0)
                })
            
            seg_start_idx = i
            seg_sum_amps = amps[i]
            seg_count = 1
            last_step_delta = deltas[i]
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
            "initial_step_delta_a": round(float(last_step_delta), 2),
            "is_startup": False
        })

    events_df = pd.DataFrame(events)
    
    # Calculate operating-to-operating state transitions (Mean of State B - Mean of State A)
    if not events_df.empty:
        events_df["prev_mean_amps"] = events_df["mean_amps"].shift(1)
        events_df["state_transition_delta"] = (events_df["mean_amps"] - events_df["prev_mean_amps"]).round(2)
        # Wipe out transitions that were actually fresh startups from Idle
        events_df.loc[events_df["is_startup"], "state_transition_delta"] = np.nan

    return events_df


def summarize_distinct_levels(events_df, level_bin_width=2.2):
    if events_df.empty:
        return pd.DataFrame()
        
    events_df["amp_bucket"] = (events_df["mean_amps"] // level_bin_width) * level_bin_width
    
    summary = events_df.groupby("amp_bucket").agg(
        total_occurrences=("duration_min", "count"),
        total_runtime_hours=("duration_min", lambda x: round(x.sum() / 60.0, 2)),
        avg_amps=("mean_amps", "mean"),
    ).reset_index()

    # Apply physical hardware observations
    def label_state(amps):
        if 53.5 <= amps <= 56.5:
            return "Observed: 20hp + 2 Fans (~55A) [Contains Heater state]"
        return "Unknown"

    summary["heuristic_label"] = summary["avg_amps"].apply(label_state)
    summary = summary.sort_values(by="total_runtime_hours", ascending=False)
    return summary


def analyze_startup_jumps(events_df):
    """Isolates jumps from Idle -> Operating (Usually single compressors)."""
    starts = events_df[events_df["is_startup"]].copy()
    # Group by ~1.0A buckets to cluster similar startups
    starts["startup_bucket"] = (starts["initial_step_delta_a"] // 1.0) * 1.0
    
    summary = starts.groupby("startup_bucket").agg(
        occurrences=("initial_step_delta_a", "count"),
        avg_jump=("initial_step_delta_a", "mean")
    ).reset_index()
    
    # Filter for dominant signatures
    summary = summary[summary["occurrences"] > 500].sort_values("occurrences", ascending=False)
    return summary


def analyze_operating_steps(events_df):
    """Isolates step transitions while operating (Fans or extra compressors)."""
    steps = events_df.dropna(subset=["state_transition_delta"]).copy()
    # Filter minor electrical noise below 1.5A
    steps = steps[abs(steps["state_transition_delta"]) >= 1.5]
    
    steps["step_bucket"] = (steps["state_transition_delta"] // 0.5) * 0.5
    summary = steps.groupby("step_bucket").agg(
        occurrences=("state_transition_delta", "count"),
        avg_step=("state_transition_delta", "mean")
    ).reset_index()
    
    summary = summary[summary["occurrences"] > 1000].sort_values("occurrences", ascending=False)
    return summary


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "chiller_out"
    
    events = extract_current_plateaus(out_dir, delta_threshold=2.2)
    print(f"\nExtracted {len(events):,} continuous active state plateaus.")
    
    levels = summarize_distinct_levels(events, level_bin_width=2.2)
    print("\n=== Discovered Distinct Operating Levels ===")
    print(levels.to_string(index=False))
    
    startups = analyze_startup_jumps(events)
    print("\n=== Common Startup Jumps (Idle -> Operating) ===")
    print("Expected: Single compressor turn-on signatures (Delta = Compressor Draw - Crankcase Heater Drop)")
    print(startups.to_string(index=False))
    
    steps = analyze_operating_steps(events)
    print("\n=== Common Operating Steps (Operating -> Operating) ===")
    print("Expected: Fan toggles or secondary compressor staging")
    print(steps.to_string(index=False))
    
    events.to_csv(f"{out_dir}/extracted_state_events.csv", index=False)
    print(f"\nDetailed event log saved to: {out_dir}/extracted_state_events.csv")
    