import sys
from pathlib import Path
import pandas as pd
import numpy as np


def extract_states_and_transitions(minutely_path, min_amp_threshold=2.7, delta_threshold=2.2):
    """
    Extracts continuous operating plateaus and valid step-changes, strictly
    enforcing time-continuity to avoid artifacts from missing data gaps.
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
    
    df["time_gap_min"] = df.index.to_series().diff().dt.total_seconds() / 60.0
    df["amp_delta"] = df["chiller_a"].diff()
    df["prev_chiller_a"] = df["chiller_a"].shift(1)
    
    continuous = df[df["time_gap_min"] < 1.5].copy()
    
    startups = continuous[
        (continuous["prev_chiller_a"] < min_amp_threshold) & 
        (continuous["chiller_a"] >= min_amp_threshold) &
        (continuous["amp_delta"] > 0)
    ].copy()
    
    operating_steps = continuous[
        (continuous["prev_chiller_a"] >= min_amp_threshold) & 
        (continuous["chiller_a"] >= min_amp_threshold) &
        (continuous["amp_delta"].abs() >= 1.5)
    ].copy()
    
    active = df[df["chiller_a"] >= min_amp_threshold].copy()
    amps = active["chiller_a"].to_numpy()
    times = active.index.to_numpy()
    time_gaps = active["time_gap_min"].to_numpy()
    
    if "ambient_temp_c" in active.columns:
        ambients = active["ambient_temp_c"].to_numpy()
    else:
        ambients = np.full(len(active), np.nan)
        
    events = []
    if len(amps) > 0:
        seg_start_idx = 0
        seg_sum_amps = amps[0]
        seg_count = 1
        seg_sum_ambient = ambients[0] if not np.isnan(ambients[0]) else 0.0
        seg_ambient_count = 1 if not np.isnan(ambients[0]) else 0
        
        for i in range(1, len(amps)):
            gap = time_gaps[i]
            step = amps[i] - amps[i-1]
            
            if gap > 1.5 or abs(step) >= delta_threshold:
                duration = (times[i-1] - times[seg_start_idx]) / np.timedelta64(1, "m") + 1.0
                if duration >= 1.0:
                    events.append({
                        "start_time": times[seg_start_idx],
                        "end_time": times[i-1],
                        "duration_min": duration,
                        "mean_amps": round(float(seg_sum_amps / seg_count), 2),
                        "mean_ambient_c": round(float(seg_sum_ambient / seg_ambient_count), 2) if seg_ambient_count > 0 else np.nan,
                    })
                
                seg_start_idx = i
                seg_sum_amps = amps[i]
                seg_count = 1
                seg_sum_ambient = ambients[i] if not np.isnan(ambients[i]) else 0.0
                seg_ambient_count = 1 if not np.isnan(ambients[i]) else 0
            else:
                seg_sum_amps += amps[i]
                seg_count += 1
                if not np.isnan(ambients[i]):
                    seg_sum_ambient += ambients[i]
                    seg_ambient_count += 1
                    
        duration = (times[-1] - times[seg_start_idx]) / np.timedelta64(1, "m") + 1.0
        if duration >= 1.0:
            events.append({
                "start_time": times[seg_start_idx],
                "end_time": times[-1],
                "duration_min": duration,
                "mean_amps": round(float(seg_sum_amps / seg_count), 2),
                "mean_ambient_c": round(float(seg_sum_ambient / seg_ambient_count), 2) if seg_ambient_count > 0 else np.nan,
            })

    return pd.DataFrame(events), startups, operating_steps


def summarize_distinct_levels(events_df, level_bin_width=2.2):
    if events_df.empty:
        return pd.DataFrame()
        
    events_df["amp_bucket"] = (events_df["mean_amps"] // level_bin_width) * level_bin_width
    
    summary = events_df.groupby("amp_bucket").agg(
        total_occurrences=("duration_min", "count"),
        total_runtime_hours=("duration_min", lambda x: round(x.sum() / 60.0, 2)),
        avg_amps=("mean_amps", "mean"),
        avg_ambient_c=("mean_ambient_c", "mean")
    ).reset_index()

    def label_state(row):
        amps = row["avg_amps"]
        amb = row["avg_ambient_c"]
        heater_flag = " (+ Offline Heaters)" if amb < 10.0 else ""
        
        if amps < 8.0:
            return f"Idle / Fans Only{heater_flag}"
        elif 8.0 <= amps < 18.0:
            return f"5hp Compressor + 0-2 Fans{heater_flag}"
        elif 18.0 <= amps < 28.0:
            return f"10hp Compressor + 0-2 Fans{heater_flag}"
        elif 28.0 <= amps < 42.0:
            return f"5hp + 10hp Compressors + 0-4 Fans{heater_flag}"
        elif 42.0 <= amps < 50.0:
            return f"Ambiguous: Heavy 5hp+10hp OR Light 20hp Solo{heater_flag}"
        elif 50.0 <= amps < 57.0:
            return f"20hp Compressor + Fans (Observed Anchor){heater_flag}"
        else:
            return f"All Compressors (Stage 3 Peak){heater_flag}"

    summary["heuristic_label"] = summary.apply(label_state, axis=1)
    summary["avg_ambient_c"] = summary["avg_ambient_c"].round(1)
    summary = summary.sort_values(by="total_runtime_hours", ascending=False)
    return summary

# [analyze_startup_jumps and analyze_operating_steps remain exactly the same]
def analyze_startup_jumps(startups_df):
    if startups_df.empty:
        return pd.DataFrame()
    startups_df["startup_bucket"] = (startups_df["amp_delta"] // 1.0) * 1.0
    summary = startups_df.groupby("startup_bucket").agg(
        occurrences=("amp_delta", "count"),
        avg_jump=("amp_delta", "mean"),
        avg_ambient_c=("ambient_temp_c", "mean")
    ).reset_index()
    summary["avg_ambient_c"] = summary["avg_ambient_c"].round(1)
    summary = summary[summary["occurrences"] > 200].sort_values("occurrences", ascending=False)
    return summary

def analyze_operating_steps(steps_df):
    if steps_df.empty:
        return pd.DataFrame()
    steps_df["step_bucket"] = (steps_df["amp_delta"] // 0.5) * 0.5
    summary = steps_df.groupby("step_bucket").agg(
        occurrences=("amp_delta", "count"),
        avg_step=("amp_delta", "mean"),
        avg_ambient_c=("ambient_temp_c", "mean")
    ).reset_index()
    summary["avg_ambient_c"] = summary["avg_ambient_c"].round(1)
    summary = summary[summary["occurrences"] > 1000].sort_values("occurrences", ascending=False)
    return summary


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "chiller_out"
    events, startups, operating_steps = extract_states_and_transitions(out_dir, delta_threshold=2.2)
    levels = summarize_distinct_levels(events, level_bin_width=2.2)
    print("\n=== Discovered Distinct Operating Levels ===")
    print(levels.to_string(index=False))
    