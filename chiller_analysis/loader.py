# Read tandem_archive_*.tgz files directly from the tarball, no extraction.
#
# Timestamp format inside tandem.time: dd/mm/yyyy-HH:MM:SS (naive lab time).
#
# Alignment rules:
#   - per-channel time files are used when present (nearest timestamp within
#     ALIGN_TOLERANCE);
#   - otherwise, if every channel has exactly len(tandem.time) lines, channels
#     are zipped by line index;
#   - otherwise (single tandem.time, unequal line counts), the joint alignment
#     is chosen between "start" and "end" by operational coherence: under the
#     correct alignment the magnets are on almost exactly when the terminal
#     voltage is up. The chosen mode and both scores are printed.
# Rows where every channel is unmatched are dropped; partially matched rows
# keep NaN in the missing channels (the pipeline drops those rows later).

import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_RE = re.compile(r"^tandem_archive_(\d{4})-.*\.tgz$")
TS_FORMAT = "%d/%m/%Y-%H:%M:%S"
ALIGN_TOLERANCE = pd.Timedelta("5s")
TERMINAL_ON_MV = 0.25  # duplicated from physics.py to keep the loader standalone

# tarball member basename -> column name
CHANNELS = {
    "tandem.time": "time",
    "tandem.analyzer": "analyzer_a",
    "tandem.Chiller_Current": "chiller_a",
    "tandem.switcher": "switcher_a",
    "tandem.terminal": "terminal_mv",
}
SENSOR_CHANNELS = {k: v for k, v in CHANNELS.items() if k != "tandem.time"}


def find_archives(data_dir, start_year=2011):
    """Sorted list of archive files for years >= start_year."""
    data_dir = Path(data_dir)
    out = []
    for p in sorted(data_dir.glob("tandem_archive_*-*.tgz")):
        m = ARCHIVE_RE.match(p.name)
        if m and int(m.group(1)) >= start_year:
            out.append(p)
    return out


def inspect_archive(path):
    """{member basename: uncompressed size in bytes} for one tarball."""
    with tarfile.open(path, "r:gz") as tf:
        return {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}


def _tokens(tf, members, basename):
    m = members.get(basename)
    if m is None:
        raise KeyError(f"member {basename!r} not found in tarball")
    return tf.extractfile(m).read().decode("ascii", errors="replace").split()


def _parse_times(tokens):
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(tokens), format=TS_FORMAT))


def _own_time_names(basename):
    # candidate per-channel time-file names, e.g. tandem.Chiller_Current.time
    stem = basename.split(".", 1)[-1]
    return (f"{basename}.time", f"{stem}.time")


def _coherence_score(frame):
    """How strongly magnet activity coincides with the accelerator being on.

    Near 1 under correct alignment; drops toward 0 (or negative) when channels
    are shifted relative to each other. Returns 0 if the needed channels are
    missing.
    """
    if "terminal_mv" not in frame:
        return 0.0
    term = frame["terminal_mv"].to_numpy(dtype=float)
    term = np.where(np.isfinite(term), term, 0.0)
    accel_on = np.clip(term, 0.0, None) > TERMINAL_ON_MV
    mag_active = np.zeros(len(term), dtype=bool)
    used = False
    for col in ("analyzer_a", "switcher_a"):
        if col in frame:
            v = frame[col].to_numpy(dtype=float)
            finite = np.isfinite(v)
            mx = np.nanmax(v) if finite.any() else 0.0
            thr = 0.005 * mx if mx > 0 else np.inf
            mag_active |= finite & (v > thr)
            used = True
    if not used:
        return 0.0
    if not accel_on.any():
        return 0.0
    # Per-block scoring: for each contiguous accelerator-on block, do the
    # magnets run for the central half? Gross misalignment (a channel covering
    # the wrong part of the month) kills whole blocks and scores near 0;
    # correct alignment scores near 1. Far more sensitive than a global
    # duty-cycle coincidence for detecting large shifts.
    a = np.concatenate(([False], accel_on, [False]))
    d = np.diff(a.astype(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    hits = 0
    for s, e in zip(starts, ends):
        length = e - s
        i0 = s + length // 4
        i1 = e - length // 4
        if i1 <= i0:
            i0, i1 = s, e
        if mag_active[i0:i1].mean() > 0.5:
            hits += 1
    return hits / starts.size


def _count_tolerance(master_n):
    # A few missing lines at an archiver edge are immaterial (20 lines is
    # ~10 minutes out of a month of ~70k samples), so treat near-matches as
    # exact and zip directly instead of guessing start vs end.
    return max(10, int(0.001 * master_n))


def _joint_align(master, raw, quiet=False, where=""):
    """Align {col: values} to master with a single tandem.time file.

    If counts match the master exactly (or within a few lines), zip by index.
    Only for grossly unequal counts does it compare the "start" and "end"
    hypotheses with the coherence score and keep the winner.
    """
    counts = {c: len(v) for c, v in raw.items()}
    if not raw:
        return pd.DataFrame(index=master)
    tol = _count_tolerance(len(master))
    if (
        all(abs(n - len(master)) <= tol for n in counts.values())
        and max(counts.values()) - min(counts.values()) <= tol
    ):
        n = min(min(counts.values()), len(master))
        if not quiet and len(set(counts.values())) > 1:
            print(
                f"  {where}: line counts {dict(counts)} vs {len(master)} timestamps "
                f"(within {tol}) - direct alignment"
            )
        idx = master[:n]
        return pd.DataFrame({c: pd.Series(v[:n], index=idx) for c, v in raw.items()})

    n = min([len(master)] + list(counts.values()))
    scores = {}
    frames = {}
    for mode in ("start", "end"):
        idx = master[:n] if mode == "start" else master[-n:]
        frame = {
            c: pd.Series(v[:n] if mode == "start" else v[-n:], index=idx)
            for c, v in raw.items()
        }
        frames[mode] = frame
        scores[mode] = _coherence_score(frame)
    best = max(scores, key=scores.get)
    if not quiet:
        print(
            f"  {where}: unequal line counts {counts} vs {len(master)} timestamps; "
            f"coherence start={scores['start']:.3f} end={scores['end']:.3f} "
            f"-> aligning to {best}"
        )
    return pd.DataFrame(frames[best])


def load_archive(path, align_tolerance=ALIGN_TOLERANCE, only=None, quiet=False):
    """Load one tarball -> DataFrame indexed by tandem.time.

    only: optional iterable of column names to load (e.g. ("chiller_a",)).
    """
    path = Path(path)
    channels = SENSOR_CHANNELS if only is None else {k: v for k, v in SENSOR_CHANNELS.items() if v in only}
    with tarfile.open(path, "r:gz") as tf:
        members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
        if "tandem.time" not in members:
            raise ValueError(f"{path.name}: no tandem.time member")
        master = _parse_times(_tokens(tf, members, "tandem.time"))
        master = master[~master.duplicated(keep="first")]

        raw = {}
        own_map = {}
        for basename, col in channels.items():
            if basename not in members:
                if not quiet:
                    print(f"  {path.name}: WARNING missing {basename}, column will be all-NaN")
                continue
            vals = np.array(_tokens(tf, members, basename), dtype=np.float64)
            own = next((nm for nm in _own_time_names(basename) if nm in members), None)
            if own is not None:
                ts = _parse_times(_tokens(tf, members, own))
                s = pd.Series(vals, index=ts)
                s = s[~s.index.duplicated(keep="first")]
                raw[col] = s.reindex(master, method="nearest", tolerance=align_tolerance)
                own_map[col] = own
            else:
                raw[col] = vals

        if own_map and len(own_map) == len(raw):
            df = pd.DataFrame(raw, index=master)
        elif own_map:
            # mixed: per-channel times where available, joint alignment for the rest
            fixed = {c: v for c, v in raw.items() if c in own_map}
            rest = {c: v for c, v in raw.items() if c not in own_map}
            df_rest = _joint_align(master, rest, quiet=quiet, where=path.name)
            df = pd.concat([pd.DataFrame(fixed, index=master), df_rest], axis=1)
        else:
            df = _joint_align(master, raw, quiet=quiet, where=path.name)

        df = df.dropna(how="all")
        return df


def _fmt_delta(td):
    """Timedelta -> '38.4 s' or '9.3 min' (no days)."""
    s = float(pd.Timedelta(td).total_seconds())
    return f"{s:.1f} s" if s < 120 else f"{s / 60:.1f} min"


def diagnose_archive(path, basenames=None):
    """Print line counts, time coverage, per-channel stats, and the alignment
    decision for one tarball - run this on a real file and paste the output."""
    path = Path(path)
    basenames = basenames or list(SENSOR_CHANNELS)
    with tarfile.open(path, "r:gz") as tf:
        members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
        master = _parse_times(_tokens(tf, members, "tandem.time"))
        dts = master.to_series().diff().dropna()
        print(f"tandem.time: {len(master)} timestamps, {master[0]} -> {master[-1]}")
        if len(dts):
            print(
                f"  median spacing: {_fmt_delta(dts.median())} "
                f"(min {_fmt_delta(dts.min())}, max {_fmt_delta(dts.max())})"
            )
        raw = {}
        for basename in basenames:
            if basename not in members:
                print(f"{basename:30s} MISSING")
                continue
            vals = np.array(_tokens(tf, members, basename), dtype=np.float64)
            col = SENSOR_CHANNELS[basename]
            raw[col] = vals
            fin = vals[np.isfinite(vals)]
            print(
                f"{basename:30s} {len(vals):>9d} lines  "
                f"min {fin.min():.3g}  max {fin.max():.3g}  mean {fin.mean():.4g}"
            )
        print()
        counts = {c: len(v) for c, v in raw.items()}
        tol = _count_tolerance(len(master))
        if counts and all(abs(n - len(master)) <= tol for n in counts.values()):
            print(
                f"line counts within {tol} of tandem.time ({dict(counts)}) - "
                f"direct index alignment; edge truncation is negligible"
            )
        else:
            n = min([len(master)] + list(counts.values())) if counts else 0
            for mode in ("start", "end"):
                idx = master[:n] if mode == "start" else master[-n:]
                frame = {
                    c: pd.Series(v[:n] if mode == "start" else v[-n:], index=idx)
                    for c, v in raw.items()
                }
                print(f"coherence ({mode:5s}): {_coherence_score(frame):.3f}")
