# Read tandem_archive_*.tgz files directly from the tarball, no extraction.
#
# Timestamp format inside tandem.time: dd/mm/yyyy-HH:MM:SS (naive lab time).
# Alignment: per-channel time files are used when present (nearest timestamp
# within ALIGN_TOLERANCE); otherwise channels are aligned by line index against
# tandem.time, with a warning if line counts differ.

import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_RE = re.compile(r"^tandem_archive_(\d{4})-.*\.tgz$")
TS_FORMAT = "%d/%m/%Y-%H:%M:%S"
ALIGN_TOLERANCE = pd.Timedelta("5s")

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
        return {Path(m.name).name: m.size for m in tf.getmembers() if m.isfile()}


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


def load_archive(path, align_tolerance=ALIGN_TOLERANCE, only=None, quiet=False):
    """Load one tarball -> DataFrame indexed by tandem.time.

    only: optional iterable of column names to load (e.g. ("chiller_a",)).
    Rows where every channel is unmatched are dropped; partially matched rows
    keep NaN in the missing channels (the pipeline drops those rows later).
    """
    path = Path(path)
    channels = SENSOR_CHANNELS if only is None else {k: v for k, v in SENSOR_CHANNELS.items() if v in only}
    with tarfile.open(path, "r:gz") as tf:
        members = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
        if "tandem.time" not in members:
            raise ValueError(f"{path.name}: no tandem.time member")
        master = _parse_times(_tokens(tf, members, "tandem.time"))
        master = master[~master.duplicated(keep="first")]

        data = {}
        for basename, col in channels.items():
            if basename not in members:
                if not quiet:
                    print(f"  {path.name}: WARNING missing {basename}, column will be all-NaN")
                continue
            vals = np.array(_tokens(tf, members, basename), dtype=np.float64)

            own = next((n for n in _own_time_names(basename) if n in members), None)
            if own is not None:
                ts = _parse_times(_tokens(tf, members, own))
                s = pd.Series(vals, index=ts)
                s = s[~s.index.duplicated(keep="first")]
                data[col] = s.reindex(master, method="nearest", tolerance=align_tolerance)
                if not quiet:
                    n_matched = int(data[col].notna().sum())
                    print(f"  {path.name}: {col}: {n_matched}/{len(master)} matched via {own}")
            else:
                if len(vals) != len(master) and not quiet:
                    print(
                        f"  {path.name}: WARNING {basename}: {len(vals)} values vs "
                        f"{len(master)} timestamps; aligning by line index"
                    )
                n = min(len(vals), len(master))
                data[col] = pd.Series(vals[:n], index=master[:n]).reindex(master)

        df = pd.DataFrame(data, index=master)
        df = df.dropna(how="all")
        return df
