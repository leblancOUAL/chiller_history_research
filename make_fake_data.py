# Generate synthetic tandem_archive_*.tgz files for testing the pipeline.
#
# Uses the real chiller nameplate numbers so the nameplate validation in the
# report should show small diffs: 4 fans x 3.5 FLA = 14 A idle, compressors
# 12.8 A (5 hp, 4 t), 21.2 A (10 hp, 8 t), 37.8 A (20 hp, 18 t).
# A latent cooling load with a daily cycle and an accelerator-dependent term
# drives compressor staging with hysteresis.

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

FANS_A = 14.0                                  # 4 fans x 3.5 FLA
COMPRESSOR_A = np.array([12.8, 21.2, 37.8])    # RLA: 5, 10, 20 hp
COMPRESSOR_TONS = np.array([4.0, 8.0, 18.0])
# stage k = first k compressors running; stage up/down load thresholds
STAGE_UP = np.array([0.20, 0.55, 0.85])     # add 5hp, 10hp, 20hp above these
STAGE_DOWN = np.array([0.05, 0.35, 0.65])   # drop the largest stage below these
DT_SECONDS = 10


def simulate_month(rng, start, magnet_coupling):
    t = pd.date_range(start, start + pd.DateOffset(months=1), freq=f"{DT_SECONDS}s", inclusive="left")
    n = len(t)
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60.0

    # accelerator schedule: a handful of beam blocks per month
    accel_on = np.zeros(n, dtype=bool)
    n_blocks = rng.integers(6, 15)
    for s in rng.integers(0, n - 1, size=n_blocks):
        length = int(rng.integers(4, 48) * 3600 / DT_SECONDS)
        accel_on[s : min(n, s + length)] = True

    terminal = np.where(accel_on, rng.uniform(1.5, 8.0, n), 0.0)
    m_off = ~accel_on
    terminal[m_off] = np.where(
        rng.random(int(m_off.sum())) < 0.05, -rng.uniform(0.02, 0.3, int(m_off.sum())), 0.0
    )
    terminal += rng.normal(0, 0.1, n)

    analyzer = np.clip(np.where(accel_on, rng.uniform(35, 95, n), 0.0) + rng.normal(0, 0.5, n), 0, None)
    switcher = np.clip(np.where(accel_on, rng.uniform(15, 60, n), 0.0) + rng.normal(0, 0.4, n), 0, None)
    magnet_kw = analyzer**2 * 0.16 / 1000.0 + switcher**2 * 0.20 / 1000.0

    # latent cooling load index (roughly 0..1.5), daily cycle + magnet heat
    load = (
        0.45
        + 0.18 * np.sin((hour - 14.0) / 24.0 * 2 * np.pi)
        + rng.normal(0, 0.06, n)
        + magnet_coupling * magnet_kw / 4.0
    )
    load = np.clip(load, 0.0, 1.5)

    chiller = np.empty(n)
    stage = 0
    for k in range(n):
        if stage < 3 and load[k] > STAGE_UP[stage]:
            stage += 1
        elif stage > 0 and load[k] < STAGE_DOWN[stage - 1]:
            stage -= 1
        chiller[k] = FANS_A + COMPRESSOR_A[:stage].sum()
    chiller += rng.normal(0, 0.25, n)

    return t, {
        "analyzer": analyzer,
        "Chiller_Current": chiller,
        "switcher": switcher,
        "terminal": terminal,
    }


def ts_text(t):
    return "\n".join(x.strftime("%d/%m/%Y-%H:%M:%S") for x in t) + "\n"


def add_bytes(tf, arcname, text):
    data = text.encode()
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def write_archive(path, t, channels, rng, drop_frac=0.003, per_channel_times=True):
    with tarfile.open(path, "w:gz") as tf:
        add_bytes(tf, "tandem.time", ts_text(t))
        for name, vals in channels.items():
            keep = rng.random(len(vals)) > drop_frac
            add_bytes(tf, f"tandem.{name}", "\n".join(f"{v:.6f}" for v in vals[keep]) + "\n")
            if per_channel_times:
                # truncated files keep their original timestamps only when untruncated
                tt = t[: len(vals)] if len(vals) != len(t) else t
                add_bytes(tf, f"tandem.{name}.time", ts_text(tt[keep]))


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic tandem_archive_*.tgz files.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2011])
    ap.add_argument("--months-per-year", type=int, default=12)
    ap.add_argument("--magnet-coupling", type=float, default=0.3,
                    help="how strongly magnet heat drives the synthetic cooling load, per kW")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-channel-times", action="store_true",
                    help="write only tandem.time (exercises line-index alignment)")
    ap.add_argument("--truncate", choices=["none", "start", "end"], default="none",
                    help="drop lines from the START or END of every channel file "
                         "(tandem.time stays full; exercises auto-alignment)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_written = 0
    for year in args.years:
        for month in range(1, args.months_per_year + 1):
            start = pd.Timestamp(year=year, month=month, day=1)
            t, channels = simulate_month(rng, start, args.magnet_coupling)
            if args.truncate != "none":
                k = int(rng.integers(2000, 8000))  # 5.5-22 h at 10 s sampling
                if args.truncate == "start":
                    channels = {name: v[k:] for name, v in channels.items()}
                else:
                    channels = {name: v[:-k] for name, v in channels.items()}
            name = f"tandem_archive_{year}-{month:02d}.tgz"
            write_archive(out / name, t, channels, rng, per_channel_times=not args.no_per_channel_times)
            n_written += 1
    print(f"wrote {n_written} archives to {out}")


if __name__ == "__main__":
    main()
