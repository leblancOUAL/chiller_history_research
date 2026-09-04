# Static PNG figures.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _month_axis(ax, summary):
    n = len(summary)
    step = max(1, n // 24)
    ax.set_xticks(np.arange(0, n, step))
    ax.set_xticklabels(
        summary["month"].iloc[::step], rotation=45, ha="right", fontsize=8
    )


def make_plots(minute, summary, payload, out_dir):
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1: monthly demand bands + accelerator on fraction
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(summary))
    for col, label in [
        ("chiller_kw_p50", "p50"),
        ("chiller_kw_p95", "p95"),
        ("chiller_kw_p99", "p99"),
        ("chiller_kw_max", "max"),
    ]:
        ax.plot(x, summary[col], label=label)
    ax.set_ylabel("chiller electrical kW")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, summary["accel_on_frac"], color="tab:red", alpha=0.5, label="accel on")
    ax2.set_ylabel("accelerator on fraction", color="tab:red")
    _month_axis(ax, summary)
    fig.tight_layout()
    fig.savefig(plot_dir / "monthly_bands.png", dpi=150)
    plt.close(fig)

    # 2: chiller power distribution, on vs off
    on = minute.loc[minute["accel_on"], "chiller_kw"].dropna()
    off = minute.loc[~minute["accel_on"], "chiller_kw"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    lo = float(min(on.min(), off.min()))
    hi = float(max(on.max(), off.max()))
    bins = np.linspace(lo, hi, 200)
    ax.hist(off, bins=bins, density=True, histtype="step", label=f"off (n={len(off):,})")
    ax.hist(on, bins=bins, density=True, histtype="step", label=f"on (n={len(on):,})")
    ax.set_yscale("log")
    ax.set_xlabel("chiller electrical kW")
    ax.set_ylabel("density (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "on_vs_off_hist.png", dpi=150)
    plt.close(fig)

    # 3: magnet heat (I^2 R) vs chiller power
    s = minute.dropna(subset=["magnet_kw", "chiller_kw"])
    if len(s) > 400_000:
        s = s.sample(400_000, random_state=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    hb = ax.hexbin(s["magnet_kw"], s["chiller_kw"], gridsize=80, mincnt=1, cmap="viridis")
    ax.set_xlabel("magnet power (kW, I^2 R)")
    ax.set_ylabel("chiller electrical kW")
    fig.colorbar(hb, ax=ax, label="minutes")
    fig.tight_layout()
    fig.savefig(plot_dir / "magnet_vs_chiller.png", dpi=150)
    plt.close(fig)

    # 4: duty cycle on nameplate levels (tons), with cumulative coverage
    lv = pd.DataFrame(payload["nameplate_levels"])
    if not lv.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(lv["tons"], lv["fraction_of_time"], width=1.2)
        ax.set_xlabel("nominal cooling at level (tons)")
        ax.set_ylabel("fraction of time")
        ax.set_title("chiller duty cycle (nameplate levels)")
        for _, r in lv.iterrows():
            ax.text(
                r["tons"],
                r["fraction_of_time"],
                f"{r['amps']:.1f} A",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
        ax2 = ax.twinx()
        ax2.step(lv["tons"], lv["cum_fraction_of_time"], where="post", color="tab:red")
        ax2.set_ylabel("cumulative fraction of time", color="tab:red")
        ax2.set_ylim(0, 1.02)
        fig.tight_layout()
        fig.savefig(plot_dir / "chiller_levels.png", dpi=150)
        plt.close(fig)

    # 5: recent overview (last 60 days), hourly
    cutoff = minute.index.max() - pd.Timedelta(days=60)
    tail = minute.loc[minute.index >= cutoff]
    if tail.empty:
        tail = minute
    hourly = tail.resample("1h").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hourly.index, hourly["chiller_kw"], lw=0.8, label="chiller kW (hourly mean)")
    ax.set_ylabel("chiller electrical kW")
    ax2 = ax.twinx()
    ax2.fill_between(hourly.index, 0, hourly["accel_on_frac"], color="tab:red", alpha=0.15)
    ax2.set_ylabel("accelerator on fraction", color="tab:red")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plot_dir / "recent_overview.png", dpi=150)
    plt.close(fig)

    return plot_dir
