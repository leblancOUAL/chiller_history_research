# Command-line interface: python -m chiller_analysis <command> ...
#
# NOTE: run these commands from the directory that CONTAINS the
# chiller_analysis/ package folder, e.g.:
#
#   cd chiller_history_research
#   python -m chiller_analysis inspect data/tandem_archive_2020-05-01_0101.tgz

import argparse
import json
from pathlib import Path

import pandas as pd

from .loader import inspect_archive, diagnose_archive
from . import pipeline
from .report import print_report
from .plots import make_plots


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="chiller_analysis",
        description="Analyze tandem chiller power from tandem_archive_*.tgz files",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="List members of one archive")
    p.add_argument("archive")

    p = sub.add_parser("diagnose", help="Show line counts and alignment for one archive")
    p.add_argument("archive")

    p = sub.add_parser("build", help="Read all archives and build the downsampled dataset")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="chiller_out")
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--resample", default="1min")
    p.add_argument("--clear-cache", action="store_true")

    p = sub.add_parser("stages", help="Print empirically discovered chiller operating stages")
    p.add_argument("--out", default="chiller_out")

    p = sub.add_parser("report", help="Print the summary report")
    p.add_argument("--out", default="chiller_out")

    p = sub.add_parser("plots", help="Write static PNG plots")
    p.add_argument("--out", default="chiller_out")

    p = sub.add_parser("all", help="build + report + plots")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="chiller_out")
    p.add_argument("--start-year", type=int, default=2011)
    p.add_argument("--clear-cache", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "inspect":
        for name, m in sorted(inspect_archive(args.archive).items()):
            print(f"{name:40s} {m.size / 1e6:9.2f} MB uncompressed")
        return

    if args.command == "diagnose":
        diagnose_archive(args.archive)
        return

    if args.command == "build":
        pipeline.build(args.data_dir, args.out, args.start_year, args.resample, clear_cache=args.clear_cache)
        print(f"\nBuild complete. Output written to {args.out}/")
        return

    if args.command == "stages":
        levels_file = Path(args.out) / "levels.json"
        if not levels_file.exists():
            print(f"Error: {levels_file} not found. Run `build` command first.")
            return
        payload = json.loads(levels_file.read_text())
        print("\n=== Empirically Discovered Operating Modes ===")
        for mode in payload.get("discovered_empirical_stages", []):
            print(f"  • {mode['peak_amps']:5.1f} A ({mode['estimated_kw']:5.1f} kW) -> {mode['probable_equipment']}")
        print()
        return

    if args.command in ("report", "plots"):
        minute = pipeline.load_table(Path(args.out) / "minutely")
        payload = json.loads((Path(args.out) / "levels.json").read_text())
        if args.command == "report":
            stats = pipeline.compute_stats(minute, payload)
            (Path(args.out) / "report.json").write_text(json.dumps(stats, indent=2))
            print_report(stats)
        else:
            summary = pd.read_csv(Path(args.out) / "monthly_summary.csv")
            plot_dir = make_plots(minute, summary, payload, args.out)
            print(f"Wrote plots to {plot_dir}")
        return

    if args.command == "all":
        minute, payload = pipeline.build(args.data_dir, args.out, args.start_year, clear_cache=args.clear_cache)
        summary = pipeline.monthly_summary(minute)
        summary.to_csv(Path(args.out) / "monthly_summary.csv", index=False)
        stats = pipeline.compute_stats(minute, payload)
        (Path(args.out) / "report.json").write_text(json.dumps(stats, indent=2))
        print_report(stats)
        plot_dir = make_plots(minute, summary, payload, args.out)
        print(f"\nWrote plots to {plot_dir}")


if __name__ == "__main__":
    main()
