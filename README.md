# Tandem chiller power analysis

Reads `tandem_archive_YYYY-*.tgz` files directly (no extraction to disk) and:

1. Converts chiller leg current -> 3-phase electrical power -> approximate cooling load.
2. Compares chiller load with the accelerator ON vs OFF (terminal > 0.25 MV),
   with effect sizes and bootstrap confidence intervals, not just p-values.
3. Computes the duty cycle against the chiller nameplate levels (every
   combination of the 3 compressors +/- the 4 fans, mapped to nominal tons),
   and cross-checks data-detected current levels against nameplate RLA.
4. Produces static plots, a printed report, and an interactive Streamlit app.

## Directory layout (important)

`python -m chiller_analysis ...` only works if your files are arranged like
this and you run the commands from the folder that CONTAINS the
`chiller_analysis/` package directory:

    chiller_history_research/
    |-- chiller_analysis/          <- package: these 8 files must be together here
    |   |-- __init__.py
    |   |-- __main__.py
    |   |-- loader.py
    |   |-- physics.py
    |   |-- pipeline.py
    |   |-- plots.py
    |   |-- report.py
    |   |-- cli.py
    |-- streamlit_app.py
    |-- make_fake_data.py
    |-- requirements.txt
    |-- README.md
    |-- data/                      <- put the tandem_archive_*.tgz files here

If you get `No module named chiller_analysis`, either you are in the wrong
directory (`cd` to the one containing `chiller_analysis/`) or the package
files were flattened out of their folder. Note: `python -m inspect ...` runs
Python's unrelated standard-library `inspect` module - the inspect here is a
subcommand: `python -m chiller_analysis inspect <file>`.

## Install

    pip install -r requirements.txt        # Python 3.10+

## First step: inspect a real archive

    python -m chiller_analysis inspect data/tandem_archive_2020-05-01_0101.tgz

This lists the members of one tarball. The archives have a single
`tandem.time` per tarball (no per-channel time files), so the loader:

- zips channels by line index when counts match `tandem.time` exactly;
- otherwise tries "start" vs "end" alignment and keeps the one where the
  physics agrees (magnets on exactly when the terminal is up - the coherence
  score is printed for both hypotheses).

To see the line counts, time coverage, per-channel ranges, and the alignment
decision for one archive:

    python -m chiller_analysis diagnose data/tandem_archive_2020-05-01_0101.tgz

Paste that output into the conversation if anything looks odd. If your member
names differ from the expected four, adjust `CHANNELS` in
`chiller_analysis/loader.py`.

## Run the full pipeline

    python -m chiller_analysis all --data-dir data --out chiller_out

Subcommands: `inspect`, `build`, `report`, `plots`, `all`.
`build` reads every tarball twice: once to build the 1-minute dataset, once to
get exact level occupancy. With a few GB of archives this takes minutes.

Outputs in `chiller_out/`:

- `minutely.parquet` (or `.pkl`) - 1-minute means of all channels plus derived
  power columns (`chiller_kw`, `analyzer_kw`, `switcher_kw`, `magnet_kw`,
  `accel_on`, `accel_on_frac`)
- `monthly_summary.csv` - per-month statistics
- `levels.json` - nameplate duty-cycle table, data-detected levels, validation
- `report.json` + printed report - key numbers incl. the on/off comparison
  and sizing guidance (capacity covering 95/99/100% of observed time)
- `plots/*.png` - static figures

## Interactive exploration

    streamlit run streamlit_app.py -- chiller_out

## Test with synthetic data

    python make_fake_data.py --out fake_data --years 2011 2012
    python -m chiller_analysis all --data-dir fake_data --out fake_out

## Physics / assumptions

Constants live at the top of `chiller_analysis/physics.py` - edit there.

- Chiller power: P = sqrt(3) * 480 V * I_leg * pf, pf assumed 0.85 (balanced
  3-phase). pf uncertainty scales everything together and cancels in the
  on/off comparison.
- Nameplate: 4 fans x 1.5 hp @ 3.5 FLA (= 14 A total); compressors
  12.8 A RLA / 4 tons (5 hp), 21.2 A / 8 tons (10 hp), 37.8 A / 18 tons
  (20 hp); nominal 30 tons. RLA is a conservative rating, so measured amps
  typically land at or slightly below it - the report's validation section
  quantifies exactly that.
- Analyzer magnet: P = I^2 * 0.16 ohm; switcher magnet: P = I^2 * 0.20 ohm.
- Accelerator ON: terminal voltage > 0.25 MV (negatives clipped to 0 first).
- Cooling estimate: electrical kW * COP (assumed 3.0); tons = kW_cooling / 3.517.
  With the nameplate tons known, the duty-cycle table reports nominal tons
  directly and COP is only needed for the kW<->tons cross-check.
- Timestamps are parsed as dd/mm/yyyy-HH:MM:SS, naive lab time (no timezone).

## Interpreting the results

With a decade of 1-minute data, any real difference will be "statistically
significant" - the engineering question is magnitude. Look at:

- `median_diff_kw` and its bootstrap 95% CI (does the CI include ~0?),
- mean compressor stage fraction on vs off,
- Spearman correlation between magnet I^2*R power and chiller power,
- the duty-cycle table: if the chiller rarely leaves the low stages, the
  accelerator contributes little to the load.

For sizing the replacement: the report lists the capacity (in tons, at the
nameplate level structure) that covers 95%, 99%, and 100% of observed time.
Add margin for ambient extremes, aging, and future load growth; an HVAC
engineer should confirm the final selection.
