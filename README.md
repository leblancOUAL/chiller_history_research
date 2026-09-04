# Tandem chiller power analysis

Reads `tandem_archive_YYYY-*.tgz` files directly (no extraction to disk) and:

1. Converts chiller leg current -> 3-phase electrical power -> approximate cooling load.
2. Compares chiller load with the accelerator ON vs OFF (terminal > 0.25 MV),
   with effect sizes and bootstrap confidence intervals, not just p-values.
3. Detects the chiller's discrete current levels (fans + 3 compressors) and
   computes the duty cycle / fraction of time at each level.
4. Produces static plots, a printed report, and an interactive Streamlit app.

## Install

    pip install -r requirements.txt        # Python 3.10+

## First step: inspect a real archive

    python -m chiller_analysis inspect /path/to/tandem_archive_2024-06.tgz

This lists the members of one tarball. **Check whether per-channel time files
exist** (e.g. `tandem.Chiller_Current.time`). The loader uses them when present
and falls back to line-index alignment against `tandem.time` otherwise. If your
files use different member names, adjust `CHANNELS` and `_own_time_names()` in
`chiller_analysis/loader.py`.

## Run the full pipeline

    python -m chiller_analysis all --data-dir /data/tandem_archives --out chiller_out

Subcommands: `inspect`, `build`, `report`, `plots`, `all`.
`build` reads every tarball twice: once to build the 1-minute dataset, once to
get exact compressor-level occupancy. With ~2 GB of archives this takes minutes,
not hours.

Outputs in `chiller_out/`:

- `minutely.parquet` (or `.pkl`) - 1-minute means of all channels plus derived
  power columns (`chiller_kw`, `analyzer_kw`, `switcher_kw`, `magnet_kw`,
  `accel_on`, `accel_on_frac`)
- `monthly_summary.csv` - per-month statistics
- `levels.json` - detected chiller current levels and fraction-of-time occupancy
- `report.json` + printed report - key numbers including the on/off comparison
- `plots/*.png` - static figures

## Interactive exploration

    streamlit run streamlit_app.py -- chiller_out

## Test with synthetic data

    python make_fake_data.py --out fake_data --years 2011 2012
    python -m chiller_analysis all --data-dir fake_data --out fake_out

## Physics / assumptions

Constants live at the top of `chiller_analysis/physics.py` - edit them there.

- Chiller power: P = sqrt(3) * 480 V * I_leg * pf, pf assumed 0.85 (balanced
  3-phase). pf uncertainty scales everything together and cancels in the
  on/off comparison.
- Analyzer magnet: P = I^2 * 0.16 ohm; switcher magnet: P = I^2 * 0.20 ohm.
- Accelerator ON: terminal voltage > 0.25 MV (negative terminal values are
  clipped to 0 first).
- Cooling estimate: electrical kW * COP (assumed 3.0); tons = kW_cooling / 3.517.
- Compressor duty cycle: the chiller current is a step signal; levels are found
  by peak detection on the current histogram. Effective stage fraction =
  (I - I_idle) / (I_full - I_idle), clipped to [0, 1].
- Timestamps are parsed as dd/mm/yyyy-HH:MM:SS, naive lab time (no timezone).

## Interpreting the on/off comparison

With a decade of 1-minute data, any real difference will be "statistically
significant" - the engineering question is magnitude. Look at:

- `median_diff_kw` and its bootstrap 95% CI (does the CI include ~0?),
- mean compressor stage fraction on vs off,
- Spearman correlation between magnet I^2*R power and chiller power.

If all three are near zero, the data support "chiller load is independent of
accelerator operation" - exactly the evidence you want for sizing.

For sizing the replacement: use p99/p99.9/max electrical demand converted at
your chosen COP, then add margin for ambient extremes, aging, and future load
growth. This tool characterizes the historical load; an HVAC engineer should
confirm the final equipment selection.
