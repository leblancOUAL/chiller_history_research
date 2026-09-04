# Interactive exploration dashboard.
#
# Run:  streamlit run streamlit_app.py -- <out_dir>
# where <out_dir> is the directory produced by `python -m chiller_analysis build`.

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Tandem chiller analysis", layout="wide")


def parse_args():
    args = sys.argv[1:]
    if "--" in args:
        args = args[args.index("--") + 1:]
    return Path(args[0]) if args else Path("chiller_out")


out_dir = parse_args()


@st.cache_data(show_spinner=False)
def load(out_dir_str):
    out = Path(out_dir_str)
    minute = pd.read_parquet(out / "minutely.parquet")
    payload = json.loads((out / "levels.json").read_text())
    rep = {}
    if (out / "report.json").exists():
        rep = json.loads((out / "report.json").read_text())
    return minute, payload, rep


minute, payload, rep = load(str(out_dir))
minute.index = pd.to_datetime(minute.index)

st.title("Tandem chiller load analysis")

year_min, year_max = int(minute.index.year.min()), int(minute.index.year.max())
years = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))
rule = st.sidebar.selectbox("Display resolution", ["raw (1 min)", "hourly", "daily"], index=1)
rule_map = {"raw (1 min)": None, "hourly": "1h", "daily": "1D"}

sel = minute[(minute.index.year >= years[0]) & (minute.index.year <= years[1])]
r = rule_map[rule]
view = sel if r is None else sel.resample(r).mean(numeric_only=True)
if r is not None:
    view["accel_on"] = view["accel_on_frac"] >= 0.5
if len(view) > 400_000:
    view = view.iloc[:: int(np.ceil(len(view) / 400_000))]

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "On vs off", "Chiller states", "Correlations"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Coverage", f"{minute.index.min():%Y-%m-%d} to {minute.index.max():%Y-%m-%d}")
    if rep:
        c2.metric("Median chiller power", f"{rep['chiller_kw_percentiles']['50']:.1f} kW")
        c3.metric("p99 chiller power", f"{rep['chiller_kw_percentiles']['99']:.1f} kW")
    v = view.reset_index()
    xcol = v.columns[0]
    st.plotly_chart(px.line(v, x=xcol, y="chiller_kw", title="Chiller electrical power"), use_container_width=True)
    st.plotly_chart(px.line(v, x=xcol, y="accel_on_frac", title="Fraction of time accelerator is on"), use_container_width=True)
    st.write("Monthly summary:")
    st.dataframe(pd.read_csv(out_dir / "monthly_summary.csv"))

with tab2:
    parts = []
    if view["accel_on"].any():
        parts.append(view[view["accel_on"]].assign(state="on"))
    if (~view["accel_on"]).any():
        parts.append(view[~view["accel_on"]].assign(state="off"))
    sample = pd.concat(parts)
    if len(sample) > 200_000:
        sample = sample.sample(200_000, random_state=0)
    st.plotly_chart(
        px.histogram(sample, x="chiller_kw", color="state", barmode="overlay",
                     nbins=200, title="Chiller power: accelerator on vs off"),
        use_container_width=True,
    )
    if rep:
        st.subheader("Reported comparison (full dataset, not just selected years)")
        st.json({"on": rep.get("on"), "off": rep.get("off"),
                 "median_diff_kw": rep.get("median_diff_kw"),
                 "median_diff_ci95": rep.get("median_diff_ci95"),
                 "spearman_magnet_chiller": rep.get("spearman_magnet_chiller")})

with tab3:
    lv = pd.DataFrame(payload.get("nameplate_levels", []))
    if not lv.empty:
        st.caption(f"Nominal capacity: {payload.get('nominal_tons', float('nan')):.0f} tons")
        st.plotly_chart(
            px.bar(lv, x="tons", y="fraction_of_time", color="fans",
                   hover_data=["compressors", "amps", "kw"],
                   title="Time spent at each nameplate level"),
            use_container_width=True,
        )
        st.dataframe(lv)
        st.subheader("Validation: detected levels vs nameplate")
        st.dataframe(pd.DataFrame(payload.get("validation", [])))
    else:
        st.info("No levels found - run the full pipeline first.")

with tab4:
    s = view.dropna(subset=["magnet_kw", "chiller_kw"])
    if len(s) > 200_000:
        s = s.sample(200_000, random_state=0)
    st.plotly_chart(
        px.scatter(s, x="magnet_kw", y="chiller_kw", color="accel_on", opacity=0.25,
                   title="Chiller power vs magnet power (I^2 R)"),
        use_container_width=True,
    )
    if rep:
        st.write("Spearman correlation (full dataset):", rep.get("spearman_magnet_chiller"))
