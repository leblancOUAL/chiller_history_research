# Human-readable rendering of the statistics dict.


def _fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def print_report(stats):
    p = stats["chiller_kw_percentiles"]
    a = stats["assumptions"]
    lv = stats["levels"]
    lines = []
    lines.append("=" * 78)
    lines.append("Tandem chiller analysis report")
    lines.append("=" * 78)
    lines.append(
        f"Coverage: {stats['coverage_start']} -> {stats['coverage_end']}  "
        f"({stats['n_minutes']:,} minutes)"
    )
    lines.append(f"Accelerator on: {100 * stats['accel_on_fraction']:.1f}% of the time")
    lines.append(f"Total chiller electrical energy: {stats['total_kwh'] / 1e3:,.1f} MWh")
    lines.append("")
    lines.append("Chiller electrical demand (all data):")
    for q in ("50", "90", "95", "99", "99.9", "100"):
        kw = p[q]
        tons = kw * a["COP"] / a["KW_PER_TON"]
        lines.append(
            f"  p{q:>4}: {kw:8.2f} kW   (~{tons:6.2f} tons cooling at COP={a['COP']:.1f})"
        )
    lines.append("")
    lines.append("Accelerator on vs off:")
    lines.append(
        f"  on : n={stats['on']['n']:,} min, median {_fmt(stats['on']['median_kw'])} kW, "
        f"p95 {_fmt(stats['on']['p95_kw'])} kW, mean stage fraction "
        f"{_fmt(stats['on']['mean_stage_fraction'], 3)}"
    )
    lines.append(
        f"  off: n={stats['off']['n']:,} min, median {_fmt(stats['off']['median_kw'])} kW, "
        f"p95 {_fmt(stats['off']['p95_kw'])} kW, mean stage fraction "
        f"{_fmt(stats['off']['mean_stage_fraction'], 3)}"
    )
    lo, hi = stats["median_diff_ci95"]
    lines.append(
        f"  median difference (on - off): {_fmt(stats['median_diff_kw'], 3)} kW, "
        f"bootstrap 95% CI [{_fmt(lo, 3)}, {_fmt(hi, 3)}]"
    )
    off_med = stats["off"]["median_kw"]
    if off_med:
        rel = 100 * stats["median_diff_kw"] / off_med
        lines.append(f"  relative to off-median: {rel:.1f}%")
    if stats["mannwhitney_p"] is not None:
        pv = stats["mannwhitney_p"]
        p_txt = f"{pv:.3g}" if pv > 0 else "0 (underflow)"
        lines.append(
            f"  Mann-Whitney p = {p_txt} "
            f"(with this much data, judge by the effect size and CI, not the p-value)"
        )
    sp = stats["spearman_magnet_chiller"]
    lines.append(
        f"  Spearman(magnet kW, chiller kW): rho = {sp['rho']:.3f} (p = {sp['p']:.3g})"
    )
    lines.append("")
    lines.append(
        f"Chiller duty cycle (nameplate levels, nominal {lv['nominal_tons']:.0f} tons):"
    )
    lines.append(
        f"  {'compressors':14s} {'fans':4s} {'amps':>7s} {'kW':>8s} {'tons':>6s} "
        f"{'% time':>8s} {'cum %':>8s}"
    )
    for row in lv["nameplate_levels"]:
        lines.append(
            f"  {row['compressors']:14s} {row['fans']:4s} {row['amps']:7.2f} "
            f"{row['kw']:8.2f} {row['tons']:6.1f} "
            f"{100 * row['fraction_of_time']:8.2f} {100 * row['cum_fraction_of_time']:8.2f}"
        )
    lines.append("")
    lines.append("Sizing guidance from observed duty cycle:")
    for target in (0.95, 0.99, 1.0):
        cover = [r for r in lv["nameplate_levels"] if r["cum_fraction_of_time"] >= target]
        if cover:
            r = cover[0]
            lines.append(
                f"  covers {100 * target:5.1f}% of observed time: "
                f"{r['tons']:5.1f} tons ({r['compressors']}, fans {r['fans']}, "
                f"{r['amps']:.1f} A = {r['kw']:.1f} kW elec)"
            )
    if lv["validation"]:
        lines.append("")
        lines.append("Validation: data-detected levels vs nameplate (RLA) levels")
        worst = max(abs(v["diff_a"]) for v in lv["validation"])
        for v in lv["validation"]:
            lines.append(
                f"  detected {v['detected_a']:7.2f} A -> nearest nameplate "
                f"{v['nearest_nameplate_a']:7.2f} A (diff {v['diff_a']:+.2f} A)"
            )
        note = "OK" if worst <= 3.0 else "check - large deviation"
        lines.append(
            f"  worst |diff| = {worst:.2f} A ({note}; measured amps are typically "
            f"at or slightly below RLA, which is a conservative rating)"
        )
    text = "\n".join(lines)
    print(text)
    return text
