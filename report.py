# Human-readable rendering of the statistics dict.


def _fmt(v, digits=2):
    return "n/a" if v is None else f"{v:.{digits}f}"


def print_report(stats):
    p = stats["chiller_kw_percentiles"]
    a = stats["assumptions"]
    lines = []
    lines.append("=" * 72)
    lines.append("Tandem chiller analysis report")
    lines.append("=" * 72)
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
        p = stats["mannwhitney_p"]
        p_txt = f"{p:.3g}" if p > 0 else "< 1e-300 (underflow)"
        lines.append(
            f"  Mann-Whitney p = {p_txt} "
            f"(with this much data, judge by the effect size and CI, not the p-value)"
        )
    sp = stats["spearman_magnet_chiller"]
    lines.append(
        f"  Spearman(magnet kW, chiller kW): rho = {sp['rho']:.3f} (p = {sp['p']:.3g})"
    )
    lines.append("")
    lines.append("Detected chiller current levels (duty cycle):")
    for lv in stats["levels"]:
        lines.append(
            f"  {lv['level_a']:7.2f} A = {lv['level_kw']:7.2f} kW   "
            f"{100 * lv['fraction_of_time']:.2f}% of time"
        )
    text = "\n".join(lines)
    print(text)
    return text
