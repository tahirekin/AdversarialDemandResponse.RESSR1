"""LaTeX tables and number macros for Section 7.4, generated from the CSVs.

  python3 tables_tex.py            # writes ../files/sec74/*.tex and the figure

Every number in Section 7.4 comes from here: tables are \\input, and numbers in
the prose are macros defined in sec74/numbers.tex. Nothing is typed by hand.
"""
from __future__ import annotations

import os
import pickle
import shutil

import numpy as np
import pandas as pd

import calibrate
import data
import tables as T
from config import BASE, STRATEGIES, TYPES, episode_starts

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("RESS_TEX_DIR", os.path.join(os.path.dirname(HERE), "files", "sec74"))
LAB = {"no_defense": "No defense", "static_rule": "Static rule", "ara_static": r"\ARA{}-static",
       "ara_dddas": r"\ARA{}--\DDDAS{}", "benchmark": "Complete information"}
TLAB = {"honest": "Honest", "cautious": "Type 1 (cautious)", "moderate": "Type 2 (moderate)",
        "aggressive": "Type 3 (aggressive)"}
SENS_LAB = {"base": "Base settings", "L10": r"$L=10\%$", "L30": r"$L=30\%$",
            "CD25": r"$C_D=\$25$", "CD400": r"$C_D=\$400$",
            "sig025": r"$\sigma=0.25$", "sig1": r"$\sigma=1$", "sig_inf": r"$\sigma=\infty$ (feedback only)",
            "sg_half": r"$\sigma_g$ halved", "sg_double": r"$\sigma_g$ doubled",
            "s1_10": r"$\gamma_1=10\%$", "s1_40": r"$\gamma_1=40\%$"}
MAC = {}


YW = {2016: "Sixteen", 2017: "Seventeen", 2023: "TwentyThree", 2024: "TwentyFour", 2025: "TwentyFive"}


def mac(name, value):
    for y, w in YW.items():
        name = name.replace(str(y), w)
    if any(ch.isdigit() for ch in name):
        raise ValueError(f"macro name with digit: {name}")
    MAC[name] = value


def money(x):
    s = f"{abs(x):,.0f}"
    return ("$-$" if x < 0 else "") + r"\$" + s


def ci(m, lo, hi, scale=1.0, fmt="{:,.0f}", dollar=True):
    f = (lambda v: money(v * scale)) if dollar else (lambda v: ("$-$" if v * scale < 0 else "") + fmt.format(abs(v * scale)))
    star = "" if (lo <= 0 <= hi) else r"$^{*}$"
    return f"{f(m)}{star} [{f(lo)}, {f(hi)}]"


def write(name, body):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write(body)


def tab_strategies(tb):
    t = tb.groupby(["true_type", "strategy"])[T.METRICS].mean()
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"True type & Strategy & $\Delta\LOLE{}$ & of which attack & Fraud & Audit & Correction & Total \\",
             r" & & ($10^{-3}$\,h) & ($10^{-3}$\,h) & (\$) & (\$) & (\$) & (\$) \\", r"\midrule"]
    for tt in TYPES:
        for i, s in enumerate(STRATEGIES):
            r = t.loc[(tt, s)]
            lines.append(f"{TLAB[tt] if i == 0 else ''} & {LAB[s]} & {1e3 * r.dLOLE:.2f} & {1e3 * r.dLOLE_attack:.2f} & "
                         f"{r.fraud:,.0f} & {r.audit_cost:,.0f} & {r.correction_cost:,.0f} & {r.total_cost:,.0f} \\\\")
        lines.append(r"\addlinespace" if tt != TYPES[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("tab_strategies.tex", "\n".join(lines) + "\n")
    for tt in TYPES:
        for s in STRATEGIES:
            mac(f"tot{tt}{s.replace('_', '')}", f"{t.loc[(tt, s)].total_cost:,.0f}")


def tab_paired(tb):
    pairs = [("ara_dddas", "ara_static"), ("ara_dddas", "static_rule"),
             ("ara_dddas", "no_defense"), ("benchmark", "ara_dddas")]
    mets = [("total_cost", "Total cost (\\$)", 1.0, True), ("fraud", "Fraud (\\$)", 1.0, True),
            ("dLOLE", r"$\Delta\LOLE{}$ ($10^{-3}$\,h)", 1e3, False)]
    lines = [r"\begin{tabular}{ll" + "c" * len(TYPES) + "}", r"\toprule",
             "Comparison & Measure & " + " & ".join(TLAB[t] for t in TYPES) + r" \\", r"\midrule"]
    for a, b in pairs:
        for j, (m, ml, sc, dl) in enumerate(mets):
            res = {r["true_type"]: r for r in T.paired_bootstrap(tb, a, b, m)}
            cells = []
            for tt in TYPES:
                r = res[tt]
                cells.append(ci(r["mean"], r["lo"], r["hi"], sc, "{:.2f}", dl))
                key = f"pd{a.replace('_', '')}{b.replace('_', '')}{m.replace('_', '')}{tt}"
                f = (lambda v: money(v * sc)) if dl else (lambda v: f"{v * sc:.2f}")
                mac(key, f(r["mean"])); mac(key + "lo", f(r["lo"])); mac(key + "hi", f(r["hi"]))
            first = f"{LAB[a]} $-$ {LAB[b]}" if j == 0 else ""
            lines.append(f"{first} & {ml} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace" if (a, b) != pairs[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("tab_paired.tex", "\n".join(lines) + "\n")
    sr = [abs(r["mean"]) for r in T.paired_bootstrap(tb, "ara_dddas", "static_rule", "total_cost")]
    mac("staticrangelo", money(min(sr))); mac("staticrangehi", money(max(sr)))
    n_ep = {r["true_type"]: r["n_episodes"] for r in T.paired_bootstrap(tb, "ara_dddas", "ara_static")}
    mac("nepmain", str(n_ep["aggressive"])); mac("nepside", str(n_ep["cautious"]))


def tab_summers(tb):
    t = tb[tb.true_type.isin(["honest", "aggressive"])].groupby(["true_type", "summer", "strategy"]).total_cost.mean()
    fr = tb[tb.true_type == "aggressive"].groupby(["summer", "strategy"]).fraud.mean()
    lines = [r"\begin{tabular}{ll" + "r" * len(STRATEGIES) + "r}", r"\toprule",
             "True type & Summer & " + " & ".join(LAB[s] for s in STRATEGIES) + r" & Fraud, no defense \\", r"\midrule"]
    for tt in ["aggressive", "honest"]:
        for i, E in enumerate(data.EVAL_SUMMERS):
            vals = " & ".join(f"{t.loc[(tt, E, s)]:,.0f}" for s in STRATEGIES)
            frs = f"{fr.loc[(E, 'no_defense')]:,.0f}" if tt == "aggressive" else "--"
            lines.append(f"{TLAB[tt] if i == 0 else ''} & {E} & {vals} & {frs} \\\\")
        lines.append(r"\addlinespace" if tt == "aggressive" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("tab_summers.tex", "\n".join(lines) + "\n")
    for E in data.EVAL_SUMMERS:
        mac(f"aggnd{E}", f"{t.loc[('aggressive', E, 'no_defense')]:,.0f}")
        mac(f"aggdd{E}", f"{t.loc[('aggressive', E, 'ara_dddas')]:,.0f}")
        mac(f"fraudnd{E}", f"{fr.loc[(E, 'no_defense')]:,.0f}")


def tab_forecast(base):
    f = T.forecast_accuracy(base)
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Data and strategy & MAPE (\%) & CRPS (MW) & 90\% coverage (\%) \\", r"\midrule"]
    names = {"Truthful data": "Truthful readings", "Manipulated, no defense": "Manipulated, no defense"}
    for _, r in f.iterrows():
        n = r.case
        for s, l in LAB.items():
            n = n.replace(f"Manipulated, {T.LABEL[s]}", f"Manipulated, {l}")
        lines.append(f"{names.get(r.case, n)} & {r.MAPE:.2f} & {r.CRPS:.2f} & {r.coverage90:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_forecast.tex", "\n".join(lines) + "\n")
    g = f.set_index("case")
    mac("mapetruth", f"{g.loc['Truthful data'].MAPE:.1f}"); mac("mapend", f"{g.loc['Manipulated, no defense'].MAPE:.1f}")
    mac("covtruth", f"{g.loc['Truthful data'].coverage90:.0f}"); mac("covnd", f"{g.loc['Manipulated, no defense'].coverage90:.0f}")
    mac("crpstruth", f"{g.loc['Truthful data'].CRPS:.2f}"); mac("crpsnd", f"{g.loc['Manipulated, no defense'].CRPS:.2f}")
    d = g.drop(["Truthful data", "Manipulated, no defense"])
    mac("mapedeflo", f"{d.MAPE.min():.1f}"); mac("mapedefhi", f"{d.MAPE.max():.1f}")
    mac("covdeflo", f"{d.coverage90.min():.0f}"); mac("covdefhi", f"{d.coverage90.max():.0f}")


def tab_sensitivity(tot):
    s = tot[(tot.true_type == "aggressive") & tot.summer.isin([2017, 2023])]
    order = ["base", "L10", "L30", "CD25", "CD400", "sig025", "sig1", "sig_inf", "sg_half", "sg_double", "s1_10", "s1_40"]
    lines = [r"\begin{tabular}{lrrccc}", r"\toprule",
             r" & \multicolumn{2}{c}{No defense} & \multicolumn{2}{c}{\ARA{}--\DDDAS{} $-$ \ARA{}-static} & \ARA{}--\DDDAS{} $-$ no defense \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-6}",
             r"Setting & $\Delta\LOLE{}^{a}$ ($10^{-3}$\,h) & Fraud (\$) & Fraud (\$) & Total cost (\$) & Total cost (\$) \\", r"\midrule"]
    for st in order:
        d = s[s.setting == st]
        nd = d[d.strategy == "no_defense"]
        r1 = T.paired_bootstrap(d, "ara_dddas", "ara_static", "fraud")[0]
        r2 = T.paired_bootstrap(d, "ara_dddas", "ara_static", "total_cost")[0]
        r3 = T.paired_bootstrap(d, "ara_dddas", "no_defense", "total_cost")[0]
        lines.append(f"{SENS_LAB[st]} & {1e3 * nd.dLOLE_attack.mean():.2f} & {nd.fraud.mean():,.0f} & "
                     f"{ci(r1['mean'], r1['lo'], r1['hi'])} & {ci(r2['mean'], r2['lo'], r2['hi'])} & {ci(r3['mean'], r3['lo'], r3['hi'])} \\\\")
        k = {"base": "Base", "L10": "Lten", "L30": "Lthirty", "CD25": "CDlow", "CD400": "CDhigh",
             "sig025": "Siglow", "sig1": "Sigone", "sig_inf": "Siginf", "sg_half": "Sghalf",
             "sg_double": "Sgdouble", "s1_10": "Shareten", "s1_40": "Shareforty"}[st]
        mac(f"sens{k}ndla", f"{1e3 * nd.dLOLE_attack.mean():.1f}"); mac(f"sens{k}ndfraud", f"{nd.fraud.mean():,.0f}")
        mac(f"sens{k}ddfraud", money(r1["mean"])); mac(f"sens{k}ddtot", money(r2["mean"]))
        mac(f"sens{k}ddtotlo", money(r2["lo"])); mac(f"sens{k}ddtothi", money(r2["hi"]))
        mac(f"sens{k}ndtot", money(r3["mean"]))
        if st == "base":
            fr = (r1["hi"] < 0)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_sensitivity.tex", "\n".join(lines) + "\n")
    nlow = 0
    for st in order[1:]:
        d = s[s.setting == st]
        if T.paired_bootstrap(d, "ara_dddas", "ara_static", "fraud")[0]["hi"] < 0:
            nlow += 1
    mac("senslowerfraud", str(nlow)); mac("senssettings", str(len(order) - 1))


def tab_traced(base, n=30):
    tr = T.traced(base, n=n)
    lines = [r"\begin{tabular}{rcrrcccrrr}", r"\toprule",
             r"$\tau$ & Event & $\nu_3^{(\tau-1)}$ & $\nu_0^{(\tau-1)}$ & $\delta_\tau$ & $\beta_\tau$ & $a_\tau$ (MWh) & Flag & Audit & $\nu_3^{(\tau)}$ \\",
             r"\midrule"]
    for _, r in tr.iterrows():
        if r.event:
            lines.append(f"{int(r.tau)} & yes & {r.nu3_before:.3f} & {r.nu0_before:.3f} & -- & -- & -- & -- & -- & {r.nu3_after:.3f} \\\\")
        else:
            lines.append(f"{int(r.tau)} & & {r.nu3_before:.3f} & {r.nu0_before:.3f} & {r.delta:g} & {int(r.beta)} & {r.attempt:.2f} & "
                         f"{int(r.flagged)} & {int(r.audited)} & {r.nu3_after:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_traced.tex", "\n".join(lines) + "\n")
    ep = sorted(base[base.summer == 2017].episode_id.unique())[0]
    mac("tracedepisode", ep.split("_")[1])


def tab_diagnostics(lc_path):
    rows = []
    for E in data.EVAL_SUMMERS:
        cal = calibrate.calibrate(E)
        d = cal["diag"]
        cells = []
        for m in ("system", "portfolio", "mixture"):
            rh = max(v[0] for v in d[m].values()); es = min(v[1] for v in d[m].values())
            cells.append(f"{rh:.3f} & {es:,.0f}")
        p = cal["edist"].params
        rows.append(f"{E} & {cal['C']} & " + " & ".join(cells) + f" & {cal['edist'].family} ({p.get('df', float('nan')):.1f}) & {cal['R']:.1f} \\\\")
    lines = [r"\begin{tabular}{llrrrrrrlr}", r"\toprule",
             r" & & \multicolumn{2}{c}{System load} & \multicolumn{2}{c}{Portfolio load} & \multicolumn{2}{c}{Mixture} & & \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Summer & Calibration & $\hat R$ & ESS & $\hat R$ & ESS & $\hat R$ & ESS & $e$ (df) & $R$ (MW) \\", r"\midrule"]
    lines += rows + [r"\bottomrule", r"\end{tabular}"]
    write("tab_diagnostics.tex", "\n".join(lines) + "\n")
    allrh = []; alless = []
    for E in data.EVAL_SUMMERS:
        for m, dd in calibrate.calibrate(E)["diag"].items():
            allrh += [v[0] for v in dd.values()]; alless += [v[1] for v in dd.values()]
    mac("maxrhat", f"{max(allrh):.2f}"); mac("miness", f"{min(alless):,.0f}")
    # long-chain comparison
    import tests.longchain_check as lcc  # noqa
    lc = []
    for E, start in lcc.EPS:
        fx = pd.read_csv(os.path.join(HERE, "results", "B", "base", f"{E}_{start}_s1_aggressive.csv.gz"))
        for s in lcc.STRATS:
            l_ = pd.read_csv(os.path.join(HERE, "results", "longchain", f"{E}_{start}_{s}.csv.gz"))
            f_ = fx[fx.strategy == s].reset_index(drop=True)
            ne = (f_.event == 0).to_numpy()
            same = ((f_.delta.fillna(-1) == l_.delta.fillna(-1)) & (f_.beta == l_.beta)).to_numpy()[ne].mean()
            lc.append(dict(E=E, start=start, s=s, cf=-f_.utility.sum(), cl=-l_.utility.sum(),
                           same=same, fd=(f_.forecast_mean - l_.forecast_mean).abs().mean(),
                           crf=f_.crps.mean(), crl=l_.crps.mean()))
    lc = pd.DataFrame(lc)
    # seed-to-seed s.d. of the same episode and strategy, version B
    tb = T.episode_totals(T.load("B", "base"))
    sd = tb[tb.true_type == "aggressive"].groupby(["episode_id", "strategy"]).total_cost.std()
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Episode & Strategy & \multicolumn{2}{c}{Total cost (\$)} & Seed s.d. (\$) & Same decision & $|\Delta M|$ (MW) & CRPS \\",
             r"\cmidrule(lr){3-4}", r" & & Fixed & Re-estimated & & (\%) & & fixed / re-est. \\", r"\midrule"]
    for _, r in lc.iterrows():
        sdv = sd.loc[(f"{r.E}_{r.start}", r.s)]
        lines.append(f"{r.start} & {LAB[r.s]} & {r.cf:,.0f} & {r.cl:,.0f} & {sdv:,.0f} & "
                     f"{'--' if r.s == 'no_defense' else f'{100 * r.same:.0f}'} & {r.fd:.2f} & {r.crf:.2f} / {r.crl:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_longchain.tex", "\n".join(lines) + "\n")
    nd = lc[lc.s == "no_defense"]
    mac("lcndmaxrel", f"{100 * np.max(np.abs(nd.cl - nd.cf) / nd.cf):.1f}")
    ar = lc[lc.s != "no_defense"]
    mac("lcsame", f"{100 * ar.same.mean():.0f}")
    mac("lcfdlo", f"{lc.fd.min():.2f}"); mac("lcfdhi", f"{lc.fd.max():.2f}")
    mac("lcndfdhi", f"{nd.fd.max():.2f}")


def misc(base, tot):
    q = T.posterior_true_type(base)
    last = q[q.tau == q.tau.max()].set_index("true_type")
    for tt in TYPES:
        mac(f"post{tt}", f"{last.loc[tt, 'mean']:.2f}")
    mac("posthonestpten", f"{last.loc['honest', 'p10']:.2f}")
    ev = {}
    for E in data.EVAL_SUMMERS:
        import episode
        cal = calibrate.calibrate(E); st = episode.episode_setup(E, episode_starts(E)[0], BASE, cal)
        summ = np.asarray(st["idx"] >= pd.Timestamp(f"{E}-06-01 01:00"))
        ev[E] = 100 * st["ev"][summ].mean()
    mac("evlo", f"{min(ev.values()):.1f}"); mac("evhi", f"{max(ev.values()):.1f}")
    nd = base[(base.strategy == "no_defense") & (base.event == 0)]
    for tt in ("cautious", "moderate", "aggressive"):
        mac(f"attshare{tt}", f"{100 * (nd[nd.true_type == tt].a_attempt > 0).mean():.0f}")
    mac("nepisodes", str(base.groupby(['summer', 'episode_id']).ngroups))
    mac("nreplicates", str(tot[tot.setting == "base"].groupby(["summer", "episode_id", "seed", "true_type"]).ngroups))


def main():
    os.makedirs(OUT, exist_ok=True)
    df = T.load("B")
    base = df[df.setting == "base"]
    tot = T.episode_totals(df)
    tb = tot[tot.setting == "base"]
    tab_strategies(tb)
    tab_paired(tb)
    tab_summers(tb)
    tab_forecast(base)
    tab_sensitivity(tot)
    tab_traced(base)
    tab_diagnostics(None)
    misc(base, tot)
    q = T.posterior_true_type(base)
    _figure(q, os.path.join(OUT, "fig_posterior_true_type.pdf"))
    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write("% generated by code/tables_tex.py -- do not edit\n")
        for k, v in MAC.items():
            fh.write(f"\\newcommand{{\\N{k}}}{{{v}}}\n")
    print(f"wrote {len(MAC)} macros and tables to {OUT}")


def _figure(q, path):
    import matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType, no Type 3 fonts
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    names = {"honest": "Honest", "cautious": "Type 1 (cautious)", "moderate": "Type 2 (moderate)",
             "aggressive": "Type 3 (aggressive)"}
    for tt in TYPES:
        d = q[q.true_type == tt]
        ax.plot(d.tau, d["mean"], label=names[tt], lw=1.4)
        ax.fill_between(d.tau, d.p10, d.p90, alpha=0.12)
    ax.axhline(0.25, color="grey", lw=0.6, ls=":")
    ax.set_xlabel(r"interval $\tau$ (hours)"); ax.set_ylabel("posterior probability of the true type")
    ax.set_xlim(1, q.tau.max()); ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path)


if __name__ == "__main__":
    main()
