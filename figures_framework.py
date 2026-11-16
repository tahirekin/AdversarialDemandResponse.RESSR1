"""Figures 1 (workflow) and 2 (BAID) of the manuscript, drawn with matplotlib.

  python3 figures_framework.py      # writes ../files/fig1_workflow.pdf/.png, fig2_baid.pdf/.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, RegularPolygon, Rectangle  # noqa: E402

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
OUT = os.environ.get("RESS_FIG_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "files"))

DEF = ("#e8eef8", "#2b4f8a")      # defender: fill, edge
ATT = ("#fbe9e7", "#a8322b")      # attacker
STAT = ("#e9f3ea", "#2e6b36")     # statistical models / outcomes
UTIL = ("#fdf1dc", "#b5761a")     # utilities
NEUT = ("#f4f4f4", "#555555")


def arrow(ax, p, q, dashed=False, color="#333333", rad=0.0, lw=1.0, shrinkA=0, shrinkB=0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color,
                                 linestyle=(0, (4, 3)) if dashed else "solid",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=shrinkA, shrinkB=shrinkB))


# ----------------------------------------------------------------------
#  Figure 1: workflow
# ----------------------------------------------------------------------
def fig1():
    steps = [
        ("Step 1: Defender commitment", DEF,
         r"commits to $d_\tau=(\beta_\tau,\delta_\tau)$ using beliefs $\nu^{(\tau-1)}$" "\n"
         r"and data through $\tau-1$ (Section 6.1)"),
        ("Step 2: Attack decision and data acquisition", ATT,
         r"attacker perceives $\delta_\tau$ with noise, chooses attempt $a_\tau$;" "\n"
         r"reports $Y_{1,\tau}=X_{1,\tau}+a_\tau$; other participants report truthfully"),
        ("Step 3: Detection, audit and imputation", STAT,
         r"flag if $u_{i,\tau}<\delta_\tau$; flagged reading of participant 1 audited if $\beta_\tau=1$" "\n"
         r"(true value, penalty); other flagged readings imputed (forecast input only)"),
        ("Step 4: Forecasting, reliability and settlement", STAT,
         r"SARIMA update $\rightarrow$ committed supply $G_{\tau+h}$ $\rightarrow$ LOLP of event hours;" "\n"
         r"settlement from the baseline of audited / reported readings"),
        ("Step 5: Bayesian learning", DEF,
         r"defender: $\nu^{(\tau-1)}\rightarrow\nu^{(\tau)}$ from $Y_{1,\tau}$, detection and audit outcome;" "\n"
         r"attacker: belief over $\delta$ from delayed feedback ($\tau-\kappa$)"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.6); ax.axis("off")
    x0, w, h = 1.6, 7.6, 1.42
    ys = [9.0 - i * 1.95 for i in range(len(steps))]
    for (title, (fc, ec), body), y in zip(steps, ys):
        ax.add_patch(FancyBboxPatch((x0, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x0 + w / 2, y + 0.33, title, ha="center", va="center", fontsize=9.5, weight="bold", color=ec)
        ax.text(x0 + w / 2, y - 0.24, body, ha="center", va="center", fontsize=8, linespacing=1.35)
    for y1, y2 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x0 + w / 2, y1 - h / 2), (x0 + w / 2, y2 + h / 2), lw=1.2)
    # feedback loop
    xl = 0.75
    ax.plot([x0, xl, xl], [ys[-1], ys[-1], ys[0]], color="#333333", lw=1.2)
    arrow(ax, (xl, ys[0]), (x0, ys[0]), lw=1.2)
    ax.text(xl - 0.12, (ys[0] + ys[-1]) / 2, r"$\tau\rightarrow\tau+1$", rotation=90, ha="right", va="center", fontsize=9)
    # legend
    for i, (lab, (fc, ec)) in enumerate([("defender", DEF), ("attacker", ATT), ("statistical models and outcomes", STAT)]):
        ax.add_patch(Rectangle((x0 + i * 2.35, 10.15), 0.28, 0.2, fc=fc, ec=ec, lw=1))
        ax.text(x0 + i * 2.35 + 0.38, 10.25, lab, va="center", fontsize=7.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig1_workflow.{ext}"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------
#  Figure 2: bi-agent influence diagram
# ----------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(7.0, 7.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 15.2); ax.axis("off")
    R = 0.92
    nodes = {}

    def dec(key, xy, label, sub, col):
        w, h = 2.5, 1.25
        ax.add_patch(Rectangle((xy[0] - w / 2, xy[1] - h / 2), w, h, fc=col[0], ec=col[1], lw=1.4))
        ax.text(xy[0], xy[1] + 0.2, label, ha="center", va="center", fontsize=10, weight="bold", color=col[1])
        ax.text(xy[0], xy[1] - 0.28, sub, ha="center", va="center", fontsize=7.2)
        nodes[key] = ("rect", xy, w, h)

    def chance(key, xy, label, sub, col, det=False, r=None):
        r = r or R
        ax.add_patch(Circle(xy, r, fc=col[0], ec=col[1], lw=1.3))
        if det:
            ax.add_patch(Circle(xy, r - 0.1, fc="none", ec=col[1], lw=0.9))
        ax.text(xy[0], xy[1] + 0.22, label, ha="center", va="center", fontsize=9.5, weight="bold", color=col[1])
        ax.text(xy[0], xy[1] - 0.25, sub, ha="center", va="center", fontsize=6.6, linespacing=1.1)
        nodes[key] = ("circ", xy, r, r)

    def util(key, xy, label, sub):
        ax.add_patch(RegularPolygon(xy, 6, radius=0.95, orientation=0.5236, fc=UTIL[0], ec=UTIL[1], lw=1.4))
        ax.text(xy[0], xy[1] + 0.22, label, ha="center", va="center", fontsize=10, weight="bold", color=UTIL[1])
        ax.text(xy[0], xy[1] - 0.27, sub, ha="center", va="center", fontsize=6.6, linespacing=1.1)
        nodes[key] = ("circ", xy, 0.9, 0.9)

    def edge_point(key, toward):
        kind, (x, y), a, b = nodes[key]
        dx, dy = toward[0] - x, toward[1] - y
        d = (dx ** 2 + dy ** 2) ** 0.5
        if kind == "circ":
            return (x + a * dx / d, y + a * dy / d)
        # rectangle: scale to boundary
        sx = (a / 2) / abs(dx) if dx else 1e9
        sy = (b / 2) / abs(dy) if dy else 1e9
        s_ = min(sx, sy)
        return (x + dx * s_, y + dy * s_)

    def link(a, b, dashed=False, rad=0.0, color="#333333", label=None, lpos=0.5, loff=(0, 0)):
        pa, pb = nodes[a][1], nodes[b][1]
        p = edge_point(a, pb); q = edge_point(b, pa)
        arrow(ax, p, q, dashed=dashed, rad=rad, color=color, lw=1.0)
        if label:
            mx = p[0] + (q[0] - p[0]) * lpos + loff[0]; my = p[1] + (q[1] - p[1]) * lpos + loff[1]
            ax.text(mx, my, label, fontsize=7, color=color, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.4))

    # row 1: beliefs, decisions
    chance("nu", (1.4, 13.8), r"$\nu^{(\tau-1)}$", "defender's\nbelief", DEF)
    dec("D", (4.6, 13.8), r"$d_\tau=(\beta_\tau,\delta_\tau)$", "defender decision", DEF)
    chance("theta", (8.3, 13.8), r"$\theta_A$", "attacker\ntype", ATT)
    dec("A", (11.8, 11.4), r"$a_\tau$", "attempt", ATT)
    chance("omega", (8.3, 11.4), r"$\omega^{(\tau)}$", "attacker's\n" r"belief on $\delta$", ATT)
    chance("X", (11.8, 13.8), r"$X_{1,\tau}$", "true\nconsumption", NEUT)
    # row 3: report and detection
    chance("Y", (11.8, 8.9), r"$Y_{1,\tau}$", "reported\nreading", ATT, det=True)
    chance("S", (8.3, 8.9), r"$S_\tau$", "detection\n" r"$u_{1,\tau}\geq\delta_\tau$", STAT, det=True)
    chance("f", (4.6, 8.9), r"$f_{1,\tau}$", "predictive\n" r"density ($\Theta_{\mathrm{OD}}$)", STAT)
    chance("xi", (8.3, 6.5), r"$\xi_\tau$", "audit", DEF, det=True)
    chance("Yt", (4.6, 6.5), r"$\tilde Y_\tau$", "forecast\ninput", STAT, det=True)
    chance("Yset", (11.8, 6.5), r"$Y^{\mathrm{set}}_{1,\tau}$", "settlement\nrecord", STAT, det=True)
    # row 5: forecast, events, payments
    chance("M", (4.6, 4.1), r"$M_{\tau+h}$", "forecast\n" r"($\Theta_{\mathrm{FC}}$)", STAT)
    chance("ev", (6.6, 4.1), r"$\mathrm{ev},\,p$", "events,\nprices", NEUT, r=0.72)
    chance("L", (4.6, 1.7), r"$\Delta$LOLP", "reliability", STAT, det=True)
    chance("P", (11.8, 4.1), r"$\Pi_{\mathrm{fraud}}$", "fraudulent\npayment", STAT, det=True)
    util("UD", (8.3, 1.7), r"$U_D$", "defender\nutility")
    util("UA", (11.8, 1.7), r"$U_A$", "attacker\nutility")

    # information and decisions
    link("nu", "D")
    link("D", "omega", dashed=True, color="#2b4f8a", label="noisy perception", lpos=0.5, loff=(-0.15, 0.25))
    link("theta", "A")
    link("omega", "A")
    link("X", "A")
    link("X", "Y"); link("A", "Y")
    link("Y", "S"); link("f", "S")
    link("D", "S", rad=0.0, label=r"$\delta_\tau$", lpos=0.55, loff=(0.25, 0.1))
    link("S", "xi")
    link("D", "xi", rad=0.12, label=r"$\beta_\tau$", lpos=0.62, loff=(-0.35, 0.0))
    link("S", "Yt"); link("xi", "Yt", rad=0.0)
    link("xi", "Yset"); link("Y", "Yset")
    link("Yt", "M")
    link("M", "L"); link("ev", "L"); link("ev", "P")
    link("Yset", "P")
    link("L", "UD"); link("P", "UD"); link("P", "UA")
    link("xi", "UD")
    link("A", "UA", rad=-0.35)
    # delayed feedback to the attacker
    link("S", "omega", dashed=True, rad=0.45, color="#a8322b", label=r"feedback, delay $\kappa$", lpos=0.5, loff=(-0.85, 0))
    # belief update arc to next interval
    link("S", "nu", dashed=True, rad=-0.25, color="#2b4f8a", label=r"update $\to\nu^{(\tau)}$", lpos=0.45, loff=(-0.2, -0.1))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig2_baid.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
    print("written to", OUT)
