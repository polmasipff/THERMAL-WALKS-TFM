"""F4 - Likelihood-ratio scan: 3 vs 2 regimes per coordinate.
Horizontal barplot, one bar per coordinate showing G(3-regime); inner segment marks G(2-regime).
Bar colour: green if p_walkperm_3reg < 0.001, yellow if <0.05, red if >=0.05 (all green here).
'ns' annotation on the 2-regime segment for T_corr and HDX (p_perm2 ~ 0.06-0.09).
Numbers frozen from coordinate_comparison.ipynb (pipeline §6). Size 8x5 cm. PNG+PDF."""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from _common import set_style, save, CM, HERE

set_style()

# frozen scan-G table (pipeline §6). If a scanG_all.csv is dropped in later, it is used.
FROZEN = pd.DataFrame([
    {"coord": "T_corr", "G2": 66.8,  "G3": 102.9, "dG": 36.1, "p_chi2": 1.5e-8,
     "p_perm2": 0.085, "p_perm3": 0.0009},
    {"coord": "HDX",    "G2": 62.3,  "G3": 82.4,  "dG": 20.1, "p_chi2": 4.3e-5,
     "p_perm2": 0.055, "p_perm3": 0.0009},
    {"coord": "T_rad",  "G2": 165.8, "G3": 196.2, "dG": 30.4, "p_chi2": 2.5e-7,
     "p_perm2": 0.0009, "p_perm3": 0.0009},
    {"coord": "T_env",  "G2": 188.3, "G3": 278.0, "dG": 89.7, "p_chi2": 3.3e-20,
     "p_perm2": 0.0009, "p_perm3": 0.0009},
])

csv_path = os.path.join(HERE, "..", "data", "scanG_all.csv")
data = pd.read_csv(csv_path) if os.path.exists(csv_path) else FROZEN

DISP = {"T_corr": r"$T_{\rm corr}$", "HDX": "HDX",
        "T_rad": r"$T_{\rm rad}$", "T_env": r"$T_{\rm env}$"}
order = ["T_corr", "HDX", "T_rad", "T_env"]
data = data.set_index("coord").reindex(order).reset_index()


def bar_colour(p3):
    if p3 < 0.001:
        return "#2ca02c"   # green
    if p3 < 0.05:
        return "#e6c700"   # yellow
    return "#d62728"       # red


fig, ax = plt.subplots(figsize=(8 * CM, 5 * CM))
y = np.arange(len(order))[::-1]

for yi, (_, r) in zip(y, data.iterrows()):
    col = bar_colour(r["p_perm3"])
    # full bar = G(3)
    ax.barh(yi, r["G3"], color=col, alpha=0.85, height=0.6, zorder=2)
    # inner segment = G(2)
    ax.barh(yi, r["G2"], color="black", alpha=0.0, height=0.6, zorder=3,
            edgecolor="black", lw=0.0)
    ax.plot([r["G2"], r["G2"]], [yi - 0.30, yi + 0.30], color="black", lw=1.1, zorder=4)
    # label G3 value at bar end
    ax.text(r["G3"] + 4, yi, f"{r['G3']:.0f}", va="center", fontsize=6)
    # 'ns' annotation for marginal 2-regime
    if r["p_perm2"] >= 0.05:
        ax.text(r["G2"] - 3, yi, "ns", va="center", ha="right",
                fontsize=6, color="0.25", style="italic")

ax.set_yticks(y)
ax.set_yticklabels([DISP[c] for c in order])
ax.set_xlabel(r"Likelihood-ratio statistic $G$")
ax.set_xlim(0, data["G3"].max() * 1.18)
ax.grid(axis="x", color="0.92", lw=0.5)
ax.set_axisbelow(True)

# legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="#2ca02c", lw=6, alpha=0.85, label=r"$G$(3-regime), perm. $p<0.001$"),
    Line2D([0], [0], color="black", lw=1.1, label=r"$G$(2-regime) marker"),
]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=5.5)

save(fig, "F4_scanG_barplot")
