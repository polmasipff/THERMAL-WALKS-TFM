"""
F1_combined_Tcorr_HDX -- overlap of the seven TCV categories along T_corr and HDX.
Each panel shows the within-category distribution as a violin, with the category
mean +/- 95% CI overlaid. Vertical layout: TCV categories on the x-axis, thermal
coordinate on the y-axis. Left panel: T_corr. Right panel: HDX.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _common import load, set_style, set_paper_style, save, CM
from matplotlib.ticker import MultipleLocator

set_style()
set_paper_style()

df = load()

# ---------------------------------------------------------------------
# Category order and labels
# ---------------------------------------------------------------------

order = [3, 2, 1, 0, -1, -2, -3]   # comfortable -> uncomfortable

labels = {
    3: "V.C.",
    2: "C.",
    1: "S.C.",
    0: "N.",
    -1: "S.U.",
    -2: "U",
    -3: "V.U."
}

# ---------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------

COLC_FILL = "#2C7AB5"   # blue
COLN_FILL = "#918686"   # grey
COLU_FILL = "#E69F00"   # orange, colorblind-friendly

def fill_color(s):
    if s > 0:
        return COLC_FILL
    elif s < 0:
        return COLU_FILL
    else:
        return COLN_FILL

def dark_color(s):
    if s > 0:
        return "#1B4F72"
    elif s < 0:
        return "#A35F00"
    else:
        return "#4F4F4F"


# ---------------------------------------------------------------------
# Clean y-axis settings
# ---------------------------------------------------------------------

YCONF = {
    "T_corr": {
        "ylim": (20, 35),
        "yticks": np.arange(20, 36, 5)
    },
    "HDX": {
        "ylim": (20, 45),
        "yticks": np.arange(20, 46, 5)
    }
}


# ---------------------------------------------------------------------
# Helper to draw one panel
# ---------------------------------------------------------------------

def draw_violin_panel(ax, df, value_col, ylab):
    data = [
        df.loc[df.score == s, value_col].dropna().values
        for s in order
    ]

    positions = np.arange(len(order))

    vp = ax.violinplot(
        data,
        positions=positions,
        vert=True,
        widths=0.88,
        showmeans=False,
        showextrema=False,
        showmedians=False
    )

    for body, s in zip(vp["bodies"], order):
        body.set_facecolor(fill_color(s))
        body.set_edgecolor("0.5")
        body.set_alpha(0.72)
        body.set_linewidth(0.5)

    # Mean +/- 95% CI overlay
    for s, x in zip(order, positions):
        v = df.loc[df.score == s, value_col].dropna().values

        if len(v) == 0:
            continue

        m = v.mean()
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))

        ax.plot(
            [x, x],
            [m - ci, m + ci],
            color=dark_color(s),
            lw=4.0,
            solid_capstyle="butt",
            zorder=4
        )

        ax.scatter(
            [x],
            [m],
            color="white",
            edgecolors=dark_color(s),
            s=10,
            lw=0.8,
            zorder=5
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([labels[s] for s in order])

    ax.set_xlabel("TCV", fontsize=15)
    ax.set_ylabel(ylab, fontsize=15)

    # Clean y-axis
    # ax.set_ylim(*YCONF[value_col]["ylim"])
    # ax.set_yticks(YCONF[value_col]["yticks"])
    conf = YCONF[value_col]
    # Clean y-axis
    ax.set_ylim(*conf["ylim"])
    ax.set_yticks(conf["yticks"])
    ax.set_yticklabels([f"{t:g}" for t in conf["yticks"]])

    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.tick_params(axis="x", labelsize=12)

    ax.tick_params(
        axis="y",
        which="major",
        direction="out",
        length=4,
        width=0.8,
        labelsize=12
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(False)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, (axL, axR) = plt.subplots(
    1, 2,
    figsize=(19 * CM, 8.5 * CM)
)

fig.patch.set_facecolor("white")

for ax in (axL, axR):
    ax.set_facecolor("white")

# Left panel: T_corr
draw_violin_panel(
    axL,
    df,
    value_col="T_corr",
    ylab=r"$T_{\rm corr}$ ($^\circ$C)"
)

# Right panel: HDX
draw_violin_panel(
    axR,
    df,
    value_col="HDX",
    ylab="HDX"
)

# Common legend
legend_handles = [
    Line2D(
        [0], [0],
        color="0.30",
        lw=4.0,
        label="mean ± 95% CI"
    ),
    Line2D(
        [0], [0],
        marker="s",
        color="0.7",
        lw=0,
        markersize=7,
        label="distribution (violin)"
    )
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=2,
    frameon=False,
    fontsize=12,
    handlelength=1.8,
    columnspacing=1.5
)

fig.subplots_adjust(
    wspace=0.24,
    left=0.08,
    right=0.985,
    bottom=0.20,
    top=0.82
)

save(fig, "F1_overlap_7cat_Tcorr_HDX")
print("done")