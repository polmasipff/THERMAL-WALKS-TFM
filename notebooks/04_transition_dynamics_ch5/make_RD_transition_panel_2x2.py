
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Put these CSV files in the same folder as this script, or edit the paths.
from pathlib import Path

path_tenv_bins = "ORGANISED/thermal_next/markov_results/T_ENV_RESULSTS_BINS.csv"
path_tcorr_bins = "markov_results/T_CORR_RESULSTS_BINS.csv"
path_rd_tcorr = "markov_results/T_corr_above_RD.csv"
path_onset = "markov_results/T_corr_onset.csv"
path_recovery = "markov_results/T_corr_recovery.csv"

rd_tenv_bins = pd.read_csv(path_tenv_bins)
rd_tcorr_bins = pd.read_csv(path_tcorr_bins)
rd_tcorr = pd.read_csv(path_rd_tcorr)
onset = pd.read_csv(path_onset)
recovery = pd.read_csv(path_recovery)

def paper_style_2x2():
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 8.0,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

paper_style_2x2()
BLUE_LIGHT = "#6BAED6"
BLUE_DARK = "#08519C"
BLUE_MID = "#3182BD"
BLUE_PALE = "#9ECAE1"

def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.90", linewidth=0.6, zorder=0)

def panel_label(ax, label):
    ax.text(-0.17, 1.06, label, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, fontweight="bold")

def add_vertical_lines(ax, lines):
    if lines is None:
        return
    for x in lines:
        ax.axvline(x, linestyle="--", color="0.55",
                   linewidth=0.75, alpha=0.75, zorder=1)

def draw_rd_sweep(ax, rd, panel="a", t_lines=(26.10, 30.77)):
    rd = rd.copy().sort_values("threshold")
    x = rd["threshold"].astype(float).to_numpy()
    y = rd["R_D"].astype(float).to_numpy()
    lo = rd["R_D_low"].astype(float).to_numpy()
    hi = rd["R_D_high"].astype(float).to_numpy()
    ax.fill_between(x, lo, hi, color=BLUE_LIGHT, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(x, y, marker="o", markersize=3.0,
            markerfacecolor=BLUE_DARK, markeredgecolor="black", markeredgewidth=0.25,
            color=BLUE_DARK, linewidth=0.95, label=r"$R_D$", zorder=3)
    ax.axhline(1, linestyle="--", color="black", linewidth=0.8,
               alpha=0.75, label=r"$R_D=1$", zorder=1)
    add_vertical_lines(ax, t_lines)
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(0, 3)
    ax.set_xlabel(r"$T_{\mathrm{corr},i+1}$ threshold ($^\circ$C)")
    ax.set_ylabel(r"$R_D$")
    ax.legend(frameon=False, loc="upper left", handlelength=1.6, borderpad=0.15, labelspacing=0.25)
    clean_axis(ax)
    panel_label(ax, panel)

def draw_components_sweep(ax, onset, recovery, panel="b", t_lines=(26.10, 30.77)):
    onset = onset.copy().sort_values("threshold")
    recovery = recovery.copy().sort_values("threshold")
    xo = onset["threshold"].astype(float).to_numpy()
    yo = onset["prob"].astype(float).to_numpy()
    lo_o = onset["q025"].astype(float).to_numpy()
    hi_o = onset["q975"].astype(float).to_numpy()
    xr = recovery["threshold"].astype(float).to_numpy()
    yr = recovery["prob"].astype(float).to_numpy()
    lo_r = recovery["q025"].astype(float).to_numpy()
    hi_r = recovery["q975"].astype(float).to_numpy()
    ax.fill_between(xo, lo_o, hi_o, color=BLUE_LIGHT, alpha=0.24, linewidth=0, zorder=2)
    ax.plot(xo, yo, marker="o", markersize=2.6,
            markerfacecolor=BLUE_DARK, markeredgecolor="black", markeredgewidth=0.22,
            color=BLUE_DARK, linewidth=0.9, label=r"Entry into $U$", zorder=4)
    ax.fill_between(xr, lo_r, hi_r, color=BLUE_PALE, alpha=0.30, linewidth=0, zorder=2)
    ax.plot(xr, yr, marker="s", markersize=2.6,
            markerfacecolor=BLUE_PALE, markeredgecolor="black", markeredgewidth=0.22,
            color=BLUE_MID, linewidth=0.9, label=r"Exit from $U$", zorder=4)
    add_vertical_lines(ax, t_lines)
    ax.set_xlim(min(np.nanmin(xo), np.nanmin(xr)), max(np.nanmax(xo), np.nanmax(xr)))
    ax.set_ylim(0, 0.8)
    ax.set_xlabel(r"$T_{\mathrm{corr},i+1}$ threshold ($^\circ$C)")
    ax.set_ylabel("Transition probability")
    ax.legend(frameon=False, loc="best", handlelength=1.4, borderpad=0.15, labelspacing=0.25)
    clean_axis(ax)
    panel_label(ax, panel)

def style_for_bins(nbin_values):
    colors = {}
    markers = {}
    base_colors = ["#6BAED6", "#08519C", "#3182BD", "#9ECAE1"]
    base_markers = ["o", "s", "^", "D"]
    for i, nb in enumerate(list(nbin_values)):
        colors[nb] = base_colors[i % len(base_colors)]
        markers[nb] = base_markers[i % len(base_markers)]
    return colors, markers

def draw_rd_bins(ax, rd_df, panel="c", x_plot_col="x_median",
                 plot_bin_numbers=(5, 10),
                 x_label=r"$T_{\mathrm{corr},i+1}$ bin median ($^\circ$C)",
                 y_label=r"$R_D^{\mathrm{broad}}$",
                 y_lim=(0, 2.5), vlines=None):
    df = rd_df.copy()
    if plot_bin_numbers is not None:
        df = df[df["n_bins"].isin(plot_bin_numbers)].copy()
    df = df.sort_values(["n_bins", x_plot_col])
    nbin_values = sorted(df["n_bins"].unique())
    colors, markers = style_for_bins(nbin_values)
    ycol = "RD_broad_prob"
    low_col = f"{ycol}_low"
    high_col = f"{ycol}_high"
    for nb in nbin_values:
        sub = df[df["n_bins"] == nb].copy().sort_values(x_plot_col)
        x = sub[x_plot_col].astype(float).to_numpy()
        y = sub[ycol].astype(float).to_numpy()
        color = colors[nb]
        if low_col in sub.columns and high_col in sub.columns:
            lo = sub[low_col].astype(float).to_numpy()
            hi = sub[high_col].astype(float).to_numpy()
            mask = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
            if mask.sum() >= 2:
                ax.fill_between(x[mask], lo[mask], hi[mask],
                                color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, y, marker=markers[nb], markersize=3.2,
                markerfacecolor=color, markeredgecolor="black", markeredgewidth=0.25,
                color=color, linewidth=0.95, label=f"{nb} bins", zorder=3)
    ax.axhline(1, linestyle="--", color="black", linewidth=0.75, alpha=0.55, zorder=1)
    add_vertical_lines(ax, vlines)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(*y_lim)
    clean_axis(ax)
    ax.legend(frameon=False, loc="best", handlelength=1.4, borderpad=0.15, labelspacing=0.25)
    panel_label(ax, panel)

fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.75), constrained_layout=False)
draw_rd_sweep(axes[0, 0], rd_tcorr, panel="a", t_lines=(26.10, 30.77))
draw_components_sweep(axes[0, 1], onset, recovery, panel="b", t_lines=(26.10, 30.77))
draw_rd_bins(axes[1, 0], rd_tcorr_bins, panel="c", x_plot_col="x_median",
             plot_bin_numbers=(5, 10),
             x_label=r"$T_{\mathrm{corr},i+1}$ bin median ($^\circ$C)",
             y_lim=(0, 2.5), vlines=(26.10, 30.77))
draw_rd_bins(axes[1, 1], rd_tenv_bins, panel="d", x_plot_col="x_median",
             plot_bin_numbers=(5, 10),
             x_label=r"$T_{\mathrm{env},i+1}$ bin median",
             y_lim=(0, 5.2), vlines=(24.62, 30.83))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.115, top=0.965,
                    wspace=0.33, hspace=0.42)
fig.savefig("RD_transition_panel_2x2.pdf")
fig.savefig("RD_transition_panel_2x2.png", dpi=300)
plt.show()
