"""
F_tsv_tcv_corr -- exploratory relationship between thermal sensation (TSV)
and thermal comfort (TCV). The plot shows the row-normalised contingency
matrix P(TCV | TSV), with cell counts annotated. Spearman correlation is
computed from the paired votes using an ordinal coding.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

from _common import set_style, set_paper_style, save, CM

set_style()
set_paper_style()

HERE = os.path.dirname(os.path.abspath(__file__))

BASE = os.path.join(
    HERE,
    "..","data",
    "markovian_analysis_baseline.csv"
)

# ---------------------------------------------------------------------
# Style and palette
# ---------------------------------------------------------------------

PAL_MINT = "#9CE5E8"
PAL_BLUE = "#2C7AB5"
PAL_NAVY = "#152F61"
COL_DARK = "0.15"

CMAP = LinearSegmentedColormap.from_list(
    "tfm_blues",
    ["#FFFFFF", PAL_MINT, PAL_BLUE, PAL_NAVY]
)

# ---------------------------------------------------------------------
# Orders and labels
# ---------------------------------------------------------------------

TSV = [
    "Cool",
    "Slightly cool",
    "Neutral",
    "Slightly warm",
    "Warm",
    "Hot",
    "Very hot"
]

TCV = [
    "Very comfortable",
    "Comfortable",
    "Slightly comfortable",
    "Neutral",
    "Slightly uncomfortable",
    "Uncomfortable",
    "Very uncomfortable"
]

TSV_lab = [
    "Cool",
    "S. cool",
    "Neutral",
    "S. warm",
    "Warm",
    "Hot",
    "V. hot"
]

TCV_lab = [
    "V. comf.",
    "Comf.",
    "S. comf.",
    "Neutral",
    "S. uncomf.",
    "Uncomf.",
    "V. uncomf."
]

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def ordinal_correlation(df, x_col, y_col, x_order, y_order, method="spearman"):
    """
    Correlation between two ordinal categorical variables using the supplied
    category order.
    """
    x_map = {c: i for i, c in enumerate(x_order)}
    y_map = {c: i for i, c in enumerate(y_order)}

    tmp = df[[x_col, y_col]].dropna().copy()
    tmp["_x"] = tmp[x_col].map(x_map)
    tmp["_y"] = tmp[y_col].map(y_map)
    tmp = tmp.dropna(subset=["_x", "_y"])

    x = tmp["_x"].to_numpy()
    y = tmp["_y"].to_numpy()

    if method == "spearman":
        r, p = stats.spearmanr(x, y)
    elif method == "kendall":
        r, p = stats.kendalltau(x, y)
    elif method == "pearson":
        r, p = stats.pearsonr(x, y)
    else:
        raise ValueError("method must be 'spearman', 'kendall', or 'pearson'")

    return r, p, len(tmp)


def conditional_matrix(df, row_col, col_col, row_order, col_order):
    """
    Row-normalised contingency matrix P(col | row), together with raw counts.
    """
    ct = (
        pd.crosstab(df[row_col], df[col_col])
        .reindex(index=row_order, columns=col_order)
        .fillna(0)
    )

    counts = ct.to_numpy(dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)

    probs = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0
    )

    return counts, probs


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

d = pd.read_csv(BASE).dropna(
    subset=["thermal_sensation", "comfort7"]
).copy()

counts, probs = conditional_matrix(
    d,
    row_col="thermal_sensation",
    col_col="comfort7",
    row_order=TSV,
    col_order=TCV
)

rho, pval, n_pairs = ordinal_correlation(
    d,
    x_col="thermal_sensation",
    y_col="comfort7",
    x_order=TSV,
    y_order=TCV,
    method="spearman"
)

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    1, 1,
    figsize=(15.5 * CM, 11.0 * CM)
)

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(
    probs,
    cmap=CMAP,
    aspect="auto",
    vmin=0,
    vmax=np.nanmax(probs)
)

# Axis labels and ticks
ax.set_xticks(np.arange(len(TCV)))
ax.set_xticklabels(
    TCV_lab,
    rotation=35,
    ha="right",
    rotation_mode="anchor",
    fontsize=12
)

ax.set_yticks(np.arange(len(TSV)))
ax.set_yticklabels(
    TSV_lab,
    fontsize=12
)

ax.set_xlabel("thermal comfort vote (TCV)", fontsize=15)
ax.set_ylabel("thermal sensation vote (TSV)", fontsize=15)

ax.tick_params(axis="both", labelsize=12)

# Optional: cleaner cell borders
ax.set_xticks(np.arange(-0.5, len(TCV), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(TSV), 1), minor=True)
# ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)

# Annotate raw counts
threshold = 0.58 * np.nanmax(probs)

for i in range(len(TSV)):
    for j in range(len(TCV)):
        c = int(counts[i, j])

        if c == 0:
            continue

        ax.text(
            j,
            i,
            f"{c}",
            ha="center",
            va="center",
            fontsize=11,
            color="white" if probs[i, j] > threshold else COL_DARK
        )

# Colorbar
cb = fig.colorbar(
    im,
    ax=ax,
    fraction=0.045,
    pad=0.025
)

cb.set_label(
    r"$P(\mathrm{TCV}\mid \mathrm{TSV})$",
    fontsize=13
)

cb.ax.tick_params(labelsize=11)

fig.subplots_adjust(
    left=0.15,
    right=0.90,
    bottom=0.22,
    top=0.91
)

save(fig, "F_tsv_tcv_corr")

print(f"Spearman rho = {rho:.3f}, p = {pval:.2e}, n = {n_pairs}")
print("Row counts by TSV:")
for t, n in zip(TSV, counts.sum(axis=1).astype(int)):
    print(f"  {t:<14s} n={n}")