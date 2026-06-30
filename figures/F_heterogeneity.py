# """F_heterogeneity -- Heterogeneity of the comfort<->discomfort threshold across subgroups.

# End-of-Chapter-3 figure. The seven-category overlap (Fig 3.1) is given a mechanism:
# individuals place the comfort/discomfort boundary at different temperatures, so the
# populational transition has a FINITE WIDTH (inverse logistic slope 1/beta_1 ~ 4.8 C),
# identified in Chapter 4 with the Blume-Capel populational inverse temperature beta^-1.

# Threshold per subgroup: T(m=0) = T where P(U)=P(C), as -b0/b1 from a logistic
# [uncomfortable vs comfortable] ~ T_corr (neutral dropped). 95% CIs from a WALK-STRATIFIED
# (cluster) bootstrap -- whole walks resampled -- the bootstrap prescribed in the Methods.

# TRIBUNAL FIXES baked in (see heterogeneity_figure_review.md):
#  (1) City typo "L'Hospitalet de Llobreat" merged into "...Llobregat".
#  (2) Age uses the BINARY <=15 / >=16 split (PROJECT_STATUS open item #1).
#  (3) Under the correct walk-stratified bootstrap the paired age diff is +1.3 C with CI that
#      INCLUDES zero (younger-warmer ~94%): real in direction, NOT significant. The previously
#      reported "+1.9 C [+0.2,+4.5] significant" is recoverable only by dropping 65-84 from >=16,
#      so we plot ">=16 (all)" AND a flagged "16-64 (no 65+)" sensitivity row.

# Data: data/markovian_analysis_baseline.csv
# """
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from _common import set_style, save, CM

# set_style()
# HERE = os.path.dirname(os.path.abspath(__file__))
# (data path: ../data/  -- see data/README.md)
#                     "saved_df", "markovian_analysis_baseline.csv")
# RNG = np.random.default_rng(2)
# NBOOT = 2000
# DEG = r"$^{\circ}$C"

# d = pd.read_csv(BASE)
# d["city"] = d["city"].replace({"L'Hospitalet de Llobreat": "L'Hospitalet de Llobregat"})
# d = d.dropna(subset=["comfort3", "T_corr", "ID", "walk_id"])
# d = d[d.comfort3.isin(["comfortable", "uncomfortable"])].copy()
# d["U"] = (d.comfort3 == "uncomfortable").astype(int)

# YOUNG = {"Less than 10", "10-12", "13-15"}
# ELDERLY = {"65-74", "75-84"}
# d["agebin"] = d.age.map(lambda a: "le15" if a in YOUNG else "ge16")


# def fit(sub, iters=30):
#     if len(sub) < 40 or sub.U.nunique() < 2:
#         return np.nan, np.nan
#     x = sub["T_corr"].to_numpy(float)
#     y = sub["U"].to_numpy(float)
#     X = np.column_stack([np.ones_like(x), x])
#     beta = np.zeros(2)
#     for _ in range(iters):
#         p = 1.0 / (1.0 + np.exp(-(X @ beta)))
#         W = p * (1 - p)
#         try:
#             step = np.linalg.solve((X * W[:, None]).T @ X + 1e-8 * np.eye(2), X.T @ (y - p))
#         except np.linalg.LinAlgError:
#             return np.nan, np.nan
#         beta += step
#         if np.max(np.abs(step)) < 1e-9:
#             break
#     b0, b1 = beta
#     return (-b0 / b1 if b1 > 0 else np.nan), b1


# def boot_threshold(sub, n=NBOOT):
#     wl = sub.walk_id.unique()
#     wi = {w: sub.index[sub.walk_id == w].to_numpy() for w in wl}
#     out = []
#     for _ in range(n):
#         ch = RNG.choice(wl, size=len(wl), replace=True)
#         t, _ = fit(sub.loc[np.concatenate([wi[w] for w in ch])])
#         if not np.isnan(t):
#             out.append(t)
#     out = np.asarray(out)
#     return np.median(out), np.percentile(out, 2.5), np.percentile(out, 97.5)


# def boot_diff(sub, key, gA, gB, n=NBOOT):
#     wl = sub.walk_id.unique()
#     wi = {w: sub.index[sub.walk_id == w].to_numpy() for w in wl}
#     diffs = []
#     for _ in range(n):
#         ch = RNG.choice(wl, size=len(wl), replace=True)
#         r = sub.loc[np.concatenate([wi[w] for w in ch])]
#         ta, _ = fit(r[r[key] == gA])
#         tb, _ = fit(r[r[key] == gB])
#         ok = (not np.isnan(ta)) and (not np.isnan(tb))
#         if ok:
#             diffs.append(ta - tb)
#     diffs = np.asarray(diffs)
#     return np.median(diffs), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), np.mean(diffs > 0)


# overall_T, overall_b = fit(d)
# overall_width = 1.0 / overall_b

# g = d[d.gender.isin(["Man", "Woman"])]
# d_no_eld = d[~d.age.isin(ELDERLY)]

# rows = [
#     ("All votes",            d,                                   "ref"),
#     ("Women",                g[g.gender == "Woman"],              "clean"),
#     ("Men",                  g[g.gender == "Man"],                "clean"),
#     ("Age <=15",             d[d.agebin == "le15"],               "explor"),
#     #("Age >=16 (all)",       d[d.agebin == "ge16"],               "explor"),
#     ("Age >=16 ",   d_no_eld[d_no_eld.agebin == "ge16"], "fragile"),
# ]

# res = []
# for lab, sub, kind in rows:
#     med, lo, hi = boot_threshold(sub)
#     pt, b = fit(sub)
#     res.append((lab, pt, lo, hi, kind, len(sub), (1.0 / b if b > 0 else np.nan)))

# dW, loW, hiW, pW = boot_diff(g, "gender", "Woman", "Man")
# dA, loA, hiA, pA = boot_diff(d, "agebin", "le15", "ge16")
# dAe, loAe, hiAe, pAe = boot_diff(d_no_eld, "agebin", "le15", "ge16")

# COL = {"ref": "black", "clean": "#1f77b4", "explor": "#2ca02c",
#        "fragile": "#9467bd", "confound": "#d62728"}
# fig, (axF, axC) = plt.subplots(1, 2, figsize=(15.5 * CM, 6.9 * CM),
#                                gridspec_kw={"width_ratios": [1.45, 1]})

# y = np.arange(len(res))[::-1]
# XMAX = 38.0
# axF.axvline(overall_T, color="0.55", ls="--", lw=0.8, zorder=0)
# axF.text(overall_T, len(res) - 0.4, "  pooled %.1f" % overall_T + DEG,
#          color="0.45", fontsize=5.3, va="bottom", ha="left")
# for yi, (lab, pt, lo, hi, kind, n, w) in zip(y, res):
#     hclip = min(hi, XMAX)
#     axF.plot([lo, hclip], [yi, yi], color=COL[kind], lw=1.5, solid_capstyle="round", zorder=2)
#     if hi > XMAX:  # CI runs off-axis (unstable confounded fit): mark with a caret
#         axF.plot([XMAX], [yi], marker=">", color=COL[kind], ms=3.2, zorder=3)
#         axF.text(XMAX - 0.1, yi + 0.28, "%.0f" % hi, fontsize=4.0, color=COL[kind], ha="right", va="bottom")
#     axF.scatter([pt], [yi], color=COL[kind], s=24, zorder=4, edgecolor="white", linewidth=0.4)
#     nx = min(hi, XMAX) + 0.35
#     axF.text(nx, yi, "n=%d" % n, va="center", ha="left", fontsize=4.4, color="0.55")
# axF.set_yticks(y)
# axF.set_yticklabels([r[0] for r in res], fontsize=6.4)
# axF.set_xlabel(r"comfort$\leftrightarrow$discomfort threshold  $T(m{=}0)$  (" + DEG + ")")
# axF.set_xlim(24, XMAX)
# axF.set_title("threshold by subgroup (walk-stratified 95% CI)", fontsize=7.5)

# labels = [r[0] for r in res]
# i_all = labels.index("Age >=16 (all)")
# i_no = labels.index("Age 16-64 (no 65+)")
# axF.annotate("not significant\n(CI spans pooled)",
#              xy=(res[i_all][3], y[i_all]), xytext=(30.6, y[i_all] + 0.15),
#              fontsize=4.5, color=COL["explor"], va="center", ha="left",
#              arrowprops=dict(arrowstyle="-", color=COL["explor"], lw=0.5))
# axF.annotate("significant only\nwithout 65+",
#              xy=(res[i_no][2], y[i_no]), xytext=(30.6, y[i_no] - 0.15),
#              fontsize=4.5, color=COL["fragile"], va="center", ha="left",
#              arrowprops=dict(arrowstyle="-", color=COL["fragile"], lw=0.5))

# axF.legend(handles=[
#     Line2D([0], [0], color="#1f77b4", lw=2, label="within-walk (clean)"),
#     Line2D([0], [0], color="#2ca02c", lw=2, label="binary age (partly between-walk)"),
#     Line2D([0], [0], color="#9467bd", lw=2, label="elderly-excluded sensitivity"),
#     Line2D([0], [0], color="#d62728", lw=2, label="between-walk (confounded)")],
#     fontsize=4.6, frameon=False, loc="upper right", handlelength=1.4)

# grid = np.linspace(22, 35, 200)
# curves = [("All", d, "black", 2.0),
#           ("Women", g[g.gender == "Woman"], "#d62728", 1.4),
#           ("Men", g[g.gender == "Man"], "#1f77b4", 1.4)]
# for lab, sub, col, lw in curves:
#     T0, b = fit(sub)
#     p = 1.0 / (1.0 + np.exp(-(b * (grid - T0))))
#     axC.plot(grid, p, color=col, lw=lw,
#              label="%s: $T_0$=%.1f, $1/\\beta_1$=%.1f" % (lab, T0, 1 / b))
#     axC.axvline(T0, color=col, ls=":", lw=0.7)
# axC.axhline(0.5, color="0.6", lw=0.6, ls="--")
# axC.annotate("", xy=(overall_T + overall_width / 2, 0.5),
#              xytext=(overall_T - overall_width / 2, 0.5),
#              arrowprops=dict(arrowstyle="<->", color="0.3", lw=0.9))
# axC.text(overall_T, 0.565, "$1/\\beta_1\\approx$%.1f" % overall_width + DEG,
#          ha="center", fontsize=5.6, color="0.25")
# axC.set_xlabel(r"$T_{\rm corr}$  (" + DEG + ")")
# axC.set_ylabel(r"$P(\,U \mid U\ {\rm or}\ C\,)$")
# axC.set_ylim(0, 1)
# axC.set_title(r"finite transition width = populational $\beta^{-1}$", fontsize=7.5)
# axC.legend(fontsize=5.0, frameon=False, loc="lower right")

# fig.subplots_adjust(wspace=0.32)
# save(fig, "F_heterogeneity")

# print("\nOVERALL  T0=%.2f  width 1/b1=%.2f C" % (overall_T, overall_width))
# print("\nsubgroup             T0     CI95               n     width")
# for lab, pt, lo, hi, kind, n, w in res:
#     print("  %-19s %5.2f  [%5.2f,%5.2f]   %4d   %4.2f" % (lab, pt, lo, hi, n, w))
# print("\npaired differences (walk-stratified cluster bootstrap):")
# print("  Women - Men           : %+.2f  [%+.2f,%+.2f]  P(women cooler)=%.3f" % (dW, loW, hiW, 1 - pW))
# print("  Age<=15 - >=16 (all)  : %+.2f  [%+.2f,%+.2f]  P(young warmer)=%.3f  CI includes 0" % (dA, loA, hiA, pA))
# print("  Age<=15 - 16-64       : %+.2f  [%+.2f,%+.2f]  P(young warmer)=%.3f  CI excludes 0" % (dAe, loAe, hiAe, pAe))
"""
F_heterogeneity_age_gender -- Heterogeneity of the comfort<->discomfort threshold
across gender and age subgroups.

Threshold per subgroup:
    T(m=0) = T where P(U)=P(C)

Estimated as -b0/b1 from a logistic regression:
    [uncomfortable vs comfortable] ~ T_corr

Neutral votes are dropped.

95% CIs are obtained from a walk-stratified bootstrap:
    whole walks are resampled with replacement.

This version:
    - only shows the left threshold plot
    - removes the right logistic-curve plot
    - keeps only All votes, Women, Men, Age <=15, Age >=16
    - Age >=16 excludes 65+ internally, but this is not written in the plot
    - uses the requested colour palette
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import set_style, save, CM


# ============================================================
# STYLE / PATHS
# ============================================================

set_style()

HERE = os.path.dirname(os.path.abspath(__file__))

BASE = os.path.join(
    HERE,
    "..","data",
    "markovian_analysis_baseline.csv"
)

RNG = np.random.default_rng(2)
NBOOT = 2000
DEG = r"$^{\circ}$C"


# ============================================================
# PALETTE
# ============================================================

COL_RED = "#F52837"       # Strawberry Red
COL_AQUA = "#9CE5E8"      # Icy Aqua
COL_BLUE = "#2C7AB5"      # Bright Teal Blue
COL_INDIGO = "#152F61"    # Twilight Indigo

COLORS = {
    "ref": COL_INDIGO,
    "woman": COL_RED,
    "man": COL_BLUE,
    "age": COL_AQUA,
}


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

d = pd.read_csv(BASE)

d["city"] = d["city"].replace({
    "L'Hospitalet de Llobreat": "L'Hospitalet de Llobregat"
})

d = d.dropna(subset=["comfort3", "T_corr", "ID", "walk_id"]).copy()

# Keep only comfortable and uncomfortable.
# Neutral is dropped because the threshold is P(U)=P(C).
d = d[d["comfort3"].isin(["comfortable", "uncomfortable"])].copy()

d["U"] = (d["comfort3"] == "uncomfortable").astype(int)


# ------------------------------------------------------------
# Age groups
# ------------------------------------------------------------

YOUNG = {"Less than 10", "10-12", "13-15"}
ELDERLY = {"65-74", "75-84"}

def make_agebin(a):
    if a in YOUNG:
        return "le15"
    elif a in ELDERLY:
        return "elderly"
    elif pd.notna(a):
        return "ge16"
    else:
        return np.nan

d["agebin"] = d["age"].map(make_agebin)

# For the age comparison we exclude 65+ internally.
# The plot label will still simply say Age >=16.
d_age = d[d["agebin"].isin(["le15", "ge16"])].copy()


# ============================================================
# LOGISTIC FIT AND BOOTSTRAP
# ============================================================

def fit(sub, iters=30):
    """
    Logistic regression:
        P(U | U or C) = sigmoid(b0 + b1 T_corr)

    Threshold:
        T0 = -b0 / b1

    Only returns a threshold if b1 > 0.
    """
    if len(sub) < 40 or sub["U"].nunique() < 2:
        return np.nan, np.nan

    x = sub["T_corr"].to_numpy(float)
    y = sub["U"].to_numpy(float)

    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)

    for _ in range(iters):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1.0 - p)

        try:
            step = np.linalg.solve(
                (X * W[:, None]).T @ X + 1e-8 * np.eye(2),
                X.T @ (y - p)
            )
        except np.linalg.LinAlgError:
            return np.nan, np.nan

        beta += step

        if np.max(np.abs(step)) < 1e-9:
            break

    b0, b1 = beta

    if b1 <= 0:
        return np.nan, np.nan

    return -b0 / b1, b1


def boot_threshold(sub, n=NBOOT):
    """
    Walk-stratified bootstrap:
    resample whole walks with replacement.
    """
    wl = sub["walk_id"].dropna().unique()

    if len(wl) == 0:
        return np.nan, np.nan, np.nan

    wi = {
        w: sub.index[sub["walk_id"] == w].to_numpy()
        for w in wl
    }

    out = []

    for _ in range(n):
        chosen_walks = RNG.choice(wl, size=len(wl), replace=True)

        idx = np.concatenate([
            wi[w]
            for w in chosen_walks
        ])

        t, _ = fit(sub.loc[idx])

        if not np.isnan(t):
            out.append(t)

    out = np.asarray(out)

    if len(out) == 0:
        return np.nan, np.nan, np.nan

    return (
        np.median(out),
        np.percentile(out, 10),
        np.percentile(out, 10)
    )


def boot_diff(sub, key, gA, gB, n=NBOOT):
    """
    Paired walk-stratified bootstrap difference:
        T0(group A) - T0(group B)
    """
    wl = sub["walk_id"].dropna().unique()

    if len(wl) == 0:
        return np.nan, np.nan, np.nan, np.nan

    wi = {
        w: sub.index[sub["walk_id"] == w].to_numpy()
        for w in wl
    }

    diffs = []

    for _ in range(n):
        chosen_walks = RNG.choice(wl, size=len(wl), replace=True)

        idx = np.concatenate([
            wi[w]
            for w in chosen_walks
        ])

        r = sub.loc[idx]

        ta, _ = fit(r[r[key] == gA])
        tb, _ = fit(r[r[key] == gB])

        if not np.isnan(ta) and not np.isnan(tb):
            diffs.append(ta - tb)

    diffs = np.asarray(diffs)

    if len(diffs) == 0:
        return np.nan, np.nan, np.nan, np.nan

    return (
        np.median(diffs),
        np.percentile(diffs, 2.5),
        np.percentile(diffs, 97.5),
        np.mean(diffs > 0)
    )


# ============================================================
# SUBGROUPS
# ============================================================

overall_T, overall_b = fit(d)
overall_width = 1.0 / overall_b if overall_b > 0 else np.nan

g = d[d["gender"].isin(["Man", "Woman"])].copy()

rows = [
    ("All votes", d, "ref"),
    ("Women", g[g["gender"] == "Woman"], "woman"),
    ("Men", g[g["gender"] == "Man"], "man"),
    (r"Age $\leq$ 15", d_age[d_age["agebin"] == "le15"], "age"),
    (r"Age $\geq$ 16", d_age[d_age["agebin"] == "ge16"], "age"),
]

res = []

for lab, sub, kind in rows:
    med, lo, hi = boot_threshold(sub)
    pt, b = fit(sub)

    width = 1.0 / b if b > 0 else np.nan

    res.append({
        "label": lab,
        "threshold": pt,
        "lo": lo,
        "hi": hi,
        "kind": kind,
        "n": len(sub),
        "width": width,
    })


# ============================================================
# OPTIONAL DIFFERENCES FOR PRINTING
# ============================================================

dW, loW, hiW, pW = boot_diff(
    g,
    key="gender",
    gA="Woman",
    gB="Man"
)

dA, loA, hiA, pA = boot_diff(
    d_age,
    key="agebin",
    gA="le15",
    gB="ge16"
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    1,
    1,
    figsize=(9.2 * CM, 6.4 * CM)
)

y = np.arange(len(res))[::-1]

XMIN = 24.0
XMAX = 38.0

# pooled reference
ax.axvline(
    overall_T,
    color=COL_INDIGO,
    ls="--",
    lw=0.9,
    alpha=0.65,
    zorder=0
)

ax.text(
    overall_T,
    len(res) - 0.35,
    "  pooled %.1f" % overall_T + DEG,
    color=COL_INDIGO,
    fontsize=5.5,
    va="bottom",
    ha="left"
)

for yi, row in zip(y, res):
    lab = row["label"]
    pt = row["threshold"]
    lo = row["lo"]
    hi = row["hi"]
    kind = row["kind"]
    n = row["n"]

    col = COLORS[kind]

    if np.isnan(pt) or np.isnan(lo) or np.isnan(hi):
        continue

    hclip = min(hi, XMAX)

    # CI line
    ax.plot(
        [lo, hclip],
        [yi, yi],
        color=col,
        lw=1.8,
        solid_capstyle="round",
        zorder=2
    )

    # If CI extends beyond axis
    if hi > XMAX:
        ax.plot(
            [XMAX],
            [yi],
            marker=">",
            color=col,
            ms=3.4,
            zorder=3
        )

        ax.text(
            XMAX - 0.1,
            yi + 0.28,
            "%.0f" % hi,
            fontsize=4.3,
            color=col,
            ha="right",
            va="bottom"
        )

    # Point estimate
    ax.scatter(
        [pt],
        [yi],
        color=col,
        s=28,
        zorder=4,
        edgecolor="white",
        linewidth=0.45
    )

    # n label
    nx = min(hi, XMAX) + 0.35

    ax.text(
        nx,
        yi,
        "n=%d" % n,
        va="center",
        ha="left",
        fontsize=4.7,
        color="0.45"
    )


ax.set_yticks(y)
ax.set_yticklabels(
    [row["label"] for row in res],
    fontsize=6.8
)

ax.set_xlabel(
    r"comfort$\leftrightarrow$discomfort threshold  $T(m{=}0)$  ("
    + DEG +
    ")",
    fontsize=7.2,
    color=COL_INDIGO
)

ax.set_xlim(XMIN, XMAX)

ax.set_title(
    "Threshold heterogeneity by subgroup\n"
    "walk-stratified bootstrap 95% CI",
    fontsize=8.2,
    color=COL_INDIGO
)

ax.tick_params(axis="both", colors=COL_INDIGO)
ax.grid(axis="x", alpha=0.22)
ax.grid(axis="y", alpha=0.08)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()

save(fig, "F_heterogeneity_age_gender")
plt.show()


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nOVERALL")
print("  T0 = %.2f %s" % (overall_T, DEG))
print("  width 1/b1 = %.2f %s" % (overall_width, DEG))

print("\nSUBGROUP THRESHOLDS")
print("  subgroup             T0      CI95                n      width")

for row in res:
    print(
        "  %-18s %6.2f  [%6.2f,%6.2f]   %5d   %5.2f"
        % (
            row["label"].replace("$", "").replace("\\leq", "<=").replace("\\geq", ">="),
            row["threshold"],
            row["lo"],
            row["hi"],
            row["n"],
            row["width"]
        )
    )

print("\nPAIRED DIFFERENCES")
print(
    "  Women - Men          : %+.2f  [%+.2f,%+.2f]  P(Women > Men)=%.3f"
    % (dW, loW, hiW, pW)
)

print(
    "  Age <=15 - Age >=16  : %+.2f  [%+.2f,%+.2f]  P(young warmer)=%.3f"
    % (dA, loA, hiA, pA)
)