# """F_cardinality -- choosing the comfort coarse-graining, two independent criteria, full distribution.
# ML panel uses WALK-grouped cross-validation (consistent with the thesis clustering unit; precomputed in
# _cardinality_walkcv.csv by the from-scratch benchmark, logistic + engineered features). Stats panel:
# fraction of adjacent classes separable along T_corr (Mann-Whitney). All 57 contiguous partitions shown
# (grey), best envelope per k, named candidates highlighted. Both criteria favour few classes; ML peaks
# at k=3; among 3-class cuts the asymmetric VU-isolating partitions lead, symmetric C/N/U is the balanced
# choice (Appendix A)."""
# import os, itertools, numpy as np, pandas as pd, matplotlib.pyplot as plt
# from scipy import stats
# from matplotlib.lines import Line2D
# from _common import set_style, save, CM
# set_style()
# HERE=os.path.dirname(os.path.abspath(__file__))
# ml=pd.read_csv(os.path.join(HERE,"_cardinality_walkcv.csv")); ml["cand"]=ml["cand"].fillna("")
# d=pd.read_csv(os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")).dropna(subset=["category","T_corr"])
# ks=[3,4,5,6,7]
# CCOL={"C/N/U":"black","option 1":"#1f77b4","option 2":"#d62728","4-soft":"#ff7f0e","4-opt1":"#9467bd"}
# OFF={"C/N/U":-0.16,"option 1":-0.06,"option 2":0.06,"4-soft":-0.06,"4-opt1":0.06}

# # stats adjacency for cloud + candidates
# order=['Very uncomfortable','Uncomfortable','Slightly uncomfortable','Neutral(comfort)','Slightly comfortable','Comfortable','Very comfortable']
# order=[c for c in order if c in d.category.unique()]; n=len(order)
# vals={c:d[d.category==c].T_corr.values for c in order}
# def adjf(groups):
#     s=t=0
#     for a,b in zip(groups[:-1],groups[1:]):
#         s+=(stats.mannwhitneyu(np.concatenate([vals[c] for c in a]),np.concatenate([vals[c] for c in b]),alternative="two-sided")[1]<0.05); t+=1
#     return s/t
# def contig(k):
#     for cuts in itertools.combinations(range(1,n),k-1):
#         b=(0,)+cuts+(n,); yield [order[b[i]:b[i+1]] for i in range(k)]
# stats_cloud={k:[adjf(p) for p in contig(k)] for k in ks}
# CANDg={"C/N/U":[['Very uncomfortable','Uncomfortable','Slightly uncomfortable'],['Neutral(comfort)'],['Slightly comfortable','Comfortable','Very comfortable']],
#  "option 1":[['Very uncomfortable'],['Uncomfortable','Slightly uncomfortable'],['Neutral(comfort)','Slightly comfortable','Comfortable','Very comfortable']],
#  "option 2":[['Very uncomfortable'],['Uncomfortable','Slightly uncomfortable','Neutral(comfort)'],['Slightly comfortable','Comfortable','Very comfortable']],
#  "4-soft":[['Very uncomfortable','Uncomfortable'],['Slightly uncomfortable'],['Neutral(comfort)','Slightly comfortable'],['Comfortable','Very comfortable']],
#  "4-opt1":[['Very uncomfortable'],['Uncomfortable','Slightly uncomfortable'],['Neutral(comfort)','Slightly comfortable','Comfortable'],['Very comfortable']]}

# rng=np.random.default_rng(0)
# fig,(axA,axB)=plt.subplots(1,2,figsize=(16*CM,7*CM))
# # panel a (ML, walk-CV)
# bestbal={k:ml[ml.k==k].bal_acc.max() for k in ks}; adj={k:(bestbal[k]-1/k)/(1-1/k) for k in ks}
# for k in ks:
#     v=ml[ml.k==k].bal_acc.values; axA.scatter(k+rng.uniform(-0.13,0.13,len(v)),v,s=10,color="0.8",alpha=0.8,zorder=1,edgecolor="none")
# axA.plot(ks,[bestbal[k] for k in ks],"-",color="0.45",lw=1.1,zorder=2,label="best at each $k$")
# axA.plot(ks,[1/k for k in ks],"--",color="0.6",lw=0.9,zorder=2,label="chance $1/k$")
# for k in ks: axA.text(k,bestbal[k]+0.018,"%.0f%%"%(100*adj[k]),ha="center",fontsize=4.8,color="0.4")
# for _,r in ml[ml.cand!=""].iterrows():
#     c=r["cand"]; axA.scatter([r.k+OFF[c]],[r.bal_acc],marker="D",s=34,color=CCOL[c],zorder=5,edgecolor="white",lw=0.5)
# axA.set_xticks(ks); axA.set_xlabel("number of comfort classes $k$"); axA.set_ylabel("balanced accuracy (walk-grouped CV)")
# axA.set_ylim(0,0.62); axA.set_title("(a) ML recoverability",fontsize=7.5)
# axA.text(0.96,0.96,"labels: adjusted skill\nat best partition",transform=axA.transAxes,fontsize=4.4,color="0.4",ha="right",va="top")
# axA.legend(fontsize=5.0,frameon=False,loc="lower left")
# # panel b (stats)
# for k in ks:
#     v=stats_cloud[k]; axB.scatter(k+rng.uniform(-0.13,0.13,len(v)),v,s=10,color="0.8",alpha=0.8,zorder=1,edgecolor="none")
# axB.plot(ks,[max(stats_cloud[k]) for k in ks],"-",color="0.45",lw=1.1,zorder=2,label="best at each $k$")
# for c,g in CANDg.items():
#     axB.scatter([len(g)+OFF[c]],[adjf(g)],marker="D",s=34,color=CCOL[c],zorder=5,edgecolor="white",lw=0.5)
# axB.axhline(1.0,color="0.85",lw=0.6,ls=":")
# axB.set_xticks(ks); axB.set_xlabel("number of comfort classes $k$")
# axB.set_ylabel("fraction of adjacent pairs separable ($T_{\\rm corr}$)"); axB.set_ylim(-0.05,1.1)
# axB.set_title("(b) classical separability of neighbours",fontsize=7.5)
# handles=[Line2D([0],[0],marker="D",color="w",markerfacecolor=CCOL[c],markeredgecolor="white",ms=6,label=c) for c in CCOL]
# axB.legend(handles=handles,fontsize=5.0,frameon=False,loc="lower left",title="candidates",title_fontsize=5.2)
# fig.subplots_adjust(wspace=0.30)
# save(fig,"F_cardinality")
# print("walk-CV best/adj:",{k:(round(bestbal[k],3),round(adj[k],2)) for k in ks})
"""
F_cardinality -- choosing the comfort coarse-graining, two independent criteria, full distribution.
ML panel uses WALK-grouped cross-validation (consistent with the thesis clustering unit; precomputed in
_cardinality_walkcv.csv by the from-scratch benchmark, logistic + engineered features). Stats panel:
fraction of adjacent classes separable along T_corr (Mann-Whitney). All contiguous partitions shown
(grey), best envelope per k, named 3-class candidates highlighted. Both criteria favour few classes;
ML peaks at k=3; among 3-class cuts the asymmetric VU-isolating partitions lead, symmetric C/N/U is
the balanced choice.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

from _common import set_style, set_paper_style, save, CM

set_style()
set_paper_style()

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Palette: consistent with occupation-regime figure + colorblind-friendly
# ---------------------------------------------------------------------

COLC = "#2C7AB5"   # blue
COLN = "#918686"   # grey
COLU = "#E69F00"   # orange, colorblind-friendly replacement for red

COL_LINE = "#2C7AB5" # green, colorblind-friendly, for best envelope
COL_CHANCE = "0.60"
COL_DARK = "0.20"
COL_LIGHT = "0.78"

CCOL = {
    "C/N/U": COL_DARK,
    # "option 1": COLC,
    # "option 2": COLU
}

OFF = {
    "C/N/U": -0.10,
    # "option 1": 0.00,
    # "option 2": 0.10
}

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

ml = pd.read_csv(os.path.join(HERE, "_cardinality_walkcv.csv"))
ml["cand"] = ml["cand"].fillna("")

d = pd.read_csv(
    os.path.join(
        HERE,
        "..","data",
        "votes_with_Tenv_comfort.csv"
    )
).dropna(subset=["category", "T_corr"])

ks = [3, 4, 5, 6, 7]

# ---------------------------------------------------------------------
# Statistical adjacency criterion
# ---------------------------------------------------------------------

order = [
    "Very uncomfortable",
    "Uncomfortable",
    "Slightly uncomfortable",
    "Neutral(comfort)",
    "Slightly comfortable",
    "Comfortable",
    "Very comfortable"
]

order = [c for c in order if c in d.category.unique()]
n = len(order)

vals = {
    c: d[d.category == c].T_corr.values
    for c in order
}


def adjf(groups):
    s = 0
    t = 0

    for a, b in zip(groups[:-1], groups[1:]):
        x = np.concatenate([vals[c] for c in a])
        y = np.concatenate([vals[c] for c in b])

        p = stats.mannwhitneyu(
            x,
            y,
            alternative="two-sided"
        )[1]

        s += (p < 0.05)
        t += 1

    return s / t


def contig(k):
    for cuts in itertools.combinations(range(1, n), k - 1):
        b = (0,) + cuts + (n,)
        yield [order[b[i]:b[i + 1]] for i in range(k)]


stats_cloud = {
    k: [adjf(p) for p in contig(k)]
    for k in ks
}

# Only keep the 3-class candidates
CANDg = {
    "C/N/U": [
        ["Very uncomfortable", "Uncomfortable", "Slightly uncomfortable"],
        ["Neutral(comfort)"],
        ["Slightly comfortable", "Comfortable", "Very comfortable"]
    ],

    # "option 1": [
    #     ["Very uncomfortable"],
    #     ["Uncomfortable", "Slightly uncomfortable"],
    #     ["Neutral(comfort)", "Slightly comfortable", "Comfortable", "Very comfortable"]
    # ],

    # "option 2": [
    #     ["Very uncomfortable"],
    #     ["Uncomfortable", "Slightly uncomfortable", "Neutral(comfort)"],
    #     ["Slightly comfortable", "Comfortable", "Very comfortable"]
    # ]
}

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

rng = np.random.default_rng(0)

fig, (axA, axB) = plt.subplots(
    1, 2,
    figsize=(19*CM, 9.0*CM)
)

fig.patch.set_facecolor("white")

for ax in (axA, axB):
    ax.set_facecolor("white")


# ---------------------------------------------------------------------
# Left panel: ML recoverability, walk-CV
# ---------------------------------------------------------------------

bestbal = {
    k: ml[ml.k == k].bal_acc.max()
    for k in ks
}

adj = {
    k: (bestbal[k] - 1/k) / (1 - 1/k)
    for k in ks
}

# Full cloud of all partitions
for k in ks:
    v = ml[ml.k == k].bal_acc.values

    axA.scatter(
        k + rng.uniform(-0.13, 0.13, len(v)),
        v,
        s=14,
        color=COL_LIGHT,
        alpha=0.75,
        zorder=1,
        edgecolor="none"
    )

# Best envelope
axA.plot(
    ks,
    [bestbal[k] for k in ks],
    "-",
    color=COL_LINE,
    lw=1.4,
    zorder=2,
    label="best at each $k$"
)

# Chance baseline
axA.plot(
    ks,
    [1/k for k in ks],
    "--",
    color=COL_CHANCE,
    lw=1.1,
    zorder=2,
    label="chance $1/k$"
)

# Adjusted-skill labels at the best partition
for k in ks:
    axA.text(
        k,
        bestbal[k] + 0.020,
        "%.0f%%" % (100 * adj[k]),
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.35"
    )

# Highlight only the selected 3-class candidates
for _, r in ml[ml.cand.isin(CCOL.keys())].iterrows():
    c = r["cand"]

    axA.scatter(
        [r.k + OFF[c]],
        [r.bal_acc],
        marker="D",
        s=48,
        color=CCOL[c],
        zorder=5,
        edgecolor="white",
        lw=0.6
    )

axA.set_xticks(ks)

axA.set_xlabel(
    r"number of classes $k$",
    fontsize=16.5
)

axA.set_ylabel(
    "balanced accuracy",
    fontsize=16.5
)

axA.set_ylim(0, 0.62)

axA.tick_params(
    axis="both",
    labelsize=14
)

axA.legend(
    frameon=False,
    loc="lower left",
    fontsize=12,
    handletextpad=0.4
)


# ---------------------------------------------------------------------
# Right panel: classical separability
# ---------------------------------------------------------------------

# Full cloud of all contiguous partitions
for k in ks:
    v = stats_cloud[k]

    axB.scatter(
        k + rng.uniform(-0.13, 0.13, len(v)),
        v,
        s=14,
        color=COL_LIGHT,
        alpha=0.75,
        zorder=1,
        edgecolor="none"
    )

# Best envelope
axB.plot(
    ks,
    [max(stats_cloud[k]) for k in ks],
    "-",
    color=COL_LINE,
    lw=1.4,
    zorder=2,
    label="best at each $k$"
)

# Highlight only the selected 3-class candidates
for c, g in CANDg.items():
    axB.scatter(
        [len(g) + OFF[c]],
        [adjf(g)],
        marker="D",
        s=48,
        color=CCOL[c],
        zorder=5,
        edgecolor="white",
        lw=0.6
    )

axB.axhline(
    1.0,
    color="0.82",
    lw=0.8,
    ls=":"
)

axB.set_xticks(ks)

axB.set_xlabel(
    r"number of classes $k$",
    fontsize=16.5
)

axB.set_ylabel(
    r"f of separable adj. pairs",
    fontsize=16.5
)

axB.set_ylim(-0.05, 1.10)

axB.tick_params(
    axis="both",
    labelsize=14
)

handles = [
    Line2D(
        [0], [0],
        marker="D",
        color="w",
        markerfacecolor=CCOL[c],
        markeredgecolor="white",
        markersize=7,
        label=c
    )
    for c in CCOL
]

axB.legend(
    handles=handles,
    frameon=False,
    loc="lower left",
    fontsize=12,
    handletextpad=0.4
)


fig.subplots_adjust(
    wspace=0.32,
    left=0.085,
    right=0.985,
    bottom=0.18,
    top=0.95
)

save(fig, "F_cardinality")

print(
    "walk-CV best/adj:",
    {
        k: (round(bestbal[k], 3), round(adj[k], 2))
        for k in ks
    }
)