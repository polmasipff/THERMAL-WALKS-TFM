# """F_occupation_regimes -- bridge into Blume-Capel: the comfort composition P(C/N/U) reorganises
# with the thermal coordinate. Two panels: P(C),P(N),P(U) vs T_corr (left) and vs T_env (right),
# 1C bins, Wilson 95% CI bands, with each coordinate's regime cuts. P(C) falls, P(U) rises and they
# cross (-> m=P(U)-P(C) changes sign), while P(N) stays roughly flat (-> q=P(U)+P(C) ~ const). The
# crossover is sharper in T_env. Data: votes_with_Tenv_comfort.csv (all votes). Paper-style."""
# import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
# from _common import set_style, set_paper_style, save, CM
# from matplotlib.ticker import LinearLocator, FormatStrFormatter 
# set_style(); set_paper_style()
# HERE=os.path.dirname(os.path.abspath(__file__)); DEG=r"$^{\circ}$C"
# D=os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")
# COLC="#2C7AB5"; COLN="#918686"; COLU="#F52837"
# CUTS={"T_corr":(26.10,30.77),"T_env":(24.62,30.83)}
# def wil(k,n,z=1.96):
#     if n==0: return (np.nan,np.nan)
#     p=k/n; dd=1+z*z/n; c=(p+z*z/(2*n))/dd; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd; return c-h,c+h
# d=pd.read_csv(D).dropna(subset=["state3","T_corr","T_env"]).copy()
# def curves(col):
#     lo_,hi_=np.floor(d[col].min()),np.ceil(d[col].max()); edg=np.arange(lo_,hi_+0.01,1.0)
#     g=pd.cut(d[col],edg,right=False); out={"x":[],"C":[],"N":[],"U":[],"Clo":[],"Chi":[],"Nlo":[],"Nhi":[],"Ulo":[],"Uhi":[]}
#     for iv,sub in d.groupby(g,observed=True):
#         n=len(sub)
#         if n<12: continue
#         out["x"].append((iv.left+iv.right)/2)
#         for s,full in [("C","comfortable"),("N","neutral"),("U","uncomfortable")]:
#             k=(sub.state3==full).sum(); out[s].append(k/n); l,h=wil(k,n); out[s+"lo"].append(l); out[s+"hi"].append(h)
#     return {k:np.array(v) for k,v in out.items()}
# fig,(axL,axR)=plt.subplots(1,2,figsize=(17*CM,7*CM),sharey=True)
# fig.subplots_adjust(wspace=0.06,left=0.08,right=0.985,bottom=0.16,top=0.90)
# NTICKS = 7
# for ax,col,xlim,ttl in [(axL,"T_corr",(22,34),"$T_{\\rm corr}$"),(axR,"T_env",(16,34),"$T_{\\rm env}$")]:
#     c=curves(col); c0,c1=CUTS[col]
#     ax.axvspan(xlim[0],c0,color=COLC,alpha=0.05); ax.axvspan(c1,xlim[1],color=COLU,alpha=0.05)
#     for cc in (c0,c1): ax.axvline(cc,color="0.5",ls="--",lw=1.0,zorder=1)
#     for s,col2,lab in [("C",COLC,"P(C)"),("N",COLN,"P(N)"),("U",COLU,"P(U)")]:
#         ax.fill_between(c["x"],c[s+"lo"],c[s+"hi"],color=col2,alpha=0.15,zorder=2)
#         ax.plot(c["x"],c[s],"o-",color=col2,ms=3.5,lw=1.7,zorder=4,label=lab)
#     xticks = np.linspace(xlim[0], xlim[1], NTICKS)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels([f"{x:.0f}" for x in xticks])
#     #ax.set_xlim(*xlim);
#     ax.set_xlabel(r"%s  ("%ttl+DEG+")");
#     #ax.set_title("comfort occupations vs "+ttl,fontsize=9)
# axL.set_ylabel("P(cat.$\\mid$ T)");# axL.set_ylim(0,0.72)a
# axL.legend(frameon=False,loc="upper left",ncol=1,fontsize=8,handletextpad=0.4)
# save(fig,"F_occupation_regimes")
# print("done")

# """
# F_occupation_regimes -- bridge into Blume-Capel: the comfort composition P(C/N/U) reorganises
# with the thermal coordinate. Two panels: P(C),P(N),P(U) vs T_corr (left) and vs T_env (right),
# 1C bins, Wilson 95% CI bands, with each coordinate's regime cuts. P(C) falls, P(U) rises and they
# cross (-> m=P(U)-P(C) changes sign), while P(N) stays roughly flat (-> q=P(U)+P(C) ~ const). The
# crossover is sharper in T_env. Data: votes_with_Tenv_comfort.csv (all votes). Paper-style.
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# from _common import set_style, set_paper_style, save, CM

# set_style()
# set_paper_style()

# HERE = os.path.dirname(os.path.abspath(__file__))
# DEG = r"$^{\circ}$C"

# D = os.path.join(
#     HERE,
#     "..",
# (data path: ../data/  -- see data/README.md)
#     "ORGANISED",
#     "effective_temperature_results_final",
#     "votes_with_Tenv_comfort.csv"
# )

# # Colors principals de les tres ocupacions
# COLC = "#2C7AB5"   # comfortable / cold-like
# COLN = "#918686"   # neutral / central
# COLU = "#F52837"   # uncomfortable / hot-like

# # Talls de règim per a cada coordenada
# CUTS = {
#     "T_corr": (26.10, 30.77),
#     "T_env":  (24.62, 30.83)
# }

# # Límits dels eixos x
# XLIMS = {
#     "T_corr": (22, 34),
#     "T_env":  (16, 34)
# }

# LABELS = {
#     "T_corr": r"$T_{\rm corr}$",
#     "T_env":  r"$T_{\rm env}$"
# }

# # Separació real entre ticks: cada 2 ºC en tots dos panells
# XTICK_STEP = 3

# # Transparència del fons de règims
# REGIME_ALPHA = 0.15


# def wil(k, n, z=1.96):
#     """Wilson 95% CI for a binomial proportion."""
#     if n == 0:
#         return np.nan, np.nan

#     p = k / n
#     dd = 1 + z*z/n
#     c = (p + z*z/(2*n)) / dd
#     h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / dd

#     return c - h, c + h


# # ---------------------------------------------------------------------
# # Load data
# # ---------------------------------------------------------------------

# d = pd.read_csv(D).dropna(subset=["state3", "T_corr", "T_env"]).copy()


# def curves(col):
#     """
#     Compute P(C), P(N), P(U) in 1 ºC bins for a given thermal coordinate.
#     """

#     lo_ = np.floor(d[col].min())
#     hi_ = np.ceil(d[col].max())

#     edges = np.arange(lo_, hi_ + 0.01, 1.0)
#     groups = pd.cut(d[col], edges, right=False)

#     out = {
#         "x": [],
#         "C": [], "N": [], "U": [],
#         "Clo": [], "Chi": [],
#         "Nlo": [], "Nhi": [],
#         "Ulo": [], "Uhi": []
#     }

#     for iv, sub in d.groupby(groups, observed=True):
#         n = len(sub)

#         if n < 12:
#             continue

#         out["x"].append((iv.left + iv.right) / 2)

#         for s, full in [
#             ("C", "comfortable"),
#             ("N", "neutral"),
#             ("U", "uncomfortable")
#         ]:
#             k = (sub.state3 == full).sum()
#             p = k / n
#             lo, hi = wil(k, n)

#             out[s].append(p)
#             out[s + "lo"].append(lo)
#             out[s + "hi"].append(hi)

#     return {k: np.array(v) for k, v in out.items()}


# # ---------------------------------------------------------------------
# # Figure
# # ---------------------------------------------------------------------

# fig, (axL, axR) = plt.subplots(
#     1, 2,
#     figsize=(17*CM, 7*CM),
#     sharey=True
# )

# fig.subplots_adjust(
#     wspace=0.06,
#     left=0.08,
#     right=0.985,
#     bottom=0.16,
#     top=0.90
# )

# # Forçar fons blanc de la figura i dels panells
# fig.patch.set_facecolor("white")

# for ax in (axL, axR):
#     ax.set_facecolor("white")


# for ax, col in [
#     (axL, "T_corr"),
#     (axR, "T_env")
# ]:

#     c = curves(col)

#     xlim = XLIMS[col]
#     ttl = LABELS[col]
#     c0, c1 = CUTS[col]

#     # ------------------------------------------------------------
#     # Fons transparent dels tres règims
#     # ------------------------------------------------------------

#     ax.axvspan(
#         xlim[0], c0,
#         facecolor=COLC,
#         alpha=REGIME_ALPHA,
#         lw=0,
#         zorder=0
#     )

#     ax.axvspan(
#         c0, c1,
#         facecolor=COLN,
#         alpha=REGIME_ALPHA,
#         lw=0,
#         zorder=0
#     )

#     ax.axvspan(
#         c1, xlim[1],
#         facecolor=COLU,
#         alpha=REGIME_ALPHA,
#         lw=0,
#         zorder=0
#     )

#     # Talls de règim
#     for cc in (c0, c1):
#         ax.axvline(
#             cc,
#             color="0.5",
#             ls="--",
#             lw=1.0,
#             zorder=1
#         )

#     # Ocupacions P(C), P(N), P(U)
#     for s, col2, lab in [
#         ("C", COLC, "P(C)"),
#         ("N", COLN, "P(N)"),
#         ("U", COLU, "P(U)")
#     ]:

#         ax.fill_between(
#             c["x"],
#             c[s + "lo"],
#             c[s + "hi"],
#             color=col2,
#             alpha=0.15,
#             zorder=2
#         )

#         ax.plot(
#             c["x"],
#             c[s],
#             "o-",
#             color=col2,
#             ms=3.5,
#             lw=1.7,
#             zorder=4,
#             label=lab
#         )

#     # Eix x
#     ax.set_xlim(*xlim)
    

#     # Mateixa separació real entre ticks en tots dos panells: cada 2 ºC
#     ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
#     ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

#     ax.set_xlabel(ttl + r" (" + DEG + r")")


# # Eix y i llegenda
# axL.set_ylabel(r"$P(\mathrm{cat.}\mid T)$")
# axL.set_ylim(0, 1.0)

# axL.legend(
#     frameon=False,
#     loc="upper left",
#     ncol=1,
#     fontsize=8,
#     handletextpad=0.4
# )

# save(fig, "F_occupation_regimes")
# print("done")

"""
F_occupation_regimes -- bridge into Blume-Capel: the comfort composition P(C/N/U) reorganises
with the thermal coordinate. Two panels: P(C),P(N),P(U) vs T_corr (left) and vs T_env (right),
1C bins, Wilson 95% CI bands, with each coordinate's regime cuts. P(C) falls, P(U) rises and they
cross (-> m=P(U)-P(C) changes sign), while P(N) stays roughly flat (-> q=P(U)+P(C) ~ const). The
crossover is sharper in T_env. Data: votes_with_Tenv_comfort.csv (all votes). Paper-style.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from _common import set_style, set_paper_style, save, CM

set_style()
set_paper_style()

HERE = os.path.dirname(os.path.abspath(__file__))
DEG = r"$^{\circ}$C"

D = os.path.join(
    HERE,
    "..","data",
    "votes_with_Tenv_comfort.csv"
)

# Colors principals de les tres ocupacions
COLC = "#2C7AB5"   # comfortable / cold-like
COLN = "#918686"   # neutral / central
COLU = "#E07B39"  # uncomfortable / hot-like


# Talls de règim per a cada coordenada
CUTS = {
    "T_corr": (26.10, 30.77),
    "T_env":  (24.62, 30.83)
}

# Límits dels eixos x
XLIMS = {
    "T_corr": (22, 34),
    "T_env":  (16, 34)
}

LABELS = {
    "T_corr": r"$T_{\rm corr}$",
    "T_env":  r"$T_{\rm env}$"
}

# Separació real entre ticks en graus
XTICK_STEP = 3

# Transparència del fons de règims
REGIME_ALPHA = 0.105


def wil(k, n, z=1.96):
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return np.nan, np.nan

    p = k / n
    dd = 1 + z*z/n
    c = (p + z*z/(2*n)) / dd
    h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / dd

    return c - h, c + h


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

d = pd.read_csv(D).dropna(subset=["state3", "T_corr", "T_env"]).copy()


def curves(col):
    """
    Compute P(C), P(N), P(U) in 1 ºC bins for a given thermal coordinate.
    """

    lo_ = np.floor(d[col].min())
    hi_ = np.ceil(d[col].max())

    edges = np.arange(lo_, hi_ + 0.01, 1.0)
    groups = pd.cut(d[col], edges, right=False)

    out = {
        "x": [],
        "C": [], "N": [], "U": [],
        "Clo": [], "Chi": [],
        "Nlo": [], "Nhi": [],
        "Ulo": [], "Uhi": []
    }

    for iv, sub in d.groupby(groups, observed=True):
        n = len(sub)

        if n < 12:
            continue

        out["x"].append((iv.left + iv.right) / 2)

        for s, full in [
            ("C", "comfortable"),
            ("N", "neutral"),
            ("U", "uncomfortable")
        ]:
            k = (sub.state3 == full).sum()
            p = k / n
            lo, hi = wil(k, n)

            out[s].append(p)
            out[s + "lo"].append(lo)
            out[s + "hi"].append(hi)

    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

range_corr = XLIMS["T_corr"][1] - XLIMS["T_corr"][0]
range_env  = XLIMS["T_env"][1]  - XLIMS["T_env"][0]

fig, (axL, axR) = plt.subplots(
    1, 2,
    figsize=(19*CM, 8.5*CM),
    sharey=True,
    gridspec_kw={
        "width_ratios": [range_corr, range_env]
    }
)

fig.subplots_adjust(
    wspace=0.10,
    left=0.085,
    right=0.985,
    bottom=0.17,
    top=0.94
)

# Forçar fons blanc de la figura i dels panells
fig.patch.set_facecolor("white")

for ax in (axL, axR):
    ax.set_facecolor("white")


for ax, col in [
    (axL, "T_corr"),
    (axR, "T_env")
]:

    c = curves(col)

    xlim = XLIMS[col]
    ttl = LABELS[col]
    c0, c1 = CUTS[col]

    # ------------------------------------------------------------
    # Fons transparent dels tres règims
    # ------------------------------------------------------------

    ax.axvspan(
        xlim[0], c0,
        facecolor=COLC,
        alpha=REGIME_ALPHA,
        lw=0,
        zorder=0
    )

    ax.axvspan(
        c0, c1,
        facecolor=COLN,
        alpha=REGIME_ALPHA,
        lw=0,
        zorder=0
    )

    ax.axvspan(
        c1, xlim[1],
        facecolor=COLU,
        alpha=REGIME_ALPHA,
        lw=0,
        zorder=0
    )

    # Talls de règim
    # Talls de règim + valor del tall
    # Talls de règim + etiqueta a sobre del plot
    for cc in (c0, c1):
        ax.axvline(
            cc,
            color="0.5",
            ls="--",
            lw=1.0,
            zorder=1
        )

        ax.text(
            cc,
            1.025,
            f"{cc:.2f}",
            transform=ax.get_xaxis_transform(),  # x en dades, y relativa a l'eix
            ha="center",
            va="bottom",
            fontsize=15,
            color="0.35",
            clip_on=False,
            zorder=10
        )

    # Ocupacions P(C), P(N), P(U)
    for s, col2, lab in [
        ("C", COLC, "P(C)"),
        ("N", COLN, "P(N)"),
        ("U", COLU, "P(U)")
    ]:

        ax.fill_between(
            c["x"],
            c[s + "lo"],
            c[s + "hi"],
            color=col2,
            alpha=0.15,
            zorder=2
        )

        ax.plot(
            c["x"],
            c[s],
            "o-",
            color=col2,
            ms=3.5,
            lw=1.7,
            zorder=4,
            label=lab
        )

    # Eix x
    # ax.set_xlim(*xlim)

    # # Mateixa separació real entre ticks en tots dos panells
    # # ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP))
    # # ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    # # Ticks començant exactament a l'inici de l'eix
    # ticks = np.arange(xlim[0], xlim[1] + 1e-9, XTICK_STEP)
    # ax.set_xticks(ticks, fontsize = 14)
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # ax.set_xlabel(ttl + r" (" + DEG + r")", fontsize = 18)

    ax.set_xlim(*xlim)

    # Ticks començant exactament a l'inici de l'eix
    ticks = np.arange(xlim[0], xlim[1] + 1e-9, XTICK_STEP)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # Tamany real dels ticks
    ax.tick_params(axis="both", labelsize=15)

    ax.set_xlabel(ttl + r" (" + DEG + r")", fontsize=18)


# Eix y i llegenda
# axL.set_ylabel(r"$P(\mathrm{cat}\mid T)$", fontsize = 18)
# axL.set_ylim(0, 1.0)
# axL.set_yticks(fontsize = 14)

# axL.legend(
#     frameon=False,
#     loc="upper left",
#     ncol=1,
#     fontsize=14,
#     handletextpad=0.4
# )
axL.set_ylabel(r"$P(\mathrm{cat}\mid T)$", fontsize=19)
axL.set_ylim(0, 1.0)

axL.tick_params(axis="y", labelsize=15)
axR.tick_params(axis="y", labelsize=15)

axL.legend(
    frameon=False,
    loc="upper left",
    ncol=1,
    fontsize=14,
    handletextpad=0.4
)

save(fig, "F_occupation_regimes")
print("done")