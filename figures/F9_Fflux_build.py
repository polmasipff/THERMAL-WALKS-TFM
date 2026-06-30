"""Builds F9 (C/N/U transition matrices by T_env regime) and F-flux (net-flux
triangle diagrams cold vs hot: the constant-q microstate bridge)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from _common import set_style, save, CM

set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..","data", "markovian_analysis_baseline.csv")
ORDER = ["comfortable", "neutral", "uncomfortable"]
LAB = ["C", "N", "U"]
NUM = {s: i for i, s in enumerate(ORDER)}
CUTS = [24.6, 29.8]

d = pd.read_csv(BASE).dropna(subset=["comfort3", "T_env", "ID", "walk_id", "stop_idx"]).copy()
d["s"] = d["comfort3"].map(NUM)
d = d.sort_values(["ID", "walk_id", "stop_idx"])
g = d.groupby(["ID", "walk_id"])
d["s_next"] = g["s"].shift(-1)
tr = d[d["s_next"].notna()].copy(); tr["s_next"] = tr["s_next"].astype(int)
tr["reg"] = np.where(tr.T_env < CUTS[0], "cold", np.where(tr.T_env < CUTS[1], "central", "hot"))


def P_pi_F(frame):
    M = np.zeros((3, 3))
    for a, b in zip(frame.s, frame.s_next):
        M[a, b] += 1
    P = M / M.sum(1, keepdims=True)
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))]); pi = pi / pi.sum()
    F = np.array([[pi[i]*P[i, j] - pi[j]*P[j, i] for j in range(3)] for i in range(3)])
    return P, pi, F

regs = ["cold", "central", "hot"]
titles = {"cold": "Cold  (<24.6 °C)", "central": "Central  (24.6–29.8)", "hot": "Hot  (>29.8 °C)"}
res = {r: P_pi_F(tr[tr.reg == r]) for r in regs}

# ---------- F9: three transition-matrix heatmaps ----------
fig, axes = plt.subplots(1, 3, figsize=(14 * CM, 5 * CM))
for ax, r in zip(axes, regs):
    P = res[r][0]
    im = ax.imshow(P, cmap="magma_r", vmin=0, vmax=0.6)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{P[i,j]:.2f}", ha="center", va="center",
                    color="white" if P[i, j] > 0.33 else "black", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(LAB)
    ax.set_yticks(range(3)); ax.set_yticklabels(LAB)
    ax.set_title(titles[r], fontsize=7.5)
    ax.set_xlabel("to");
    if r == "cold": ax.set_ylabel("from")
fig.suptitle(r"C/N/U transition probabilities by $T_{\rm env}$ regime", fontsize=8, y=1.02)
save(fig, "F9_regime_matrices")

# ---------- F-flux: net-flux triangles cold vs hot ----------
pos = {0: (0.5, 0.88), 1: (0.12, 0.18), 2: (0.88, 0.18)}  # C top, N left, U right
COLP = {0: "#1f77b4", 1: "#7f7f7f", 2: "#d62728"}

def draw_panel(ax, F, pi, title, Jc):
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.05, 1.05); ax.axis("off")
    ax.set_title(title, fontsize=8)
    # nodes sized by stationary pi
    for i, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.07 + 0.18*pi[i], color=COLP[i], alpha=0.85, zorder=3))
        ax.text(x, y, LAB[i], ha="center", va="center", color="white", fontsize=9, zorder=4, weight="bold")
    # net-flux arrows for the 3 edges, direction = sign of net flux
    edges = [(0, 1), (1, 2), (0, 2)]
    scale = 18.0
    for (i, j) in edges:
        f = F[i, j]
        a, b = (i, j) if f >= 0 else (j, i)  # arrow points along net flux
        (x1, y1), (x2, y2) = pos[a], pos[b]
        # shorten to node edges
        dx, dy = x2 - x1, y2 - y1; L = np.hypot(dx, dy); ux, uy = dx/L, dy/L
        r1 = 0.07 + 0.18*pi[a]; r2 = 0.07 + 0.18*pi[j if f >= 0 else i]
        p1 = (x1 + ux*r1, y1 + uy*r1); p2 = (x2 - ux*r2, y2 - uy*r2)
        lw = 0.5 + scale*abs(f)
        col = "#b2182b" if (a, b) in [(0, 1), (1, 2), (0, 2)] else "#2166ac"
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9,
                     lw=lw, color=col, alpha=0.9, zorder=2))
        ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2, f"{abs(f):.3f}", fontsize=5.5,
                color="0.3", ha="center")
    dirn = "C→N→U→C" if Jc > 0 else "C→U→N→C"
    ax.text(0.5, -0.03, f"circulation {dirn}\n$J$ = {Jc:+.4f}", ha="center", va="top",
            fontsize=6.5, color="#b2182b" if Jc > 0 else "#2166ac")

fig2, (axc, axh) = plt.subplots(1, 2, figsize=(13 * CM, 6.6 * CM))
Pc, pic, Fc = res["cold"]; Ph, pih, Fh = res["hot"]
Jc_c = (Fc[0, 1] + Fc[1, 2] + Fc[2, 0]) / 3
Jc_h = (Fh[0, 1] + Fh[1, 2] + Fh[2, 0]) / 3
draw_panel(axc, Fc, pic, "Cold regime  ($J<0$)", Jc_c)
draw_panel(axh, Fh, pih, "Hot regime  ($J>0$)", Jc_h)
fig2.text(0.5, 0.97, "Net probability circulation reverses across the crossover",
          ha="center", fontsize=8.5, weight="bold")
fig2.subplots_adjust(top=0.88, bottom=0.05)
save(fig2, "F_flux_reversal")
print("cold Jc", round(Jc_c, 5), "hot Jc", round(Jc_h, 5))
