"""Standardized cycle current: the cycle AFFINITY A(T) = sum over edges of ln(P_fwd/P_bwd)
around C->N->U->C. A is the scale-free thermodynamic force (entropy production per cycle, k_B),
the full-cycle generalization of R_D = P(onset)/P(recovery): each edge term ln(P_ij/P_ji) is a
local detailed-balance log-ratio. A=0 at detailed balance. Unlike the bare current it is not the
tiny residual of two large opposing fluxes, so it does not look 'small'.
Left: A(T_env) sliding-window with walk bootstrap band, reversing sign near the crossover.
Right: relative current |net|/gross by regime (fraction of flux that circulates)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import set_style, save, CM

set_style()
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..","data", "markovian_analysis_baseline.csv")
ORD = {"comfortable": 0, "neutral": 1, "uncomfortable": 2}
RNG = np.random.default_rng(5)
W = 3.0

d = pd.read_csv(BASE).dropna(subset=["comfort3", "T_env", "ID", "walk_id", "stop_idx"]).copy()
d["s"] = d["comfort3"].map(ORD)
d = d.sort_values(["ID", "walk_id", "stop_idx"])
g = d.groupby(["ID", "walk_id"])
d["s_next"] = g["s"].shift(-1)
tr = d[d["s_next"].notna()].copy(); tr["s_next"] = tr["s_next"].astype(int)
WALKS = tr.walk_id.unique()


def affinity(frame):
    M = np.zeros((3, 3))
    for a, b in zip(frame.s, frame.s_next):
        M[a, b] += 1
    M += 0.5
    P = M / M.sum(1, keepdims=True)
    return (np.log(P[0, 1] / P[1, 0]) + np.log(P[1, 2] / P[2, 1]) + np.log(P[2, 0] / P[0, 2]))


def rel_current(frame):
    M = np.zeros((3, 3))
    for a, b in zip(frame.s, frame.s_next):
        M[a, b] += 1
    M += 0.5
    P = M / M.sum(1, keepdims=True)
    v, V = np.linalg.eig(P.T); pi = np.real(V[:, np.argmin(abs(v - 1))]); pi /= pi.sum()
    def rel(i, j):
        net = pi[i]*P[i, j] - pi[j]*P[j, i]; gross = pi[i]*P[i, j] + pi[j]*P[j, i]
        return net / gross
    return np.mean([abs(rel(0, 1)), abs(rel(1, 2)), abs(rel(0, 2))])


grid = np.linspace(20, 34, 26)
A0, Alo, Ahi = [], [], []
widx = {w: tr.index[tr.walk_id == w].to_numpy() for w in WALKS}
for t in grid:
    sub = tr[(tr.T_env >= t - W) & (tr.T_env < t + W)]
    if len(sub) < 50:
        A0.append(np.nan); Alo.append(np.nan); Ahi.append(np.nan); continue
    A0.append(affinity(sub))
    wl = sub.walk_id.unique(); wi = {w: sub.index[sub.walk_id == w].to_numpy() for w in wl}
    bs = []
    for _ in range(400):
        ch = RNG.choice(wl, size=len(wl), replace=True)
        bs.append(affinity(tr.loc[np.concatenate([wi[w] for w in ch])]))
    Alo.append(np.percentile(bs, 16)); Ahi.append(np.percentile(bs, 84))  # +/-1 sigma band
A0, Alo, Ahi = map(np.array, (A0, Alo, Ahi))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13 * CM, 5.2 * CM), gridspec_kw={"width_ratios": [1.6, 1]})
good = ~np.isnan(A0)
axL.axhspan(-5, 0, color="#1f77b4", alpha=0.05); axL.axhspan(0, 5, color="#d62728", alpha=0.05)
axL.fill_between(grid[good], Alo[good], Ahi[good], color="0.6", alpha=0.3, lw=0)
axL.plot(grid[good], A0[good], "-o", color="black", ms=2.5, lw=1.4)
axL.axhline(0, color="0.3", lw=0.8)
for x in (24.6, 29.8):
    axL.axvline(x, ls=":", color="0.5", lw=0.8)
axL.set_ylim(min(Alo[good]) - 0.1, max(Ahi[good]) + 0.1)
axL.set_xlabel(r"$T_{\rm env}$ (°C)")
axL.set_ylabel(r"cycle affinity $A$ (nats/cycle)")
axL.text(21, max(Ahi[good]), "deterioration\nforce", fontsize=5.5, color="#d62728", va="top")
axL.text(21, min(Alo[good]), "recovery\nforce", fontsize=5.5, color="#1f77b4", va="bottom")
axL.set_title("standardized cycle current (affinity)", fontsize=8)

regs = ["cold", "central", "hot"]
cuts = [24.6, 29.8]
tr["reg"] = np.where(tr.T_env < cuts[0], "cold", np.where(tr.T_env < cuts[1], "central", "hot"))
Avals = [affinity(tr[tr.reg == r]) for r in regs]
Rvals = [rel_current(tr[tr.reg == r]) for r in regs]
xx = np.arange(3)
axR.bar(xx, Avals, color=["#1f77b4", "#9467bd", "#d62728"], alpha=0.85)
axR.axhline(0, color="0.3", lw=0.8)
for i, (a, r) in enumerate(zip(Avals, Rvals)):
    axR.text(i, a + (0.03 if a >= 0 else -0.03), f"A={a:+.2f}\nrel={r:.2f}",
             ha="center", va="bottom" if a >= 0 else "top", fontsize=5.5)
axR.set_xticks(xx); axR.set_xticklabels(["cold", "central", "hot"])
axR.set_ylabel(r"affinity $A$ (nats/cycle)")
axR.set_title("by regime", fontsize=8)
axR.set_ylim(-0.5, 0.8)

save(fig, "F_affinity")
print("affinity by regime:", {r: round(a, 3) for r, a in zip(regs, Avals)})
print("rel current by regime:", {r: round(x, 3) for r, x in zip(regs, Rvals)})
