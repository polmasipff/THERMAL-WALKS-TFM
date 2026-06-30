"""F5 - Occupation probabilities P(state | T_env) for C/N/U with Wilson 90% CI.
Three curves (C blue, N grey, U red) with Wilson CI bands. Dashed regime boundaries
at T_env = 24.6 and 29.8. Bin width 1.5 C, min n_bin = 20, x from 13 to 39 C.
Size 9x5.5 cm. PNG+PDF."""
import numpy as np
import matplotlib.pyplot as plt
from _common import load, set_style, save, CM, COL_STATE

set_style()
df = load()

BIN_W = 1.5
MIN_N = 20
XMIN, XMAX = 13, 39
Z = 1.645  # 90% CI
BOUNDS = [26.1, 30.77]

states = ["comfortable", "neutral", "uncomfortable"]
slabel = {"comfortable": "P(C | $T_{\\rm env}$)", "neutral": "P(N | $T_{\\rm env}$)",
          "uncomfortable": "P(U | $T_{\\rm env}$)"}

edges = np.arange(XMIN, XMAX + BIN_W, BIN_W)
centers = 0.5 * (edges[:-1] + edges[1:])

x = df["T_corr"].values
st = df["state3"].values

fig, ax = plt.subplots(figsize=(9 * CM, 5.5 * CM))

for s in states:
    xs, ps, los, his = [], [], [], []
    for i in range(len(centers)):
        m = (x >= edges[i]) & (x < edges[i + 1])
        n = m.sum()
        if n < MIN_N:
            continue
        k = np.sum(st[m] == s)
        p = k / n
        # Wilson interval
        denom = 1 + Z**2 / n
        centre = (p + Z**2 / (2 * n)) / denom
        half = (Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2))) / denom
        xs.append(centers[i]); ps.append(p)
        los.append(centre - half); his.append(centre + half)
    xs = np.array(xs)
    ax.fill_between(xs, los, his, color=COL_STATE[s], alpha=0.18, lw=0)
    ax.plot(xs, ps, color=COL_STATE[s], lw=1.4, label=slabel[s], marker="o", ms=2.5)

for b in BOUNDS:
    ax.axvline(b, ls="--", color="0.35", lw=0.8)
ax.text(BOUNDS[0], 1.02, "regime\nboundary", ha="center", va="bottom",
        fontsize=5.5, color="0.35")
ax.text(BOUNDS[1], 1.02, "regime\nboundary", ha="center", va="bottom",
        fontsize=5.5, color="0.35")

ax.set_xlabel(r"Effective temperature $T_{\rm env}$ (°C)")
ax.set_ylabel("Occupation probability")
#ax.set_xlim(XMIN, XMAX)
ax.set_ylim(0, 1.0)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
ax.grid(color="0.93", lw=0.5)
ax.set_axisbelow(True)

save(fig, "F5_pstate_Tenv")
