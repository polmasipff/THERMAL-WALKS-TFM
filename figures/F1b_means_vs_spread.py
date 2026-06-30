"""F1b - The 'honest' version of the category-means figure.
For each of the 7 TCV categories: mean T_corr with (i) a thick 95% CI of the mean
and (ii) a thin +/-1 SD bar showing the actual within-category spread.
No two-line fit, no intersection annotation. The point is that the category means are
statistically ordered (tiny CI) yet practically inseparable (steps ~0.3 C vs spread ~2.5 C).
Size 11x6 cm. PNG 300dpi + PDF."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _common import load, set_style, save, CM

set_style()
df = load()

order = [3, 2, 1, 0, -1, -2, -3]            # comfortable -> uncomfortable
labels = {3: "Very comfortable", 2: "Comfortable", 1: "Slightly comf.",
          0: "Neutral", -1: "Slightly uncomf.", -2: "Uncomfortable",
          -3: "Very uncomf."}

st = {}
for s in order:
    v = df.loc[df.score == s, "T_corr"].dropna().values
    m = v.mean(); sd = v.std(ddof=1); n = len(v)
    st[s] = dict(mean=m, sd=sd, ci=1.96 * sd / np.sqrt(n), n=n)

ypos = {s: i for i, s in enumerate(order[::-1])}  # very uncomf at top

fig, ax = plt.subplots(figsize=(11 * CM, 6 * CM))
CSIDE, USIDE = "#1f77b4", "#d62728"
col = lambda s: CSIDE if s >= 0 else USIDE

for s in order:
    y = ypos[s]; d = st[s]
    ax.plot([d["mean"] - d["sd"], d["mean"] + d["sd"]], [y, y],
            color=col(s), lw=1.2, alpha=0.35, solid_capstyle="round", zorder=2)
    ax.plot([d["mean"] - d["ci"], d["mean"] + d["ci"]], [y, y],
            color=col(s), lw=4.0, alpha=0.95, solid_capstyle="butt", zorder=3)
    ax.scatter([d["mean"]], [y], color=col(s), s=14, zorder=4,
               edgecolors="white", linewidths=0.5)

ax.set_yticks([ypos[s] for s in order])
ax.set_yticklabels([labels[s] for s in order])
ax.set_xlabel(r"$T_{\rm corr}$ (°C)")
ax.set_ylabel("TCV")
ax.set_xlim(22, 34)
ax.grid(axis="x", color="0.92", lw=0.5)
ax.set_axisbelow(True)

legend = [
    Line2D([0], [0], color="0.35", lw=4.0, label="95% CI of the mean"),
    Line2D([0], [0], color="0.35", lw=1.4, alpha=0.4, label=r"$\pm 1$ SD (spread)"),
]
ax.legend(handles=legend, loc="lower left", frameon=True, framealpha=0.9,
          edgecolor="0.8", fontsize=6, handlelength=1.6, borderpad=0.5)

save(fig, "F1b_means_vs_spread")
