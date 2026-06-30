"""F1 (combined) - Overlap of the seven TCV categories along T_corr, with the mean +/- 95% CI
overlaid on the violins. The violins show the (overlapping) within-category distributions;
the thick dark bars show that the category MEANS are pinned to ~+/-0.3 C and ordered within a
2.3 C band -> distributions overlap, yet means are statistically distinguishable and inseparable.
Replaces the separate F1 (violins) + F1b (means). Size 12x6.5 cm. PNG 300dpi + PDF."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _common import load, set_style, save, CM

set_style()
df = load()

order = [3, 2, 1, 0, -1, -2, -3]            # comfortable (top) -> uncomfortable (bottom)
labels = {3: "Very comfortable", 2: "Comfortable", 1: "Slightly comf.",
          0: "Neutral", -1: "Slightly uncomf.", -2: "Uncomfortable", -3: "Very uncomf."}
CSIDE, USIDE,noside = "#9ecae1", "#fcbba1"  , "#8c8c8c"      # violin fills
EDGE = {1: "#3182bd", -1: "#cb181d"}
col_dark = lambda s: "#08519c" if s > 0 else "#a50f15" if s < 0 else "#545455"

print(df.columns)

data = [df.loc[df.score == s, "T_corr"].dropna().values for s in order]
positions = np.arange(len(order))[::-1]    # score 3 at top

fig, ax = plt.subplots(figsize=(12 * CM, 6.5 * CM))

vp = ax.violinplot(data, positions=positions, vert=False, widths=0.9,
                   showmeans=False, showextrema=False, showmedians=False)
for body, s in zip(vp["bodies"], order):
    body.set_facecolor(CSIDE if s > 0 else USIDE if s < 0 else noside)
    body.set_edgecolor("0.5"); body.set_alpha(0.7); body.set_linewidth(0.5)

# overlay mean +/- 95% CI
for s, y in zip(order, positions):
    v = df.loc[df.score == s, "T_corr"].dropna().values
    m = v.mean(); ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    med = np.median(v)
    #ax.plot([med, med], [y - 0.38, y + 0.38], color="0.25", lw=1.2, zorder=3)  # median tick
    ax.plot([m - ci, m + ci], [y, y], color=col_dark(s), lw=4.0, solid_capstyle="butt", zorder=4)
    ax.scatter([m], [y], color="white", edgecolors=col_dark(s), s=20, lw=1.0, zorder=5)

ax.set_yticks(positions)
ax.set_yticklabels([labels[s] for s in order])
ax.set_xlabel(r"$T_{\rm corr}$ (°C)")
ax.set_ylabel("TCV")
ax.set_xlim(12, 40)
ax.grid(False)
ax.set_axisbelow(True)

leg = [Line2D([0], [0], color="0.3", lw=4.0, label="mean ± 95% CI"),
    #    Line2D([0], [0], color="0.25", lw=1.2, label="median"),
       Line2D([0], [0], marker="s", color="0.7", lw=0, markersize=6, label="distribution (violin)")]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
          frameon=False, fontsize=6, handlelength=1.6)

save(fig, "F1_overlap_7cat")
