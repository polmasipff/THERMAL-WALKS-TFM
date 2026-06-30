"""Sun is a regime (inter-walk) forcing, wind is a local (intra-walk) forcing.
Left: within-walk separability |Spearman(coord, comfort)| per walk for three nested
coordinates: T_corr, +sun (T_corr+k_sun f_sun), +sun-wind (=T_env). Adding sun does NOT
improve within-walk separability (paired Wilcoxon ns); adding wind does (p<0.01).
Right: GLOBAL separability (pooled |rho| and AUC) - sun DOES help globally.
Conclusion: sun separates whole walks (regime), wind separates stops within a walk (local).
Size 13x5.5 cm. PNG 300dpi + PDF."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from _common import load, set_style, save, CM

set_style()
df = load()
df["T_sun"] = df.T_corr + 3.0 * df.f_mix + 9.5 * df.f_sun           # +sun, no wind
coords = ["T_corr", "T_sun", "T_env"]
names = [r"$T_{\rm corr}$", "+sun", r"+sun$-$wind"]

# within-walk |rho| per walk
perwalk = {c: [] for c in coords}
walks = []
for w, g in df.groupby("walk"):
    g = g.dropna(subset=["score"])
    if g.stop_row.nunique() < 3 or len(g) < 15:
        continue
    walks.append(w)
    for c in coords:
        perwalk[c].append(abs(stats.spearmanr(g[c], g["score"])[0]))
perwalk = {c: np.array(v) for c, v in perwalk.items()}
means = {c: np.nanmean(perwalk[c]) for c in coords}
p_sun = stats.wilcoxon(perwalk["T_sun"], perwalk["T_corr"])[1]
p_wind = stats.wilcoxon(perwalk["T_env"], perwalk["T_sun"])[1]

# global
glob_rho = {c: abs(stats.spearmanr(df[c], df["score"])[0]) for c in coords}
ybin = (df.state3 == "uncomfortable").astype(int)
glob_auc = {c: roc_auc_score(ybin, df[c]) for c in coords}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13 * CM, 5.5 * CM))
x = np.arange(3)
COLS = ["#1f77b4", "#ff7f0e", "#d62728"]

# Left: within-walk, paired faint lines + mean
rng = np.random.default_rng(0)
for i in range(len(walks)):
    axL.plot(x + rng.uniform(-0.04, 0.04), [perwalk[c][i] for c in coords],
             color="0.8", lw=0.4, alpha=0.5, zorder=1)
for j, c in enumerate(coords):
    axL.scatter(np.full(len(walks), x[j]) + rng.uniform(-0.04, 0.04, len(walks)),
                perwalk[c], s=5, color=COLS[j], alpha=0.5, zorder=2)
    axL.plot([x[j]-0.22, x[j]+0.22], [means[c], means[c]], color=COLS[j], lw=2.5, zorder=3)
    axL.text(x[j], means[c]+0.02, f"{means[c]:.3f}", ha="center", fontsize=6, color=COLS[j], weight="bold")
axL.set_xticks(x); axL.set_xticklabels(names, fontsize=6.5)
axL.set_ylabel(r"within-walk $|\rho|$ (comfort)")
axL.set_title("Within a walk (local)", fontsize=8)
axL.set_ylim(0, None)
axL.annotate(f"+sun: ns\n(p={p_sun:.2f})", (0.5, 0.92), xycoords="axes fraction",
             ha="center", fontsize=6, color="0.4")
axL.annotate(f"+wind: p={p_wind:.3f}", (1.5, 0.92), xycoords="axes fraction",
             ha="center", fontsize=6, color="#d62728")

# Right: global rho and AUC
w = 0.35
axR.bar(x - w/2, [glob_rho[c] for c in coords], width=w, color=COLS, alpha=0.9)
axR.set_ylabel(r"global $|\rho|$", color="0.3")
axR.set_ylim(0, 0.45)
axR2 = axR.twinx()
axR2.plot(x, [glob_auc[c] for c in coords], "o-", color="black", lw=1.2, ms=4)
axR2.set_ylabel("AUC (U vs rest)")
axR2.set_ylim(0.55, 0.75)
for j, c in enumerate(coords):
    axR2.text(x[j], glob_auc[c]+0.006, f"{glob_auc[c]:.2f}", ha="center", fontsize=6)
axR.set_xticks(x); axR.set_xticklabels(names, fontsize=6.5)
axR.set_title("Across all votes (global)", fontsize=8)

fig.suptitle("Sun is a regime (inter-walk) forcing; wind is a local (intra-walk) forcing",
             fontsize=8, y=1.02)
save(fig, "F_sun_wind_scales")
print("within means:", {c: round(means[c],3) for c in coords}, "p_sun", round(p_sun,3), "p_wind", round(p_wind,3))
print("global rho:", {c: round(glob_rho[c],3) for c in coords})
print("AUC:", {c: round(glob_auc[c],3) for c in coords})
