"""F2 (new lead of 3.4): within-walk discriminating power.
Per walk, |rho| between comfort score and each coordinate. Scale-free (rank) and
cluster-respecting (one point per walk). Paired walk-bootstrap + Wilcoxon for T_env, HDX vs T_corr."""
import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from _common import set_style, save, CM, COL, DISP
from _sep_core import load, within_walk_rho, cluster_boot_diff, COORDS
set_style()
d = load(); rho, walks = within_walk_rho(d)
n = len(walks)
cols = {"T_corr": COL["T_corr"], "HDX_corr": COL["HDX"], "T_env": COL["T_env"]}
disp = {"T_corr": r"$T_{\rm corr}$", "HDX_corr": "HDX", "T_env": r"$T_{\rm env}$"}

fig, ax = plt.subplots(figsize=(8.5 * CM, 6.6 * CM))
order = ["T_corr", "HDX_corr", "T_env"]
rng = np.random.default_rng(0)
for i, c in enumerate(order):
    v = rho[c]
    bp = ax.boxplot(v, positions=[i], widths=0.55, patch_artist=True,
                    showfliers=False, zorder=1)
    for b in bp["boxes"]:
        b.set(facecolor=cols[c], alpha=0.25, edgecolor=cols[c], lw=1.0)
    for key in ("whiskers", "caps", "medians"):
        for ln in bp[key]:
            ln.set(color=cols[c], lw=1.0)
    ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=6, color=cols[c],
               alpha=0.55, zorder=2, edgecolor="none")
    ax.scatter([i], [v.mean()], marker="D", s=28, color=cols[c], zorder=4,
               edgecolor="white", lw=0.6)
    ax.text(i, -0.06, "%.3f" % v.mean(), ha="center", fontsize=6, color=cols[c])
ax.set_xticks(range(3)); ax.set_xticklabels([disp[c] for c in order])
ax.set_ylabel(r"within-walk $|\rho|$(comfort, coordinate)")
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.1, 1.0)
ax.set_title("within-walk discriminating power (n=%d walks)" % n, fontsize=7.5)

# paired stats vs T_corr
def paired(c):
    a, b = rho["T_corr"], rho[c]
    W, p = stats.wilcoxon(b, a)
    md, lo, hi, pp = cluster_boot_diff(a, b)
    return p, md, lo, hi
def bracket(i, j, p, dy):
    yt = 0.86 + dy
    ax.plot([i, i, j, j], [yt-0.02, yt, yt, yt-0.02], color="0.3", lw=0.8)
    star = "n.s." if p > 0.05 else ("p<0.001" if p < 1e-3 else "p=%.3f" % p)
    ax.text((i+j)/2, yt+0.005, star, ha="center", fontsize=5.5, color="0.2")
pH, mdH, loH, hiH = paired("HDX_corr")
pE, mdE, loE, hiE = paired("T_env")
bracket(0, 1, pH, 0.0)
bracket(0, 2, pE, 0.08)
save(fig, "F2_withinwalk_rho")
print("within-walk |rho| means:", {c: round(rho[c].mean(), 3) for c in COORDS}, "n=", n)
print("T_env  vs T_corr: dmean=%.3f  CI[%.3f,%.3f]  Wilcoxon p=%.4f" % (mdE, loE, hiE, pE))
print("HDX    vs T_corr: dmean=%.3f  CI[%.3f,%.3f]  Wilcoxon p=%.4f" % (mdH, loH, hiH, pH))
