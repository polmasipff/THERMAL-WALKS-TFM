"""F2 - Local separability scan for four coordinates (recomputed from raw data).
Left: fraction of sliding windows with significant Kruskal-Wallis (p<0.05) vs window width Delta.
Right: median effect size epsilon^2 vs Delta.
Identical protocol across coordinates: 7 TCV categories, sliding windows, min_n=30, KW + eps^2.
Grey band at Delta = 3.5-4.5 ("separability range"). Size 14x5 cm. PNG+PDF."""
import numpy as np
from scipy.stats import kruskal
import matplotlib.pyplot as plt
from _common import load, set_style, save, CM, COL, DISP

set_style()
df = load()

COORDS = ["T_corr", "HDX", "T_rad", "T_env"]
DELTAS = np.arange(1.0, 8.01, 0.5)
MIN_N = 30          # min votes per window
MIN_CAT = 3        # min distinct TCV categories present
MIN_PER = 4        # min obs in each used category
STRIDE = 0.5       # window step (deg C)

# frozen Delta(50%) values from coordinate_comparison.ipynb (pipeline §6)
FROZEN_D50 = {"T_corr": 4.5, "HDX": 3.5, "T_rad": 5.5, "T_env": 4.0}


def eps2_kw(groups):
    """epsilon^2 effect size for Kruskal-Wallis."""
    n = sum(len(g) for g in groups)
    k = len(groups)
    if n <= k:
        return np.nan, np.nan
    H, p = kruskal(*groups)
    eps2 = (H - k + 1) / (n - k)
    return p, max(eps2, 0.0)


def scan_coord(x, cat):
    x = np.asarray(x); cat = np.asarray(cat)
    lo, hi = np.nanpercentile(x, 1), np.nanpercentile(x, 99)
    frac_sig, med_eps = [], []
    for D in DELTAS:
        sig, eps_vals, nwin = 0, [], 0
        c = lo
        while c + D <= hi + 1e-9:
            m = (x >= c) & (x < c + D)
            if m.sum() >= MIN_N:
                sub_cat = cat[m]
                groups = [x[m][sub_cat == u] for u in np.unique(sub_cat)]
                groups = [g for g in groups if len(g) >= MIN_PER]
                if len(groups) >= MIN_CAT:
                    p, e = eps2_kw(groups)
                    nwin += 1
                    if p < 0.05:
                        sig += 1
                    eps_vals.append(e)
            c += STRIDE
        frac_sig.append(sig / nwin if nwin else np.nan)
        med_eps.append(np.nanmedian(eps_vals) if eps_vals else np.nan)
    return np.array(frac_sig), np.array(med_eps)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(14 * CM, 5 * CM))

for co in COORDS:
    fs, me = scan_coord(df[co].values, df["score"].values)
    axL.plot(DELTAS, fs, color=COL[co], lw=1.3, label=DISP[co])
    axR.plot(DELTAS, me, color=COL[co], lw=1.3, label=DISP[co])

for ax in (axL, axR):
    ax.axvspan(3.5, 4.5, color="0.85", zorder=0)
    ax.set_xlabel(r"Window width $\Delta$ (°C)")
    ax.set_xlim(1, 8)

axL.axhline(0.5, ls="--", color="0.4", lw=0.8)
axL.set_ylabel("Fraction of windows with\nsignificant KW ($p<0.05$)")
axL.set_ylim(0, 1.02)
axL.text(4.0, 0.04, "separability\nrange", ha="center", va="bottom",
         fontsize=6, color="0.35")
axL.legend(loc="upper left", frameon=False)

axR.set_ylabel(r"Median effect size $\varepsilon^2$")
axR.legend(loc="upper left", frameon=False)

save(fig, "F2_scale_separability")

# report recomputed Delta(50%) vs frozen
print("\nDelta where fraction first crosses 0.5 (recomputed):")
for co in COORDS:
    fs, _ = scan_coord(df[co].values, df["score"].values)
    cross = DELTAS[np.argmax(fs >= 0.5)] if np.any(fs >= 0.5) else np.nan
    print(f"  {co:7s} recomputed={cross}  frozen={FROZEN_D50[co]}")
