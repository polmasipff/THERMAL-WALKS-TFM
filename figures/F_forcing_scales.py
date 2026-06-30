"""Environmental forcing acts at two scales (Chapter 3/6).
Left: within-walk variation ratio (within-walk SD / overall SD) - shows which variables
vary stop-to-stop within a walk vs only between walks.
Right: |Spearman| coupling of each variable to comfort, between-walk vs within-walk.
Message: T_corr/HDX are inter-walk (slow/regime); sun and wind vary at BOTH scales and
behave similarly to each other; T_env combines them and is strongest at both scales.
Size 14x5.5 cm. PNG 300dpi + PDF."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from _common import load, set_style, save, CM

set_style()
df = load()

VARS = ["T_corr", "HDX", "f_sun", "wind_sc", "T_env"]
DISP = {"T_corr": r"$T_{\rm corr}$", "HDX": "HDX", "f_sun": "sun",
        "wind_sc": "wind", "T_env": r"$T_{\rm env}$"}
COL = {"T_corr": "#1f77b4", "HDX": "#7f7f7f", "f_sun": "#ff7f0e",
       "wind_sc": "#2ca02c", "T_env": "#d62728"}

ratio, rb, rw = {}, {}, {}
for v in VARS:
    stop = df.groupby(["walk", "stop_row"])[v].mean().reset_index()
    ratio[v] = stop.groupby("walk")[v].std().mean() / df[v].std()
    d = df[["walk", v, "score"]].dropna().copy()
    g = d.groupby("walk").agg(e=(v, "mean"), c=("score", "mean"))
    rb[v] = abs(stats.spearmanr(g.e, g.c)[0])
    wm_e = d.groupby("walk")[v].transform("mean")
    wm_c = d.groupby("walk")["score"].transform("mean")
    rw[v] = abs(stats.spearmanr(d[v] - wm_e, d["score"] - wm_c)[0])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14 * CM, 5.5 * CM))
x = np.arange(len(VARS))

# Left: variation ratio
axL.bar(x, [ratio[v] for v in VARS], color=[COL[v] for v in VARS], alpha=0.85, width=0.6)
axL.axhline(0.5, ls=":", color="0.5", lw=0.8)
axL.set_xticks(x); axL.set_xticklabels([DISP[v] for v in VARS])
axL.set_ylabel("within-walk SD / overall SD")
axL.set_title("How much each variable varies\nwithin a walk", fontsize=8)
axL.set_ylim(0, 1)
axL.text(0.5, 0.52, "inter-walk  /  varies within", fontsize=5.5, color="0.5")

# Right: between vs within coupling
w = 0.38
axR.bar(x - w/2, [rb[v] for v in VARS], width=w, color=[COL[v] for v in VARS],
        alpha=0.9, label="between-walk")
axR.bar(x + w/2, [rw[v] for v in VARS], width=w, color=[COL[v] for v in VARS],
        alpha=0.45, label="within-walk", hatch="///")
axR.set_xticks(x); axR.set_xticklabels([DISP[v] for v in VARS])
axR.set_ylabel(r"$|\rho|$ with comfort (Spearman)")
axR.set_title("Coupling to comfort,\nby scale", fontsize=8)
from matplotlib.patches import Patch
axR.legend(handles=[Patch(facecolor="0.4", label="between-walk"),
                    Patch(facecolor="0.4", alpha=0.45, hatch="///", label="within-walk")],
           loc="upper left", frameon=False, fontsize=6)

save(fig, "F_forcing_scales")
print("variation ratio:", {v: round(ratio[v], 2) for v in VARS})
print("between |rho|:", {v: round(rb[v], 2) for v in VARS})
print("within |rho|:", {v: round(rw[v], 2) for v in VARS})
