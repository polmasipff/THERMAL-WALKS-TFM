"""FA1 (appendix) - Scale-space map of comfort-class statistical detectability along T_corr.
Top: observed coverage (votes per bin). Bottom: median -log10(Kruskal-Wallis p) over the
(window width Delta, window center) plane for the 3-state comfort space (C/N/U), using
multi-offset tiling (25 offsets per Delta) to remove bin-edge artefacts.
Dashed line at 29 C. Caption must note the coverage confound. PNG 300dpi + PDF."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
from _common import load, set_style, save, CM

set_style()
df = load()

XCOL = "T_corr"
STATE = "state3"
order3 = ["comfortable", "neutral", "uncomfortable"]
MIN_N, MIN_CLASS_N, N_OFFSETS = 30, 3, 25
DELTA_MIN, DELTA_STEP, DELTA_MAX = 0.5, 0.25, 12.0
CENTER_BIN = DELTA_STEP * 2
CLIP_P = 1e-12

x = pd.to_numeric(df[XCOL], errors="coerce")
lab = df[STATE]
mask = x.notna() & lab.notna()
x = x[mask].values
lab = lab[mask].values
xmin, xmax = x.min(), x.max()
xr = xmax - xmin

deltas = np.arange(DELTA_MIN, min(0.95 * xr, DELTA_MAX) + 1e-6, DELTA_STEP)

rows = []
for D in deltas:
    for off in np.linspace(0, D, N_OFFSETS, endpoint=False):
        edges = np.arange(xmin - D + off, xmax + 2 * D, D)
        idx = np.digitize(x, edges)
        for b in np.unique(idx):
            m = idx == b
            n = int(m.sum())
            if n < MIN_N:
                continue
            sub_lab = lab[m]; sub_x = x[m]
            groups = [sub_x[sub_lab == c] for c in order3]
            groups = [g for g in groups if len(g) >= MIN_CLASS_N]
            if len(groups) < 2:
                continue
            center = 0.5 * (edges[b - 1] + edges[b]) if 0 < b < len(edges) else np.nan
            try:
                H, p = stats.kruskal(*groups)
            except ValueError:
                continue
            rows.append((D, center, -np.log10(max(p, CLIP_P))))

res = pd.DataFrame(rows, columns=["delta", "center", "mlp"])
res["center_bin"] = np.round(res["center"] / CENTER_BIN) * CENTER_BIN
pivot = res.pivot_table(index="delta", columns="center_bin", values="mlp", aggfunc="median")

# figure
fig = plt.figure(figsize=(13 * CM, 9 * CM))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4], hspace=0.08,
                       figure=fig)

# top: coverage
ax0 = fig.add_subplot(gs[0])
bins = np.arange(np.floor(xmin), np.ceil(xmax) + 0.5, 0.5)
ax0.hist(x, bins=bins, color="#6baed6", edgecolor="white", linewidth=0.3)
ax0.axvline(29, ls="--", color="0.2", lw=0.9)
ax0.set_ylabel("Votes\nper bin", fontsize=7)
ax0.set_xlim(xmin, xmax)
ax0.tick_params(labelbottom=False)
ax0.set_title("Observed coverage and scale-space detectability of C/N/U along $T_{\\rm corr}$",
              fontsize=8)

# bottom: heatmap
ax1 = fig.add_subplot(gs[1])
cols = pivot.columns.values
rws = pivot.index.values
extent = [cols.min(), cols.max(), rws.min(), rws.max()]
im = ax1.imshow(pivot.values, aspect="auto", origin="lower", extent=extent,
                cmap="viridis", vmin=0, vmax=12, interpolation="nearest")
ax1.axvline(29, ls="--", color="white", lw=0.9)
ax1.set_xlabel(r"Window center, $T_{\rm corr}$ (°C)")
ax1.set_ylabel(r"Window width $\Delta$ (°C)")
ax1.set_xlim(xmin, xmax)

cbar = fig.colorbar(im, ax=[ax0, ax1], fraction=0.04, pad=0.02)
cbar.set_label(r"median $-\log_{10}(\mathrm{KW}\ p)$", fontsize=7)

save(fig, "FA1_scalespace_heatmap")
print("delta range:", deltas.min(), deltas.max(), "n cells:", len(res))
