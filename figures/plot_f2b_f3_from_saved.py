"""
plot_f2b_f3_from_saved.py

Plot-only helper for the TFM F2b/F3 separability figures.
It does NOT recompute bootstrap/permutation curves. It only reads the saved CSVs
produced by compute_f2b_scale_dual_nulls_stable_floor.py and regenerates figures.

Expected files inside --indir:
  _f2b_data_dual_nulls.csv
  F2b_discrimination_floors_pooled_null.csv
  F3_withinwalk_contrast_dual_nulls.csv

Typical use:
  python plot_f2b_f3_from_saved.py --indir f2b_dual_nulls_stable_final --outdir f2b_dual_nulls_stable_final/plots_restyled

Edit the PLOT SETTINGS block below for titles, labels, limits, fontsize, etc.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PLOT SETTINGS -- edit these freely
# ============================================================

COL_ORANGE = "#E07B39"
COL_ORG_FILL = "#F5D0B5"

COL = { 
    "T_corr": "tab:blue",
    "HDX_corr": "0.45",
    "T_env": COL_ORANGE,
}


COORDS = ["T_corr", "HDX_corr", "T_env"]
DISP = {"T_corr": r"$T_{\rm corr}$", "HDX_corr": "HDX", "T_env": r"$T_{\rm env}$"}


FIGSIZE_F2B = (6.4, 3.6)
FIGSIZE_F3 = (6.4, 3.6)
DPI = 300

F2B_XLABEL = r"window width $\Delta$ (SD)"
F2B_YLABEL = r"separability $\varepsilon^2$"
F3_YLABEL = "within-walk range (SD)"



GLOBAL_NULL_LABEL = "pooled permutation null"
WITHIN_NULL_LABEL = "within-walk permutation null"
FLOOR_LABEL_SUFFIX = "floor"

# Set to a number such as 0.09 if you want fixed y-limits across figures.
F2B_YMAX = None
F3_YMAX = None

SHOW_FLOOR_LINES_F2B = True
SHOW_FLOOR_MARKERS_F3 = True
FLOOR_KIND = "robust"  # "robust" uses stable robust floors; "obs" uses stable observed floors

SAVE_EXTS = ["png", "pdf"]

# ============================================================
# Helpers
# ============================================================

def color_for(coord: str, i: int | None = None) -> str:
    if coord in COL:
        return COL[coord]
    return f"C{0 if i is None else i}"


def disp_for(coord: str) -> str:
    return DISP.get(coord, coord)


def plot_f2b(results: pd.DataFrame, floors: pd.DataFrame, outdir: Path, null_name: str, pooled_null: bool = True) -> None:
    null_col = f"null_{null_name}_95"
    fig, ax = plt.subplots(figsize=FIGSIZE_F2B)

    for i, col in enumerate(COORDS):
        if col not in set(results["coord"]):
            continue
        s = results[results.coord == col].sort_values("delta")
        color = color_for(col, i)
        ax.fill_between(
            s.delta.to_numpy(),
            s.lo_plot.to_numpy(),
            s.hi_plot.to_numpy(),
            color=color,
            alpha=0.18,
            zorder=1,
        )
        ax.plot(
            s.delta.to_numpy(),
            s.obs_plot.to_numpy(),
            "-o",
            color=color,
            ms=4,
            lw=1.8,
            label=disp_for(col),
            zorder=3,
        )

    if pooled_null:
        pooled = results.groupby("delta")[null_col].max().sort_index()
        label = GLOBAL_NULL_LABEL if null_name == "global" else WITHIN_NULL_LABEL
        ax.plot(pooled.index.to_numpy(), pooled.values, "--", color="0.55", lw=1.4, label=label, zorder=2)
    else:
        for i, col in enumerate(COORDS):
            if col not in set(results["coord"]):
                continue
            s = results[results.coord == col].sort_values("delta")
            ax.plot(
                s.delta.to_numpy(),
                s[null_col].to_numpy(),
                "--",
                color=color_for(col, i),
                alpha=0.55,
                lw=1.0,
                label=f"{disp_for(col)} {null_name} null 95%",
                zorder=2,
            )

    # if SHOW_FLOOR_LINES_F2B:
    #     fcol = "robust_floor_delta" if FLOOR_KIND == "robust" else "obs_floor_delta"
    #     ft = floors[floors["null"] == null_name]
    #     for i, (_, r) in enumerate(ft.iterrows()):
    #         f = r.get(fcol, np.nan)
    #         coord = r["coord"]
    #         if np.isfinite(f):
    #             ax.axvline(float(f), color=color_for(coord, i), ls=":", lw=1.0, alpha=0.75)

    ax.set_xlim(0.4,float(results["delta"].max()) + 0.05)
    ax.set_xlabel(F2B_XLABEL, fontsize=15)
    ax.set_ylabel(F2B_YLABEL,fontsize=15)
 


    if F2B_YMAX is None:
        y_max = np.nanmax([results[null_col].max(), results["hi_plot"].max(), results["obs_plot"].max()]) * 1.10
        y_max = max(0.02, y_max)
    else:
        y_max = F2B_YMAX
    ax.set_ylim(-0.001, y_max)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.spines[['right', 'top']].set_visible(False)
   
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    fig.tight_layout()

    suffix = "pooled" if pooled_null else "coordspecific"
    for ext in SAVE_EXTS:
        fig.savefig(outdir / f"F2b_scale_std_{null_name}_null_{suffix}_restyled.{ext}", dpi=DPI)
    plt.close(fig)


def plot_f3_contrast(contrast: pd.DataFrame, floors: pd.DataFrame, outdir: Path, null_name: str) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_F3)
    rng = np.random.default_rng(0)
    baseline_label_added = False
    for i, col in enumerate(COORDS):
        if col not in set(contrast["coord"]):
            continue
        s = contrast[contrast.coord == col]
        vals = s["range_sd"].dropna().to_numpy()
        raw = s["range_raw"].dropna().to_numpy()
        color = color_for(col, i)

        bp = ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True, showfliers=False, zorder=2)
        for b in bp["boxes"]:
            b.set(facecolor=color, alpha=0.25, edgecolor=color, lw=1.2)
        for k in ("whiskers", "caps", "medians"):
            for ln in bp[k]:
                ln.set(color=color, lw=1.2)

        ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=18, color=color, alpha=0.55, edgecolor="none", zorder=3)
        #ax.scatter([i], [np.mean(vals)], marker="D", s=38, color=color, edgecolor="white", lw=0.5, zorder=4)
        #ax.text(i, -0.12, f"med {np.median(raw):.1f}\n={np.median(vals):.2f} SD", ha="center", va="top", fontsize=9, color=color)

        if SHOW_FLOOR_MARKERS_F3:
            fcol = "robust_floor_delta" if FLOOR_KIND == "robust" else "obs_floor_delta"
            fr = floors[(floors["coord"] == col) & (floors["null"] == null_name)]
            if not fr.empty:
                f = fr.iloc[0].get(fcol, np.nan)
                if np.isfinite(f):
                    ax.hlines(
                        float(f),
                        i - 0.33,
                        i + 0.33,
                        color=color,
                        ls=":",
                        lw=1.8,
                        zorder=1,
                        label=f"{disp_for(col)} {FLOOR_LABEL_SUFFIX}"
            )

    ax.set_xticks(range(len(COORDS)))
    ax.set_xticklabels([disp_for(c) for c in COORDS], fontsize=15)
    ax.set_ylabel(F3_YLABEL, fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.spines[['right', 'top']].set_visible(False)

    if F3_YMAX is None:
        upper = np.nanpercentile(contrast["range_sd"], 99) * 1.18
        finite_floors = floors.loc[floors["null"] == null_name, "robust_floor_delta"].dropna()
        if len(finite_floors):
            upper = max(upper, finite_floors.max() * 1.25)
    else:
        upper = F3_YMAX
    ax.set_ylim(-0.25, upper)
    fig.tight_layout()
    ax.legend(frameon=False, loc="upper left", fontsize=13, handlelength=2.2)
    for ext in SAVE_EXTS:
        fig.savefig(outdir / f"F3_withinwalk_contrast_{null_name}_floors_restyled.{ext}", dpi=DPI)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate F2b/F3 plots from saved CSVs without recomputing bootstrap/permutation data.")
    parser.add_argument("--indir", type=Path, required=True, help="Directory containing _f2b_data_dual_nulls.csv and floor/contrast CSVs.")
    parser.add_argument("--outdir", type=Path, default=None, help="Directory for restyled plots. Defaults to --indir.")
    parser.add_argument("--plot-coordinate-specific-nulls", action="store_true", help="Also regenerate F2b with coordinate-specific null lines.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    indir = args.indir
    outdir = args.outdir or indir
    outdir.mkdir(parents=True, exist_ok=True)

    results_path = indir / "_f2b_data_dual_nulls.csv"
    floors_path = indir / "F2b_discrimination_floors_pooled_null.csv"
    contrast_path = indir / "F3_withinwalk_contrast_dual_nulls.csv"

    missing = [p for p in [results_path, floors_path, contrast_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required saved file(s):\n" + "\n".join(str(p) for p in missing))

    results = pd.read_csv(results_path)
    floors = pd.read_csv(floors_path)
    contrast = pd.read_csv(contrast_path)

    plot_f2b(results, floors, outdir, null_name="global", pooled_null=True)
    plot_f2b(results, floors, outdir, null_name="within", pooled_null=True)
    if args.plot_coordinate_specific_nulls:
        plot_f2b(results, floors, outdir, null_name="global", pooled_null=False)
        plot_f2b(results, floors, outdir, null_name="within", pooled_null=False)
    plot_f3_contrast(contrast, floors, outdir, null_name="global")
    plot_f3_contrast(contrast, floors, outdir, null_name="within")

    print(f"Restyled plots written to: {outdir}")
    print("Edit the PLOT SETTINGS block at the top of this script to change titles, labels, limits, etc.")


if __name__ == "__main__":
    main()
