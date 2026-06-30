# 

"""F_regime_count_by_coordinate -- regime count along T_corr, HDX_corr and T_env.
Same calculation and same plot logic as the original T_env figure:
piecewise-constant multinomial segmentation of the C/N/U composition,
breakpoints by maximum likelihood (DP), BIC + marginal gain ΔG.
The selected k is chosen by elbow/parsimony, not automatically by min(BIC).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _common import set_style, save, CM
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import viridis

set_style()

HERE = os.path.dirname(os.path.abspath(__file__))

d = pd.read_csv(
    os.path.join(
        HERE, "..","data",
        "votes_with_Tenv_comfort.csv"
    )
)

# ------------------------------------------------------------
# Coordinates to analyse
# ------------------------------------------------------------
COORDS = {
    "T_corr": r"$T_{\mathrm{corr}}$",
    "HDX_corr": r"HDX$_{\mathrm{corr}}$",
    "T_env": r"$T_{\mathrm{env}}$",
}

# Elbow/parsimony selection.
# Change these after looking at the BIC + ΔG plots.
SELECTED_K = {
    "T_corr": 3,
    "HDX_corr": 3,
    "T_env": 3,
}

state_map = {
    "comfortable": 0,
    "neutral": 1,
    "uncomfortable": 2,
}

# ------------------------------------------------------------
# Segmentation function: same logic as your original code,
# but now it also returns the breakpoints.
# ------------------------------------------------------------
def regime_scan(df, x_col, state_col="state3", ks=range(1, 6), n_cand=70, margin=40):
    sub = df.dropna(subset=[x_col, state_col]).copy()
    sub["lab"] = sub[state_col].map(state_map)
    sub = sub.dropna(subset=["lab"]).copy()

    x = sub[x_col].to_numpy()
    lab = sub["lab"].astype(int).to_numpy()

    o = np.argsort(x)
    x = x[o]
    lab = lab[o]
    n = len(x)

    if n < 2 * margin + 10:
        raise ValueError(f"Not enough observations for {x_col}: n={n}")

    cand = np.unique(np.linspace(margin, n - margin, n_cand).astype(int))

    cum = np.zeros((n + 1, 3))
    for s in range(3):
        cum[1:, s] = np.cumsum(lab == s)

    def sll(i, j):
        c = cum[j] - cum[i]
        m = c.sum()
        if m == 0:
            return 0.0
        p = c[c > 0] / m
        return (c[c > 0] * np.log(p)).sum()

    def best_ll_and_breaks(k):
        cuts = [0] + list(cand) + [n]
        NEG = -1e18

        f = np.full((k + 1, len(cuts)), NEG)
        prev = np.full((k + 1, len(cuts)), -1, dtype=int)

        f[0, 0] = 0.0

        for s in range(1, k + 1):
            for ci in range(1, len(cuts)):
                best = NEG
                best_pi = -1

                for pi in range(ci):
                    if f[s - 1, pi] > NEG:
                        v = f[s - 1, pi] + sll(cuts[pi], cuts[ci])
                        if v > best:
                            best = v
                            best_pi = pi

                f[s, ci] = best
                prev[s, ci] = best_pi

        # Backtrack selected cut indices
        ci = len(cuts) - 1
        bp = []

        for s in range(k, 0, -1):
            pi = prev[s, ci]
            if pi == -1:
                break
            bp.append(cuts[pi])
            ci = pi

        bp = sorted([b for b in bp if b not in (0, n)])

        # Convert index breakpoints into coordinate values.
        # The regime boundary is between x[b-1] and x[b].
        bp_values = []
        for b in bp:
            if 0 < b < n:
                bp_values.append(0.5 * (x[b - 1] + x[b]))

        return f[k, -1], bp, bp_values

    LL = {}
    break_idx = {}
    break_values = {}

    for k in ks:
        ll, bp, bpv = best_ll_and_breaks(k)
        LL[k] = ll
        break_idx[k] = bp
        break_values[k] = bpv

    # Same BIC penalty as your original code:
    # 3 categories -> 2 probabilities per segment + one breakpoint parameter
    # p(k) = 2k + (k-1) = 3k - 1
    bic = {
        k: -2 * LL[k] + (3 * k - 1) * np.log(n)
        for k in ks
    }

    dG = {
        k: 2 * (LL[k] - LL[k - 1])
        for k in ks
        if k >= 2
    }

    return {
        "n": n,
        "x_sorted": x,
        "LL": LL,
        "bic": bic,
        "dG": dG,
        "break_idx": break_idx,
        "break_values": break_values,
    }


# ------------------------------------------------------------
# Plot function: same style as the original plot
# ------------------------------------------------------------
def plot_regime_count(res, x_col, x_label, selected_k):
    kk = list(res["bic"].keys())
    bic = res["bic"]
    dG = res["dG"]

    fig, ax = plt.subplots(figsize=(8.8 * CM, 6.4 * CM))

    COL_BIC = viridis(0.62)
    COL_GAIN = "0.78"
    COL_GAIN_EDGE = "0.65"
    COL_SEL = COL_BIC
    COL_REF = "0.35"

    # --------------------------------------------------------
    # Left axis: BIC
    # --------------------------------------------------------
    ax.plot(
        kk,
        [bic[k] for k in kk],
        "-o",
        color=COL_BIC,
        markerfacecolor=COL_BIC,
        markeredgecolor="white",
        markeredgewidth=0.4,
        ms=4.2,
        lw=1.45,
        zorder=4
    )

    ax.axvline(
        selected_k,
        color=COL_REF,
        lw=0.75,
        ls=":",
        zorder=1
    )

    ax.scatter(
        [selected_k],
        [bic[selected_k]],
        s=90,
        facecolor="white",
        edgecolor=COL_SEL,
        lw=1.6,
        zorder=6
    )

    ax.text(
        selected_k + 0.08,
        bic[selected_k] + 7,
        rf"selected $k={selected_k}$",
        fontsize=5.6,
        color=COL_REF,
        ha="left",
        va="bottom"
    )

    ax.set_xlabel(r"number of regimes $k$")
    ax.set_ylabel("BIC", color=COL_BIC)
    ax.tick_params(axis="y", labelcolor=COL_BIC)
    ax.set_xticks(kk)
    ax.set_xlim(min(kk) - 0.35, max(kk) + 0.35)

    # --------------------------------------------------------
    # Right axis: marginal gain
    # --------------------------------------------------------
    ax2 = ax.twinx()

    bar_x = list(dG.keys())
    bar_y = [dG[k] for k in bar_x]

    ax2.bar(
        bar_x,
        bar_y,
        width=0.50,
        color=COL_GAIN,
        edgecolor=COL_GAIN_EDGE,
        linewidth=0.35,
        alpha=0.70,
        zorder=0
    )

    ax2.set_ylabel(r"marginal gain $\Delta G$", color="0.45")
    ax2.tick_params(axis="y", labelcolor="0.45")

    for k in dG:
        ax2.text(
            k,
            dG[k] + 3,
            f"{dG[k]:.0f}",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="0.35"
        )

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------
    legend_handles = [
        Line2D(
            [0], [0],
            color=COL_BIC,
            lw=1.45,
            marker="o",
            markerfacecolor=COL_BIC,
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.2,
            label="BIC"
        ),
        Patch(
            facecolor=COL_GAIN,
            edgecolor=COL_GAIN_EDGE,
            linewidth=0.35,
            alpha=0.70,
            label=r"marginal gain $\Delta G$"
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor=COL_SEL,
            markeredgewidth=1.5,
            markersize=6,
            label=rf"selected $k={selected_k}$"
        )
    ]

    ax.legend(
        handles=legend_handles,
        fontsize=5.4,
        frameon=False,
        loc="upper right",
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0.3
    )

    ax.grid(axis="y", alpha=0.18, lw=0.5)
    ax.set_title(x_label, fontsize=7.0)

    safe_name = x_col.replace("_", "")
    save(fig, f"F_regime_count_{safe_name}")

    return fig, ax


# ------------------------------------------------------------
# Run all coordinates
# ------------------------------------------------------------
all_results = {}
summary_rows = []

for x_col, x_label in COORDS.items():
    print("\n" + "=" * 70)
    print(f"Coordinate: {x_col}")

    res = regime_scan(d, x_col)
    all_results[x_col] = res

    selected_k = SELECTED_K[x_col]

    plot_regime_count(
        res=res,
        x_col=x_col,
        x_label=x_label,
        selected_k=selected_k
    )

    print("n:", res["n"])
    print("BIC:", {k: round(res["bic"][k], 1) for k in res["bic"]})
    print("dG :", {k: round(res["dG"][k], 1) for k in res["dG"]})
    print(f"selected k by elbow/parsimony: {selected_k}")
    print(
        "selected breakpoints:",
        [round(v, 2) for v in res["break_values"][selected_k]]
    )

    # Print breakpoints for all k as a diagnostic
    print("breakpoints by k:")
    for k in res["break_values"]:
        print(f"  k={k}: {[round(v, 2) for v in res['break_values'][k]]}")

    for k in res["bic"]:
        summary_rows.append({
            "coordinate": x_col,
            "k": k,
            "n": res["n"],
            "LL": res["LL"][k],
            "BIC": res["bic"][k],
            "dG_from_previous": res["dG"].get(k, np.nan),
            "breakpoints": ", ".join(f"{v:.3f}" for v in res["break_values"][k]),
            "selected_k": selected_k,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv("regime_count_by_coordinate_summary.csv", index=False)
print("\nSaved summary: regime_count_by_coordinate_summary.csv")