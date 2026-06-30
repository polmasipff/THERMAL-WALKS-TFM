"""compute_f2b_scale.py
Computes the standardised separability-scale curve eps^2(Delta) for F2b, with
walk-stratified bootstrap CIs and a label-permutation null. Writes _f2b_data.csv,
which F2b_scale_std.py then plots. Run from figures_cap3/.

Definitions (audit these):
- Each coordinate is z-scored over the WHOLE sample (so Delta is in SD units of that coordinate).
- For a window of width Delta centred at c0, take votes with z in [c0-Delta/2, c0+Delta/2).
  A window is used only if it has >=30 votes AND >=5 in each of the 3 states.
- eps2 in a window = (H - k + 1)/(n - k), k=3, with H the Kruskal-Wallis statistic on the
  3 states' z-values inside that window (rank-based; tie-aware via scipy.rankdata).
- The curve value at Delta = MEDIAN of eps2 over the window centres (14 centres between the
  2nd and 98th percentile of z).
- Bootstrap CI: resample WHOLE WALKS with replacement (walk-stratified), recompute the curve;
  report 16-84 percentile band over 120 resamples.
- Null: permute the state labels globally (destroying any T-dependence), recompute the curve;
  report the 95th percentile over 120 permutations.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..","data", "votes_with_Tenv_comfort.csv")
COORDS = ["T_corr", "HDX_corr", "T_env"]
DELTAS = np.round(np.arange(0.4, 3.01, 0.2), 2)   # window widths in SD units
NCENTRES = 14
MIN_WIN = 30          # min votes per window
MIN_GROUP = 5         # min votes per state per window
NBOOT = 120
NPERM = 120
SEED = 0


def eps2(x, lab):
    """Kruskal-Wallis effect size epsilon^2 on 3 groups in one window."""
    n = len(x)
    if n < MIN_WIN or len(np.unique(lab)) < 3:
        return np.nan
    r = rankdata(x)
    H = 12.0 / (n * (n + 1)) * sum((lab == s).sum() * r[lab == s].mean() ** 2
                                   for s in (0, 1, 2)) - 3 * (n + 1)
    return (H - 3 + 1) / (n - 3)


def curve(col, sub):
    """Median-over-windows eps^2(Delta) for one coordinate on subset `sub`."""
    z = ((sub[col] - sub[col].mean()) / sub[col].std()).to_numpy()
    lab = sub["lab"].to_numpy()
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    centres = np.linspace(lo, hi, NCENTRES)
    out = []
    for dl in DELTAS:
        vals = []
        for c0 in centres:
            m = (z >= c0 - dl / 2) & (z < c0 + dl / 2)
            if m.sum() >= MIN_WIN and all((lab[m] == s).sum() >= MIN_GROUP for s in (0, 1, 2)):
                vals.append(eps2(z[m], lab[m]))
        out.append(np.median(vals) if vals else np.nan)
    return np.array(out)


def main():
    d = pd.read_csv(DATA).dropna(subset=["state3"] + COORDS + ["walk"]).copy()
    d["lab"] = d.state3.map({"comfortable": 0, "neutral": 1, "uncomfortable": 2})
    walks = d.walk.unique()
    widx = {w: d.index[d.walk == w].to_numpy() for w in walks}
    rng = np.random.default_rng(SEED)

    rows = []
    for col in COORDS:
        obs = curve(col, d)
        # walk-stratified bootstrap
        B = np.array([curve(col, d.loc[np.concatenate(
            [widx[w] for w in rng.choice(walks, len(walks), replace=True)])]) for _ in range(NBOOT)])
        # label-permutation null
        N = np.array([curve(col, d.assign(lab=rng.permutation(d.lab.values))) for _ in range(NPERM)])
        lo, hi = np.nanpercentile(B, 16, 0), np.nanpercentile(B, 84, 0)
        null95 = np.nanpercentile(N, 95, 0)
        for i, dl in enumerate(DELTAS):
            rows.append({"coord": col, "delta": dl, "obs": obs[i],
                         "lo": lo[i], "hi": hi[i], "null95": null95[i]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "_f2b_data.csv"), index=False)
    print("wrote _f2b_data.csv,", len(out), "rows")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
