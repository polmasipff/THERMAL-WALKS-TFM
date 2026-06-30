"""Shared computation for the two separability figures of section 3.4:
 (A) within-walk discriminating power: per-walk |rho|(comfort score, coordinate);
 (B) separability scale eps^2(Delta) on STANDARDISED coordinates with permutation null.
Scale-free (A) + scale-fair (B) => the coordinate ranking cannot be a units artefact."""
import os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
VOTES = os.path.join(HERE, "..","data", "votes_with_Tenv_comfort.csv")
COORDS = ["T_corr", "HDX_corr", "T_env"]

def load():
    d = pd.read_csv(VOTES)
    return d

# ---- (A) within-walk |rho| -------------------------------------------------
def within_walk_rho(d):
    out = {c: [] for c in COORDS}; walks = []
    for w, g in d.dropna(subset=["score", "walk"]).groupby("walk"):
        if g.score.nunique() < 2 or len(g) < 5:
            continue
        ok = {c: (g[c].nunique() > 2) for c in COORDS}
        if not all(ok.values()):
            continue
        walks.append(w)
        for c in COORDS:
            out[c].append(abs(stats.spearmanr(g.score, g[c]).correlation))
    return {c: np.array(v) for c, v in out.items()}, walks

def cluster_boot_diff(a, b, n=4000, seed=1):
    """paired walk-level bootstrap of mean(b)-mean(a) (a,b aligned per walk)."""
    rng = np.random.default_rng(seed); m = len(a); out = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        out.append(np.mean(b[idx]) - np.mean(a[idx]))
    out = np.array(out)
    return np.median(out), np.percentile(out, 2.5), np.percentile(out, 97.5), np.mean(out > 0)

# ---- (B) standardised eps^2(Delta) with permutation null -------------------
def _eps2(xw, lw):
    N = len(xw); k = 3
    r = stats.rankdata(xw)
    H = 12.0 / (N * (N + 1)) * sum(len(r[lw == s]) * r[lw == s].mean()**2
                                   for s in (0, 1, 2)) - 3 * (N + 1)
    return (H - k + 1) / (N - k)

def scale_curve(d, col, deltas, nperm=160, seed=0, min_n=30, min_g=5, ncenters=14):
    g = d.dropna(subset=["state3", col]).copy()
    x = ((g[col] - g[col].mean()) / g[col].std()).to_numpy()       # z-score: Delta in SD units
    lab = g.state3.map({"comfortable": 0, "neutral": 1, "uncomfortable": 2}).to_numpy()
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    centers = np.linspace(lo, hi, ncenters)
    rng = np.random.default_rng(seed)
    obs = np.full(len(deltas), np.nan); null95 = np.full(len(deltas), np.nan)
    for i, dl in enumerate(deltas):
        wins = []
        for c0 in centers:
            m = (x >= c0 - dl / 2) & (x < c0 + dl / 2)
            if m.sum() >= min_n:
                lw = lab[m]
                if all((lw == s).sum() >= min_g for s in (0, 1, 2)):
                    wins.append((x[m], lw))
        if not wins:
            continue
        obs[i] = np.median([_eps2(xw, lw) for xw, lw in wins])
        nd = []
        for _ in range(nperm):
            e = [_eps2(xw, rng.permutation(lw)) for xw, lw in wins]
            nd.append(np.median(e))
        null95[i] = np.percentile(nd, 95)
    return obs, null95
