"""
Fixed separability-scale analysis for TFM thermal comfort walks.

What this fixes relative to the older compute_f2b_scale.py:
1) z-scores are computed once on the full observed sample and reused in bootstrap/permutation.
2) window centres are fixed from the observed sample, not recomputed for every bootstrap draw.
3) Kruskal-Wallis H is computed with scipy.stats.kruskal, i.e. with tie correction.
4) epsilon^2 is clipped at 0 by default because negative values are finite-sample noise.
5) bootstrap output keeps percentile, basic/pivotal, normal and centred bands, so you can audit bias.
6) the permutation null can be global or within-walk; within-walk is safer for local/walk-clustered claims.

Example:
python compute_f2b_scale_fixed.py \
  --data "../data/votes_with_Tenv_comfort.csv" \
  --outdir . --nboot 1000 --nperm 500 --ci-mode basic --level 0.68 --permute-within-walk
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -------------------------
# Defaults / configuration
# -------------------------
DEFAULT_COORDS = ["T_corr", "HDX_corr", "T_env"]
DEFAULT_DELTAS = np.round(np.arange(0.4, 3.01, 0.2), 2)
DEFAULT_STATE_MAP = {
    # original 3-state labels used in older script
    "comfortable": 0,
    "neutral": 1,
    "uncomfortable": 2,
    # common alternatives in this project
    "comfortable-neutral": 0,
    "comfortable_neutral": 0,
    "comfortableneutral": 0,
    "slightly uncomfortable": 1,
    "slightly_uncomfortable": 1,
    "slightly-uncomfortable": 1,
    "uncomfortable-basin": 1,
    "uncomfortable_basin": 1,
    "very uncomfortable": 2,
    "very_uncomfortable": 2,
    "very-uncomfortable": 2,
    "very uncomfortable basin": 2,
}
DISP = {"T_corr": r"$T_{\rm corr}$", "HDX_corr": "HDX", "T_env": r"$T_{\rm env}$"}
COL = {"T_corr": "tab:blue", "HDX_corr": "0.45", "T_env": "tab:red"}


@dataclass
class PreparedData:
    df: pd.DataFrame
    coords: List[str]
    walk_col: str
    lab_col: str
    z_cols: Dict[str, str]
    centres: Dict[str, np.ndarray]
    walk_indices: Dict[object, np.ndarray]
    deltas: np.ndarray


def _clean_label(x) -> str:
    return str(x).strip().lower().replace("–", "-").replace("—", "-")


def map_state_labels(s: pd.Series, state_map: Dict[str, int]) -> pd.Series:
    """Map a categorical state column to labels 0,1,2.

    Numeric 0/1/2 columns are accepted directly. String labels are normalised.
    """
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        # accept 0,1,2 or 1,2,3, but convert 1,2,3 to 0,1,2
        vals = set(out.dropna().astype(int).unique())
        if vals.issubset({0, 1, 2}):
            return out.astype("float")
        if vals.issubset({1, 2, 3}):
            return (out - 1).astype("float")
    return s.map(lambda x: state_map.get(_clean_label(x), np.nan)).astype("float")


def kruskal_eps2(x: np.ndarray, lab: np.ndarray, min_win: int, min_group: int, clip_zero: bool = True) -> float:
    """Tie-corrected Kruskal-Wallis epsilon^2 for exactly 3 groups.

    eps^2 = (H - k + 1)/(n - k), k=3. Negative estimates are clipped to 0 by default.
    """
    x = np.asarray(x)
    lab = np.asarray(lab)
    valid = np.isfinite(x) & np.isfinite(lab)
    x, lab = x[valid], lab[valid].astype(int)
    n = len(x)
    if n < min_win:
        return np.nan
    groups = [x[lab == g] for g in (0, 1, 2)]
    if any(len(g) < min_group for g in groups):
        return np.nan
    # scipy.kruskal includes tie correction; manual rank formulas often forget this.
    H, _ = stats.kruskal(*groups, nan_policy="omit")
    k = 3
    eps = (H - k + 1) / (n - k)
    if clip_zero:
        eps = max(0.0, eps)
    return float(eps)


def aggregate_values(vals: List[float], ns: List[int], mode: str) -> float:
    vals_arr = np.asarray(vals, dtype=float)
    good = np.isfinite(vals_arr)
    if not np.any(good):
        return np.nan
    vals_arr = vals_arr[good]
    ns_arr = np.asarray(ns, dtype=float)[good]
    if mode == "median":
        return float(np.nanmedian(vals_arr))
    if mode == "mean":
        return float(np.nanmean(vals_arr))
    if mode == "weighted_mean":
        if np.nansum(ns_arr) <= 0:
            return np.nan
        return float(np.nansum(vals_arr * ns_arr) / np.nansum(ns_arr))
    raise ValueError(f"Unknown aggregator: {mode}")


def curve_from_indices(
    z_all: np.ndarray,
    lab_all: np.ndarray,
    indices: np.ndarray,
    centres: np.ndarray,
    deltas: np.ndarray,
    min_win: int,
    min_group: int,
    aggregator: str,
    clip_zero: bool,
) -> np.ndarray:
    """Compute eps^2(Delta) using fixed z values and fixed window centres.

    `indices` may contain duplicates; this is necessary for cluster bootstrap.
    """
    z = z_all[indices]
    lab = lab_all[indices]
    out = np.full(len(deltas), np.nan)
    for j, delta in enumerate(deltas):
        vals: List[float] = []
        ns: List[int] = []
        half = delta / 2.0
        for c0 in centres:
            m = (z >= c0 - half) & (z < c0 + half)
            n = int(m.sum())
            if n < min_win:
                continue
            # Fast pre-check before scipy call.
            counts = [(lab[m] == g).sum() for g in (0, 1, 2)]
            if any(c < min_group for c in counts):
                continue
            vals.append(kruskal_eps2(z[m], lab[m], min_win, min_group, clip_zero=clip_zero))
            ns.append(n)
        out[j] = aggregate_values(vals, ns, aggregator)
    return out


def bootstrap_intervals(B: np.ndarray, obs: np.ndarray, level: float = 0.68) -> pd.DataFrame:
    """Return several bootstrap bands for audit.

    Important: neither percentile nor basic bootstrap intervals are mathematically required to
    contain the observed point estimate. If you need a visual error band around the observed line,
    use the centred or normal band and label it as an error band, not as a formal percentile CI.
    """
    alpha = (1.0 - level) / 2.0
    qlo, qhi = np.nanquantile(B, [alpha, 1 - alpha], axis=0)
    mean_b = np.nanmean(B, axis=0)
    med_b = np.nanmedian(B, axis=0)
    se_b = np.nanstd(B, axis=0, ddof=1)
    zcrit = stats.norm.ppf(1 - alpha)

    # Percentile: direct quantiles of bootstrap estimates.
    lo_p, hi_p = qlo, qhi
    # Basic/pivotal: mirrors the bootstrap error distribution around theta_hat.
    lo_b, hi_b = 2 * obs - qhi, 2 * obs - qlo
    # Normal/SE band: visually centred at observed estimate.
    lo_n, hi_n = obs - zcrit * se_b, obs + zcrit * se_b
    # Centred bootstrap error band: remove bootstrap mean bias, then add deviations to obs.
    dev = B - mean_b
    dlo, dhi = np.nanquantile(dev, [alpha, 1 - alpha], axis=0)
    lo_c, hi_c = obs + dlo, obs + dhi

    return pd.DataFrame({
        "boot_mean": mean_b,
        "boot_median": med_b,
        "boot_bias": mean_b - obs,
        "boot_se": se_b,
        "lo_percentile": lo_p,
        "hi_percentile": hi_p,
        "lo_basic": lo_b,
        "hi_basic": hi_b,
        "lo_normal": lo_n,
        "hi_normal": hi_n,
        "lo_centered": lo_c,
        "hi_centered": hi_c,
    })


def prepare_data(
    data_path: Path,
    coords: List[str],
    state_col: str,
    walk_col: str,
    deltas: np.ndarray,
    n_centres: int,
    center_q_low: float,
    center_q_high: float,
    state_map: Dict[str, int] = DEFAULT_STATE_MAP,
) -> PreparedData:
    df = pd.read_csv(data_path)
    missing = [c for c in coords + [state_col, walk_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {data_path}: {missing}\nAvailable columns: {list(df.columns)}")
    df = df.dropna(subset=coords + [state_col, walk_col]).copy()
    df["lab"] = map_state_labels(df[state_col], state_map)
    unmapped = df["lab"].isna().sum()
    if unmapped:
        examples = df.loc[df["lab"].isna(), state_col].drop_duplicates().head(20).tolist()
        raise ValueError(
            f"Could not map {unmapped} labels from column {state_col}. Examples: {examples}. "
            "Edit DEFAULT_STATE_MAP or pass a cleaner 0/1/2 state column."
        )
    df["lab"] = df["lab"].astype(int)
    z_cols: Dict[str, str] = {}
    centres: Dict[str, np.ndarray] = {}
    for col in coords:
        mu = df[col].mean()
        sd = df[col].std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(f"Column {col} has invalid sd={sd}")
        zc = f"z__{col}"
        df[zc] = (df[col] - mu) / sd
        z_cols[col] = zc
        lo, hi = np.nanpercentile(df[zc], [center_q_low, center_q_high])
        centres[col] = np.linspace(lo, hi, n_centres)
    walk_indices = {w: df.index[df[walk_col] == w].to_numpy() for w in df[walk_col].drop_duplicates()}
    return PreparedData(df=df, coords=coords, walk_col=walk_col, lab_col="lab", z_cols=z_cols, centres=centres, walk_indices=walk_indices, deltas=deltas)


def permuted_labels(lab_all: np.ndarray, walk_indices: Dict[object, np.ndarray], rng: np.random.Generator, within_walk: bool) -> np.ndarray:
    if not within_walk:
        return rng.permutation(lab_all)
    out = lab_all.copy()
    for idx in walk_indices.values():
        out[idx] = rng.permutation(out[idx])
    return out


def compute(
    prep: PreparedData,
    nboot: int,
    nperm: int,
    level: float,
    ci_mode: str,
    min_win: int,
    min_group: int,
    aggregator: str,
    seed: int,
    permute_within_walk: bool,
    clip_zero: bool,
    outdir: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = prep.df
    full_idx = df.index.to_numpy()
    lab_all = df[prep.lab_col].to_numpy()
    walks = np.array(list(prep.walk_indices.keys()), dtype=object)

    rows = []
    outdir.mkdir(parents=True, exist_ok=True)

    for col in prep.coords:
        print(f"Computing {col} ...")
        z_all = df[prep.z_cols[col]].to_numpy()
        centres = prep.centres[col]

        obs = curve_from_indices(z_all, lab_all, full_idx, centres, prep.deltas, min_win, min_group, aggregator, clip_zero)

        B = np.full((nboot, len(prep.deltas)), np.nan)
        for b in range(nboot):
            sampled_walks = rng.choice(walks, size=len(walks), replace=True)
            idx = np.concatenate([prep.walk_indices[w] for w in sampled_walks])
            B[b] = curve_from_indices(z_all, lab_all, idx, centres, prep.deltas, min_win, min_group, aggregator, clip_zero)
        np.save(outdir / f"_f2b_boot_{col}.npy", B)

        N = np.full((nperm, len(prep.deltas)), np.nan)
        for p in range(nperm):
            lab_perm = permuted_labels(lab_all, prep.walk_indices, rng, within_walk=permute_within_walk)
            N[p] = curve_from_indices(z_all, lab_perm, full_idx, centres, prep.deltas, min_win, min_group, aggregator, clip_zero)
        np.save(outdir / f"_f2b_null_{col}.npy", N)

        ci = bootstrap_intervals(B, obs, level=level)
        null95 = np.nanquantile(N, 0.95, axis=0)
        null50 = np.nanquantile(N, 0.50, axis=0)
        null99 = np.nanquantile(N, 0.99, axis=0)

        for i, delta in enumerate(prep.deltas):
            row = {
                "coord": col,
                "delta": float(delta),
                "obs": float(obs[i]) if np.isfinite(obs[i]) else np.nan,
                "null50": float(null50[i]) if np.isfinite(null50[i]) else np.nan,
                "null95": float(null95[i]) if np.isfinite(null95[i]) else np.nan,
                "null99": float(null99[i]) if np.isfinite(null99[i]) else np.nan,
                "nboot_valid": int(np.isfinite(B[:, i]).sum()),
                "nperm_valid": int(np.isfinite(N[:, i]).sum()),
                "ci_mode_for_plot": ci_mode,
            }
            row.update(ci.iloc[i].to_dict())
            row["lo"] = row[f"lo_{ci_mode}"]
            row["hi"] = row[f"hi_{ci_mode}"]
            row["lo_plot"] = max(0.0, row["lo"]) if np.isfinite(row["lo"]) else np.nan
            row["hi_plot"] = max(0.0, row["hi"]) if np.isfinite(row["hi"]) else np.nan
            row["obs_plot"] = max(0.0, row["obs"]) if np.isfinite(row["obs"]) else np.nan
            rows.append(row)

    res = pd.DataFrame(rows)
    return res


def infer_floor(results: pd.DataFrame, coord_for_floor: str = "T_env", require_ci: bool = True) -> float:
    """First Delta where the chosen lower band is above the pooled 95% null.

    If no such Delta exists, return NaN.
    """
    pooled_null = results.groupby("delta")["null95"].max()
    s = results[results["coord"] == coord_for_floor].sort_values("delta").copy()
    for _, r in s.iterrows():
        threshold = pooled_null.loc[r["delta"]]
        criterion = r["lo_plot"] > threshold if require_ci else r["obs_plot"] > threshold
        if bool(criterion):
            return float(r["delta"])
    return float("nan")


def plot_f2b(results: pd.DataFrame, outdir: Path, floor: float | None = None, y_max: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if floor is not None and np.isfinite(floor):
        ax.axvspan(0, floor, color="0.94", zorder=0)
        ax.axvline(floor, color="0.45", ls=":", lw=1.2)
    for col in DEFAULT_COORDS:
        if col not in set(results["coord"]):
            continue
        s = results[results.coord == col].sort_values("delta")
        ax.fill_between(s.delta.to_numpy(), s.lo_plot.to_numpy(), s.hi_plot.to_numpy(), color=COL.get(col, None), alpha=0.18, zorder=1)
        ax.plot(s.delta.to_numpy(), s.obs_plot.to_numpy(), "-o", color=COL.get(col, None), ms=4, lw=1.8, label=DISP.get(col, col), zorder=3)
    pooled = results.groupby("delta")["null95"].max().sort_index()
    ax.plot(pooled.index.to_numpy(), pooled.values, "--", color="0.55", lw=1.2, label="perm. null (95%)", zorder=2)
    ax.set_xlabel(r"window width  $\Delta$  (SD units of each coordinate)")
    ax.set_ylabel(r"separability  $\varepsilon^2$  (3-state KW)")
    ax.set_xlim(0, max(results["delta"]) + 0.05)
    if y_max is None:
        y_max = np.nanmax([results["hi_plot"].max(), results["obs_plot"].max(), pooled.max()]) * 1.10
        y_max = max(0.02, y_max)
    ax.set_ylim(-0.001, y_max)
    mode = str(results["ci_mode_for_plot"].dropna().iloc[0]) if "ci_mode_for_plot" in results else "bootstrap"
    ax.set_title(f"separability scale, standardised coordinates ({mode} bootstrap band)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"F2b_scale_std_fixed.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f3_contrast(prep: PreparedData, outdir: Path, floor: float | None = None, annotate: str = "median") -> pd.DataFrame:
    df = prep.df
    rows = []
    for col in prep.coords:
        sd = df[col].std(ddof=1)
        grouped = df.groupby(prep.walk_col)[col]
        raw_range = grouped.max() - grouped.min()
        range_sd = raw_range / sd
        rows.append(pd.DataFrame({
            "walk": raw_range.index,
            "coord": col,
            "range_raw": raw_range.values,
            "range_sd": range_sd.values,
        }))
    contrast = pd.concat(rows, ignore_index=True)
    contrast.to_csv(outdir / "F3_withinwalk_contrast_fixed.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    rng = np.random.default_rng(0)
    if floor is not None and np.isfinite(floor):
        ax.axhspan(0, floor, color="0.90", zorder=0)
        ax.axhline(floor, color="0.5", ls="--", lw=1.2)
    for i, col in enumerate(prep.coords):
        s = contrast[contrast.coord == col]
        vals = s["range_sd"].dropna().to_numpy()
        bp = ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True, showfliers=False, zorder=2)
        color = COL.get(col, None)
        for b in bp["boxes"]:
            b.set(facecolor=color, alpha=0.25, edgecolor=color, lw=1.2)
        for k in ("whiskers", "caps", "medians"):
            for ln in bp[k]:
                ln.set(color=color, lw=1.2)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=18, color=color, alpha=0.55, edgecolor="none", zorder=3)
        ax.scatter([i], [np.mean(vals)], marker="D", s=38, color=color, edgecolor="white", lw=0.5, zorder=4)
        raw = s["range_raw"].dropna().to_numpy()
        if annotate == "mean":
            text = f"mean {np.mean(raw):.1f}\n={np.mean(vals):.2f} SD"
        else:
            text = f"med {np.median(raw):.1f}\n={np.median(vals):.2f} SD"
        ax.text(i, -0.12, text, ha="center", va="top", fontsize=9, color=color)
    ax.set_xticks(range(len(prep.coords)))
    ax.set_xticklabels([DISP.get(c, c) for c in prep.coords], fontsize=12)
    ax.set_ylabel("within-walk range (SD units of each coordinate)")
    ax.set_title("within-walk thermal contrast vs the discrimination floor")
    upper = np.nanpercentile(contrast["range_sd"], 99) * 1.18
    if floor is not None and np.isfinite(floor):
        upper = max(upper, floor * 1.4)
    ax.set_ylim(-0.25, upper)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"F3_intrawalk_vs_threshold_fixed.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return contrast


def write_audit_report(results: pd.DataFrame, outdir: Path, args: argparse.Namespace, floor: float) -> None:
    lines = []
    lines.append("F2b separability-scale audit\n")
    lines.append(f"CI mode used for plot: {args.ci_mode}; level={args.level}")
    lines.append(f"Aggregator over windows: {args.aggregator}")
    lines.append(f"Bootstrap: {args.nboot} walk-cluster resamples")
    lines.append(f"Permutation null: {args.nperm}; within_walk={args.permute_within_walk}")
    lines.append(f"Epsilon^2 clipped at zero: {not args.no_clip_zero}")
    lines.append(f"Inferred floor from {args.floor_coord}: {floor if np.isfinite(floor) else 'not found'}")
    lines.append("")
    lines.append("Observed estimate outside percentile bootstrap 16-84/selected level band is not automatically an error; it signals bootstrap bias or skew.")
    lines.append("For publication, report which band is shown. If you want a visual uncertainty band around the observed line, use ci_mode='normal' or ci_mode='centered', not 'percentile'.")
    lines.append("")
    # Coverage diagnostics for each mode (does the observed estimate lie inside the band?)
    for mode in ["percentile", "basic", "normal", "centered"]:
        lo, hi = f"lo_{mode}", f"hi_{mode}"
        ok = (results[lo] <= results["obs"]) & (results["obs"] <= results[hi])
        lines.append(f"Observed inside {mode} interval: {int(ok.sum())}/{len(ok)}")
    lines.append("")
    lines.append("Head of results:")
    lines.append(results.head(20).round(5).to_string(index=False))
    (outdir / "F2b_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed F2b separability-scale bootstrap analysis.")
    parser.add_argument("--data", type=Path, required=True, help="Path to votes_with_Tenv_comfort.csv or equivalent.")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Output directory.")
    parser.add_argument("--coords", nargs="+", default=DEFAULT_COORDS, help="Coordinate columns.")
    parser.add_argument("--state-col", default="state3", help="3-state comfort column. Can be numeric 0/1/2.")
    parser.add_argument("--walk-col", default="walk", help="Walk/cluster column.")
    parser.add_argument("--nboot", type=int, default=1000)
    parser.add_argument("--nperm", type=int, default=500)
    parser.add_argument("--level", type=float, default=0.68, help="Bootstrap interval level. Use 0.95 for 95%% bands.")
    parser.add_argument("--ci-mode", choices=["percentile", "basic", "normal", "centered"], default="basic")
    parser.add_argument("--aggregator", choices=["median", "mean", "weighted_mean"], default="median")
    parser.add_argument("--dmin", type=float, default=0.4)
    parser.add_argument("--dmax", type=float, default=3.0)
    parser.add_argument("--dstep", type=float, default=0.2)
    parser.add_argument("--n-centres", type=int, default=14)
    parser.add_argument("--center-q-low", type=float, default=2.0)
    parser.add_argument("--center-q-high", type=float, default=98.0)
    parser.add_argument("--min-win", type=int, default=30)
    parser.add_argument("--min-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permute-within-walk", action="store_true", help="Shuffle labels within each walk for a cluster-respecting null.")
    parser.add_argument("--no-clip-zero", action="store_true", help="Do not clip negative epsilon^2 values to zero.")
    parser.add_argument("--floor-coord", default="T_env", help="Coordinate used to infer the discrimination floor.")
    parser.add_argument("--contrast-annotate", choices=["median", "mean"], default="median")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    deltas = np.round(np.arange(args.dmin, args.dmax + 1e-12, args.dstep), 10)
    prep = prepare_data(
        data_path=args.data,
        coords=args.coords,
        state_col=args.state_col,
        walk_col=args.walk_col,
        deltas=deltas,
        n_centres=args.n_centres,
        center_q_low=args.center_q_low,
        center_q_high=args.center_q_high,
    )
    print(f"Loaded {len(prep.df)} rows, {prep.df[args.walk_col].nunique()} walks")
    print("State counts:", prep.df["lab"].value_counts().sort_index().to_dict())
    res = compute(
        prep=prep,
        nboot=args.nboot,
        nperm=args.nperm,
        level=args.level,
        ci_mode=args.ci_mode,
        min_win=args.min_win,
        min_group=args.min_group,
        aggregator=args.aggregator,
        seed=args.seed,
        permute_within_walk=args.permute_within_walk,
        clip_zero=not args.no_clip_zero,
        outdir=args.outdir,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    csv_path = args.outdir / "_f2b_data_fixed.csv"
    res.to_csv(csv_path, index=False)
    floor = infer_floor(res, coord_for_floor=args.floor_coord, require_ci=True)
    plot_f2b(res, args.outdir, floor=floor)
    plot_f3_contrast(prep, args.outdir, floor=floor, annotate=args.contrast_annotate)
    # Small config dump for reproducibility.
    config = vars(args).copy()
    config["data"] = str(config["data"])
    config["outdir"] = str(config["outdir"])
    config["deltas"] = deltas.tolist()
    config["inferred_floor"] = floor
    (args.outdir / "F2b_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_audit_report(res, args.outdir, args, floor)
    print(f"Wrote {csv_path}")
    print(f"Inferred discrimination floor from {args.floor_coord}: {floor}")
    print("Wrote F2b_scale_std_fixed.png/pdf and F3_intrawalk_vs_threshold_fixed.png/pdf")


if __name__ == "__main__":
    main()
