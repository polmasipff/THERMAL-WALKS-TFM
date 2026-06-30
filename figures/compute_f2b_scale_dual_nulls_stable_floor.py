"""
compute_f2b_scale_dual_nulls.py

Scale-dependent separability analysis for the TFM thermal comfort walks.

This version computes, in one run:
1) the observed eps^2(Delta) curves for T_corr, HDX_corr and T_env;
2) walk-cluster bootstrap uncertainty bands;
3) a GLOBAL/POOLED permutation null;
4) a WITHIN-WALK permutation null;
5) coordinate-specific discrimination floor tables for each null, reporting both first-crossing and stable floors.

Key definitions:
- Coordinates are z-scored ONCE on the full observed sample.
- Window centres are fixed from the observed sample.
- eps^2 is Kruskal-Wallis epsilon squared for 3 comfort states.
- Bootstrap resamples whole walks with replacement.
- Global null permutes labels over the full pooled dataset.
- Within-walk null permutes labels independently inside each walk.
- The recommended floor is the stable floor: the first Delta after which the curve stays above the null.

Example:
python compute_f2b_scale_dual_nulls.py \
  --data "../data/votes_with_Tenv_comfort.csv" \
  --outdir "f2b_dual_nulls" \
  --nboot 1000 --nperm 500 \
  --ci-mode centered --level 0.68 \
  --aggregator median
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DEFAULT_COORDS = ["T_corr", "HDX_corr", "T_env"]
STATE3_MAP = {
    "comfortable": 0,
    "neutral": 1,
    "uncomfortable": 2,
}
BASIN3_MAP = {
    "comfortable-neutral": 0,
    "comfortable_neutral": 0,
    "comfortableneutral": 0,
    "comfortable": 0,
    "neutral": 0,
    "slightly uncomfortable": 1,
    "slightly_uncomfortable": 1,
    "slightly-uncomfortable": 1,
    "uncomfortable": 1,
    "uncomfortable-basin": 1,
    "uncomfortable_basin": 1,
    "very uncomfortable": 2,
    "very_uncomfortable": 2,
    "very-uncomfortable": 2,
    "very uncomfortable basin": 2,
}
DEFAULT_STATE_MAP = {**STATE3_MAP, **{k: v for k, v in BASIN3_MAP.items() if k not in STATE3_MAP}}
DISP = {"T_corr": r"$T_{\rm corr}$", "HDX_corr": "HDX", "T_env": r"$T_{\rm env}$"}
COL = {"T_corr": "tab:blue", "HDX_corr": "0.45", "T_env": "tab:red"}


def color_for(coord: str, i: int | None = None):
    if coord in COL:
        return COL[coord]
    if "HDX" in coord.upper():
        return "0.45"
    if "T_ENV" in coord.upper() or coord == "T_env":
        return "tab:red"
    if "T" in coord.upper():
        return "tab:blue"
    return f"C{0 if i is None else i}"


def disp_for(coord: str) -> str:
    if coord in DISP:
        return DISP[coord]
    if "HDX" in coord.upper():
        return "HDX"
    return coord


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
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        vals = set(out.dropna().astype(int).unique())
        if vals.issubset({0, 1, 2}):
            return out.astype(float)
        if vals.issubset({1, 2, 3}):
            return (out - 1).astype(float)
    return s.map(lambda x: state_map.get(_clean_label(x), np.nan)).astype(float)


def kruskal_eps2(x: np.ndarray, lab: np.ndarray, min_win: int, min_group: int, clip_zero: bool = True) -> float:
    x = np.asarray(x)
    lab = np.asarray(lab)
    valid = np.isfinite(x) & np.isfinite(lab)
    x = x[valid]
    lab = lab[valid].astype(int)
    n = len(x)
    if n < min_win:
        return np.nan
    groups = [x[lab == g] for g in (0, 1, 2)]
    if any(len(g) < min_group for g in groups):
        return np.nan
    if np.nanmax(x) == np.nanmin(x):
        return 0.0
    try:
        H, _ = stats.kruskal(*groups, nan_policy="omit")
    except ValueError as e:
        if "All numbers are identical" in str(e):
            return 0.0
        raise
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
            counts = [(lab[m] == g).sum() for g in (0, 1, 2)]
            if any(c < min_group for c in counts):
                continue
            vals.append(kruskal_eps2(z[m], lab[m], min_win, min_group, clip_zero=clip_zero))
            ns.append(n)
        out[j] = aggregate_values(vals, ns, aggregator)
    return out


def bootstrap_intervals(B: np.ndarray, obs: np.ndarray, level: float = 0.68) -> pd.DataFrame:
    alpha = (1.0 - level) / 2.0
    qlo, qhi = np.nanquantile(B, [alpha, 1.0 - alpha], axis=0)
    mean_b = np.nanmean(B, axis=0)
    med_b = np.nanmedian(B, axis=0)
    se_b = np.nanstd(B, axis=0, ddof=1)
    zcrit = stats.norm.ppf(1.0 - alpha)

    # Percentile CI
    lo_p, hi_p = qlo, qhi
    # Basic/pivotal CI
    lo_b, hi_b = 2.0 * obs - qhi, 2.0 * obs - qlo
    # Normal/SE error band centred on observed line
    lo_n, hi_n = obs - zcrit * se_b, obs + zcrit * se_b
    # Centred bootstrap error band around observed line
    dev = B - mean_b
    dlo, dhi = np.nanquantile(dev, [alpha, 1.0 - alpha], axis=0)
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
    df = df.dropna(subset=coords + [state_col, walk_col]).copy().reset_index(drop=True)
    df["lab"] = map_state_labels(df[state_col], state_map)
    unmapped = int(df["lab"].isna().sum())
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
    return PreparedData(
        df=df,
        coords=coords,
        walk_col=walk_col,
        lab_col="lab",
        z_cols=z_cols,
        centres=centres,
        walk_indices=walk_indices,
        deltas=deltas,
    )


def permuted_labels(lab_all: np.ndarray, walk_indices: Dict[object, np.ndarray], rng: np.random.Generator, within_walk: bool) -> np.ndarray:
    if not within_walk:
        return rng.permutation(lab_all)
    out = lab_all.copy()
    for idx in walk_indices.values():
        out[idx] = rng.permutation(out[idx])
    return out


def compute_dual_nulls(
    prep: PreparedData,
    nboot: int,
    nperm: int,
    level: float,
    ci_mode: str,
    min_win: int,
    min_group: int,
    aggregator: str,
    seed: int,
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
        ci = bootstrap_intervals(B, obs, level=level)

        null_arrays = {}
        for null_name, within_walk in [("global", False), ("within", True)]:
            N = np.full((nperm, len(prep.deltas)), np.nan)
            for p in range(nperm):
                lab_perm = permuted_labels(lab_all, prep.walk_indices, rng, within_walk=within_walk)
                N[p] = curve_from_indices(z_all, lab_perm, full_idx, centres, prep.deltas, min_win, min_group, aggregator, clip_zero)
            np.save(outdir / f"_f2b_null_{null_name}_{col}.npy", N)
            null_arrays[null_name] = N

        null_stats = {}
        for null_name, N in null_arrays.items():
            null_stats[f"null_{null_name}_50"] = np.nanquantile(N, 0.50, axis=0)
            null_stats[f"null_{null_name}_95"] = np.nanquantile(N, 0.95, axis=0)
            null_stats[f"null_{null_name}_99"] = np.nanquantile(N, 0.99, axis=0)
            null_stats[f"nperm_{null_name}_valid"] = np.isfinite(N).sum(axis=0)

        for i, delta in enumerate(prep.deltas):
            row = {
                "coord": col,
                "delta": float(delta),
                "obs": float(obs[i]) if np.isfinite(obs[i]) else np.nan,
                "nboot_valid": int(np.isfinite(B[:, i]).sum()),
                "ci_mode_for_plot": ci_mode,
            }
            row.update(ci.iloc[i].to_dict())
            for key, arr in null_stats.items():
                value = arr[i]
                row[key] = int(value) if key.startswith("nperm_") else (float(value) if np.isfinite(value) else np.nan)
            row["lo"] = row[f"lo_{ci_mode}"]
            row["hi"] = row[f"hi_{ci_mode}"]
            row["lo_plot"] = max(0.0, row["lo"]) if np.isfinite(row["lo"]) else np.nan
            row["hi_plot"] = max(0.0, row["hi"]) if np.isfinite(row["hi"]) else np.nan
            row["obs_plot"] = max(0.0, row["obs"]) if np.isfinite(row["obs"]) else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def _first_crossing_delta(deltas: np.ndarray, ok: np.ndarray) -> float:
    """First grid point where ok is True."""
    deltas = np.asarray(deltas, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    if ok.size == 0 or not np.any(ok):
        return np.nan
    return float(deltas[np.argmax(ok)])


def _stable_crossing_delta(deltas: np.ndarray, ok: np.ndarray) -> float:
    """First grid point after the LAST failed point.

    This is the floor requested for the thesis figure: the smallest Delta such that
    the curve is above the null at Delta and remains above the null for all larger
    sampled Deltas. It avoids reporting an early noisy crossing if the curve later
    falls back into the null band.

    Example:
        ok = [False, True, False, True, True]
        first crossing  = second point
        stable crossing = fourth point
    """
    deltas = np.asarray(deltas, dtype=float)
    ok = np.asarray(ok, dtype=bool)
    if ok.size == 0 or not np.any(ok):
        return np.nan
    suffix_all_true = np.zeros_like(ok, dtype=bool)
    running = True
    for i in range(len(ok) - 1, -1, -1):
        running = bool(ok[i]) and running
        suffix_all_true[i] = running
    if not np.any(suffix_all_true):
        return np.nan
    return float(deltas[np.argmax(suffix_all_true)])


def compute_floor_table(results: pd.DataFrame, require_ci: bool, pooled_null: bool) -> pd.DataFrame:
    """Return coordinate-specific floors for global and within-walk nulls.

    If pooled_null=True, the threshold at each delta is max(null95) across coordinates.
    If pooled_null=False, each coordinate is compared to its own null95 curve.

    The script now reports TWO notions of floor:
    - first floor: first Delta where the curve crosses above the null.
    - stable floor: first Delta after which the curve stays above the null for all
      larger sampled Deltas. This is more conservative and avoids isolated crossings.

    For backwards compatibility, obs_floor_delta and robust_floor_delta are set equal
    to the STABLE floors. The first-crossing values are kept in first_* columns.
    """
    rows = []
    for null_name in ["global", "within"]:
        null_col = f"null_{null_name}_95"
        pooled = results.groupby("delta")[null_col].max().sort_index() if pooled_null else None
        for coord in results["coord"].drop_duplicates():
            s = results[results.coord == coord].sort_values("delta").copy()
            deltas = s["delta"].to_numpy(dtype=float)

            thresholds = []
            for _, r in s.iterrows():
                threshold = float(pooled.loc[r["delta"]]) if pooled_null else float(r[null_col])
                thresholds.append(threshold)
            thresholds = np.asarray(thresholds, dtype=float)

            obs_vals = s["obs_plot"].to_numpy(dtype=float)
            lo_vals = s["lo_plot"].to_numpy(dtype=float)

            obs_ok = np.isfinite(obs_vals) & np.isfinite(thresholds) & (obs_vals > thresholds)
            robust_ok = np.isfinite(lo_vals) & np.isfinite(thresholds) & (lo_vals > thresholds)

            first_obs_floor = _first_crossing_delta(deltas, obs_ok)
            first_robust_floor = _first_crossing_delta(deltas, robust_ok)
            stable_obs_floor = _stable_crossing_delta(deltas, obs_ok)
            stable_robust_floor = _stable_crossing_delta(deltas, robust_ok)

            max_delta = float(np.nanmax(deltas))
            def fmt(x: float) -> str:
                return f"> {max_delta:.2f} SD" if not np.isfinite(x) else f"{x:.2f} SD"

            rows.append({
                "coord": coord,
                "null": null_name,
                "threshold_type": "pooled_max_across_coords" if pooled_null else "coordinate_specific",

                # First-crossing floors: useful for audit/comparison with the old rule.
                "first_obs_floor_delta": first_obs_floor,
                "first_robust_floor_delta": first_robust_floor,
                "first_obs_floor_text": fmt(first_obs_floor),
                "first_robust_floor_text": fmt(first_robust_floor),

                # Stable floors: requested rule; these are the recommended values.
                "stable_obs_floor_delta": stable_obs_floor,
                "stable_robust_floor_delta": stable_robust_floor,
                "stable_obs_floor_text": fmt(stable_obs_floor),
                "stable_robust_floor_text": fmt(stable_robust_floor),

                # Backwards-compatible names used by plotting and F3.
                "obs_floor_delta": stable_obs_floor,
                "robust_floor_delta": stable_robust_floor,
                "obs_floor_text": fmt(stable_obs_floor),
                "robust_floor_text": fmt(stable_robust_floor),
            })
    return pd.DataFrame(rows)

def plot_f2b(
    results: pd.DataFrame,
    outdir: Path,
    null_name: str,
    pooled_null: bool,
    show_floor_lines: bool,
    y_max: float | None = None,
) -> None:
    null_col = f"null_{null_name}_95"
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    # Curves and bands
    for col in DEFAULT_COORDS:
        if col not in set(results["coord"]):
            continue
        s = results[results.coord == col].sort_values("delta")
        color = color_for(col)
        ax.fill_between(s.delta.to_numpy(), s.lo_plot.to_numpy(), s.hi_plot.to_numpy(), color=color, alpha=0.18, zorder=1)
        ax.plot(s.delta.to_numpy(), s.obs_plot.to_numpy(), "-o", color=color, ms=4, lw=1.8, label=disp_for(col), zorder=3)

    # Null line(s)
    if pooled_null:
        pooled = results.groupby("delta")[null_col].max().sort_index()
        ax.plot(pooled.index.to_numpy(), pooled.values, "--", color="0.55", lw=1.4,
                label=f"pooled {null_name} perm. null (95%)", zorder=2)
    else:
        for col in DEFAULT_COORDS:
            if col not in set(results["coord"]):
                continue
            s = results[results.coord == col].sort_values("delta")
            ax.plot(s.delta.to_numpy(), s[null_col].to_numpy(), "--", color=color_for(col), alpha=0.55, lw=1.0,
                    label=f"{disp_for(col)} {null_name} null 95%", zorder=2)

    # Coordinate-specific robust floors as vertical dotted lines
    if show_floor_lines:
        floor_table = compute_floor_table(results, require_ci=True, pooled_null=pooled_null)
        ft = floor_table[floor_table["null"] == null_name]
        for _, r in ft.iterrows():
            f = r["robust_floor_delta"]
            coord = r["coord"]
            if np.isfinite(f):
                ax.axvline(float(f), color=color_for(coord), ls=":", lw=1.0, alpha=0.75)

    ax.set_xlabel(r"window width  $\Delta$  (SD)")
    ax.set_ylabel(r"separability  $\varepsilon^2$")
    ax.set_xlim(0, max(results["delta"]) + 0.05)
    if y_max is None:
        null_max = results[null_col].max()
        hi_max = results["hi_plot"].max()
        obs_max = results["obs_plot"].max()
        y_max = np.nanmax([null_max, hi_max, obs_max]) * 1.10
        y_max = max(0.02, y_max)
    ax.set_ylim(-0.001, y_max)
    mode = str(results["ci_mode_for_plot"].dropna().iloc[0]) if "ci_mode_for_plot" in results else "bootstrap"
    pooled_txt = "pooled benchmark" if pooled_null else "coordinate-specific nulls"
    title_null = "global/pooled" if null_name == "global" else "within-walk"
    
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    suffix = "pooled" if pooled_null else "coordspecific"
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"F2b_scale_std_{null_name}_null_{suffix}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f3_contrast(prep: PreparedData, floors: pd.DataFrame, outdir: Path, null_name: str, floor_kind: str = "robust") -> pd.DataFrame:
    """Plot within-walk ranges with coordinate-specific floor markers.

    This is optional but useful: each coordinate gets its own horizontal floor line if found.
    """
    df = prep.df
    rows = []
    for col in prep.coords:
        sd = df[col].std(ddof=1)
        raw_range = df.groupby(prep.walk_col)[col].max() - df.groupby(prep.walk_col)[col].min()
        rows.append(pd.DataFrame({"walk": raw_range.index, "coord": col, "range_raw": raw_range.values, "range_sd": (raw_range / sd).values}))
    contrast = pd.concat(rows, ignore_index=True)
    contrast.to_csv(outdir / "F3_withinwalk_contrast_dual_nulls.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    rng = np.random.default_rng(0)
    for i, col in enumerate(prep.coords):
        s = contrast[contrast.coord == col]
        vals = s["range_sd"].dropna().to_numpy()
        color = color_for(col)
        bp = ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True, showfliers=False, zorder=2)
        for b in bp["boxes"]:
            b.set(facecolor=color, alpha=0.25, edgecolor=color, lw=1.2)
        for k in ("whiskers", "caps", "medians"):
            for ln in bp[k]:
                ln.set(color=color, lw=1.2)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(vals)), vals, s=18, color=color, alpha=0.55, edgecolor="none", zorder=3)
        ax.scatter([i], [np.mean(vals)], marker="D", s=38, color=color, edgecolor="white", lw=0.5, zorder=4)
        raw = s["range_raw"].dropna().to_numpy()
      

        fr = floors[(floors["coord"] == col) & (floors["null"] == null_name)]
        if not fr.empty:
            fcol = "robust_floor_delta" if floor_kind == "robust" else "obs_floor_delta"
            f = fr.iloc[0][fcol]
            if np.isfinite(f):
                ax.hlines(float(f), i - 0.33, i + 0.33, color=color, ls="--", lw=1.4, zorder=1)

    ax.set_xticks(range(len(prep.coords)))
    ax.set_xticklabels([disp_for(c) for c in prep.coords], fontsize=14)
    ax.set_ylabel("within-walk range (SD)")

    upper = np.nanpercentile(contrast["range_sd"], 99) * 1.18
    finite_floors = floors.loc[floors["null"] == null_name, "robust_floor_delta"].dropna()
    if len(finite_floors):
        upper = max(upper, finite_floors.max() * 1.25)
    ax.set_ylim(-0.25, upper)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(outdir / f"F3_withinwalk_contrast_{null_name}_floors.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return contrast


def write_audit_report(results: pd.DataFrame, floors_pooled: pd.DataFrame, floors_coord: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> None:
    lines = []
    lines.append("F2b dual-null separability-scale audit\n")
    lines.append(f"CI/error band for plotted bootstrap uncertainty: {args.ci_mode}; level={args.level}")
    lines.append(f"Aggregator over windows: {args.aggregator}")
    lines.append(f"Bootstrap: {args.nboot} walk-cluster resamples")
    lines.append(f"Permutation nulls: {args.nperm} global + {args.nperm} within-walk")
    lines.append(f"Epsilon^2 clipped at zero: {not args.no_clip_zero}")
    lines.append("")
    lines.append("Coordinate-specific floors against POOLED/MAX null benchmark:")
    lines.append(floors_pooled.to_string(index=False))
    lines.append("")
    lines.append("Coordinate-specific floors against each coordinate's own null:")
    lines.append(floors_coord.to_string(index=False))
    lines.append("")
    for mode in ["percentile", "basic", "normal", "centered"]:
        lo, hi = f"lo_{mode}", f"hi_{mode}"
        ok = (results[lo] <= results["obs"]) & (results["obs"] <= results[hi])
        lines.append(f"Observed inside {mode} interval/band: {int(ok.sum())}/{len(ok)}")
    lines.append("")
    lines.append("Head of results:")
    lines.append(results.head(20).round(5).to_string(index=False))
    (outdir / "F2b_dual_nulls_audit_report.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="F2b separability scale with both global and within-walk nulls.")
    parser.add_argument("--data", type=Path, required=True, help="Path to votes_with_Tenv_comfort.csv or equivalent.")
    parser.add_argument("--outdir", type=Path, default=Path("f2b_dual_nulls"), help="Output directory.")
    parser.add_argument("--coords", nargs="+", default=DEFAULT_COORDS, help="Coordinate columns.")
    parser.add_argument("--state-col", default="state3", help="3-state comfort column. Can be numeric 0/1/2.")
    parser.add_argument("--label-scheme", choices=["auto", "state3", "basin3"], default="auto",
                        help="Use state3 for comfortable/neutral/uncomfortable; basin3 for comfortable-neutral/uncomfortable/very uncomfortable.")
    parser.add_argument("--walk-col", default="walk", help="Walk/cluster column.")
    parser.add_argument("--nboot", type=int, default=1000)
    parser.add_argument("--nperm", type=int, default=500)
    parser.add_argument("--level", type=float, default=0.68, help="Bootstrap interval/error-band level. Use 0.95 for 95%% bands.")
    parser.add_argument("--ci-mode", choices=["percentile", "basic", "normal", "centered"], default="centered")
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
    parser.add_argument("--no-clip-zero", action="store_true")
    parser.add_argument("--plot-coordinate-specific-nulls", action="store_true", help="Also output figures with one null line per coordinate.")
    parser.add_argument("--no-floor-lines", action="store_true", help="Do not draw vertical robust floor lines in F2b plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    args.outdir.mkdir(parents=True, exist_ok=True)
    deltas = np.round(np.arange(args.dmin, args.dmax + 1e-12, args.dstep), 10)

    scheme = args.label_scheme
    if scheme == "auto":
        scheme = "basin3" if "option" in args.state_col.lower() or "comfort3" in args.state_col.lower() else "state3"
    state_map = STATE3_MAP if scheme == "state3" else BASIN3_MAP

    prep = prepare_data(
        data_path=args.data,
        coords=args.coords,
        state_col=args.state_col,
        walk_col=args.walk_col,
        deltas=deltas,
        n_centres=args.n_centres,
        center_q_low=args.center_q_low,
        center_q_high=args.center_q_high,
        state_map=state_map,
    )
    print(f"Loaded {len(prep.df)} rows, {prep.df[args.walk_col].nunique()} walks")
    print("State counts:", prep.df["lab"].value_counts().sort_index().to_dict())

    results = compute_dual_nulls(
        prep=prep,
        nboot=args.nboot,
        nperm=args.nperm,
        level=args.level,
        ci_mode=args.ci_mode,
        min_win=args.min_win,
        min_group=args.min_group,
        aggregator=args.aggregator,
        seed=args.seed,
        clip_zero=not args.no_clip_zero,
        outdir=args.outdir,
    )
    results.to_csv(args.outdir / "_f2b_data_dual_nulls.csv", index=False)

    floors_pooled = compute_floor_table(results, require_ci=True, pooled_null=True)
    floors_coord = compute_floor_table(results, require_ci=True, pooled_null=False)
    floors_pooled.to_csv(args.outdir / "F2b_discrimination_floors_pooled_null.csv", index=False)
    floors_coord.to_csv(args.outdir / "F2b_discrimination_floors_coordinate_specific_null.csv", index=False)

    # Main requested figures: same pooled benchmark, once for global null and once for within-walk null.
    plot_f2b(results, args.outdir, null_name="global", pooled_null=True, show_floor_lines=not args.no_floor_lines)
    plot_f2b(results, args.outdir, null_name="within", pooled_null=True, show_floor_lines=not args.no_floor_lines)

    # Optional diagnostic figures: null line per coordinate.
    if args.plot_coordinate_specific_nulls:
        plot_f2b(results, args.outdir, null_name="global", pooled_null=False, show_floor_lines=not args.no_floor_lines)
        plot_f2b(results, args.outdir, null_name="within", pooled_null=False, show_floor_lines=not args.no_floor_lines)

    plot_f3_contrast(prep, floors_pooled, args.outdir, null_name="global")
    plot_f3_contrast(prep, floors_pooled, args.outdir, null_name="within")

    config = vars(args).copy()
    config["data"] = str(config["data"])
    config["outdir"] = str(config["outdir"])
    config["deltas"] = deltas.tolist()
    (args.outdir / "F2b_dual_nulls_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_audit_report(results, floors_pooled, floors_coord, args.outdir, args)

    print("\nWrote:")
    print(" - _f2b_data_dual_nulls.csv")
    print(" - F2b_scale_std_global_null_pooled.png/pdf")
    print(" - F2b_scale_std_within_null_pooled.png/pdf")
    print(" - F2b_discrimination_floors_pooled_null.csv")
    print(" - F2b_discrimination_floors_coordinate_specific_null.csv")
    print(" - F2b_dual_nulls_audit_report.txt")
    print("\nPooled/max null floors:")
    print(floors_pooled.to_string(index=False))


if __name__ == "__main__":
    main()
