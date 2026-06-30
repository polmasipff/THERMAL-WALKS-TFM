import pandas as pd 
import numpy as np

ML_SELECTED_GROUPS = {
    "comfort3_option1": {
        "pretty_label": "Comfort 3 – option 1",
        "group_definition": "VC+C+SC+N || SU+U || VU",
        "description": "Model-friendly 3 groups with a broad comfortable side and explicit very-uncomfortable tail.",
    },
    "comfort3_option2": {
        "pretty_label": "Comfort 3 – option 2",
        "group_definition": "VC+C+SC || N+SU+U || VU",
        "description": "3 groups that isolate very-uncomfortable and keep a central neutral-to-uncomfortable middle block.",
    },
    "comfort4_option1": {
        "pretty_label": "Comfort 4 – option 1",
        "group_definition": "VC || C+SC+N || SU+U || VU",
        "description": "4 groups isolating very comfortable and very uncomfortable, with two central bands.",
    },
    "comfort3": {
        "pretty_label": "Comfort 3",
        "group_definition": "VC+C+SC || N || SU+U+VU",
        "description": "Classic triad split between comfortable, neutral, and uncomfortable side.",
    },
    "comfort4_soft": {
        "pretty_label": "Comfort 4 soft",
        "group_definition": "VC+C || SC+N || SU || U+VU",
        "description": "Soft 4-group gradient emphasizing near-neutral and slight discomfort transitions.",
    },
}


def recode_comfort_3(x):
    if pd.isna(x): return np.nan
    if x in ["Very comfortable", "Comfortable", "Slightly comfortable"]: return "comfortable"
    if x == "Neutral": return "neutral"
    if x in ["Slightly uncomfortable", "Uncomfortable", "Very uncomfortable"]: return "uncomfortable"
    return np.nan

def recode_comfort_3_option1(x):
    if pd.isna(x): return np.nan
    if x in ["Very comfortable", "Comfortable", "Slightly comfortable","Neutral"]: return "comfortable-neutral"
    if x in ["Slightly uncomfortable", "Uncomfortable"]: return "uncomfortable"
    if x in ["Very uncomfortable"]: return "very uncomfortable"
    return np.nan

def recode_comfort_3_UNcomfortableoption(x):
    if pd.isna(x): return np.nan
    if x in ["Very comfortable", "Comfortable", "Slightly comfortable","Neutral","Slightly uncomfortable"]: return "comfortable"
    if x in [ "Uncomfortable"]: return "uncomfortable"
    if x in ["Very uncomfortable"]: return "very uncomfortable"
    return np.nan

def recode_comfort_3_option2(x):
    if pd.isna(x): return np.nan
    if x in ["Very comfortable", "Comfortable", "Slightly comfortable"]: return "comfortable"
    if x in [ "Neutral", "Slighlty uncomfortable","Uncomfortable"]: return "neutral-uncomfortable"
    if x in ["Very uncomfortable"]: return "very uncomfortable"
    return np.nan


def recode_comfort_4_soft(x):
    if pd.isna(x): return np.nan
    if x in ["Very comfortable", "Comfortable"]: return "comfortable"
    if x in ["Slightly comfortable", "Neutral"]: return "near_neutral"
    if x == "Slightly uncomfortable": return "slightly_uncomfortable"
    if x in ["Uncomfortable", "Very uncomfortable"]: return "uncomfortable"
    return np.nan


def recode_comfort4_option1(x):
    if pd.isna(x): return np.nan
    if x in ["Very comfortable"]: return "very comfortable"
    if x in ["Comfortable", "Slightly comfortable", "Neutral"]: return "near_neutral"
    if x in ["Slightly uncomfortable","Uncomfortable"]: return "slightly_uncomfortable"
    if x in ["Very uncomfortable"]: return "uncomfortable"
    return np.nan

def recode_comfort4_option2(x):
    if pd.isna(x): return np.nan
    if x in ["Very comfortable"]: return "very comfortable"
    if x in ["Comfortable", "Slightly comfortable", "Neutral","Slightly uncomfortable"]: return "near_neutral"
    if x in ["Uncomfortable"]: return "uncomfortable"
    if x in ["Very uncomfortable"]: return "very uncomfortable"
    return np.nan

def add_outcomes_grup(df):
    out = df.copy()
    out["comfort7"] = out["thermal_comfort"]
    out["comfort3"] = out["thermal_comfort"].map(recode_comfort_3)
    out["comfort3_option1"] = out["thermal_comfort"].map(recode_comfort_3_option1)
    out["comfort3_UNoption"] = out["thermal_comfort"].map(recode_comfort_3_UNcomfortableoption)
    out["comfort4_soft"] = out["thermal_comfort"].map(recode_comfort_4_soft)
    out["comfort3_option2"] = out["thermal_comfort"].map(recode_comfort_3_option2)
    out["comfort4_option1"] = out["thermal_comfort"].map(recode_comfort4_option1)
    out["comfort4_option2"] = out["thermal_comfort"].map(recode_comfort4_option2)
    return out

def add_selected_outcomes_group(df):
    out = df.copy()
    out["comfort7"] = out["thermal_comfort"]#.map(lambda x: x if x in ["Very comfortable", "Comfortable", "Slightly comfortable", "Neutral", "Slightly uncomfortable", "Uncomfortable", "Very uncomfortable"] else np.nan)
    out["comfort3"] = out["thermal_comfort"].map(recode_comfort_3)
    out["comfort3_UNoption"] = out["thermal_comfort"].map(recode_comfort_3_UNcomfortableoption)
    out["comfort3_option1"] = out["thermal_comfort"].map(recode_comfort_3_option1)
    out["comfort4_soft"] = out["thermal_comfort"].map(recode_comfort_4_soft)
    out["comfort3_option2"] = out["thermal_comfort"].map(recode_comfort_3_option2)
    out["comfort4_option1"] = out["thermal_comfort"].map(recode_comfort4_option1)
    return out

def get_ml_selected_groups_table():
    """
    Human-readable table with the candidate groups selected in ML ranking.
    """
    rows = []
    for outcome_col, meta in ML_SELECTED_GROUPS.items():
        rows.append({
            "outcome_col": outcome_col,
            "pretty_label": meta["pretty_label"],
            "group_definition": meta["group_definition"],
            "description": meta["description"],
        })
    return pd.DataFrame(rows)


def wilson_ci(k, n, z=1.96):
    """
    Wilson confidence interval for a proportion.
    Returns (low, high). If n=0, returns (nan, nan).
    """
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def classify_bin_dominance(bin_df, outcome_col, bin_col="T_bin"):
    """
    For each bin, returns the dominant class and its proportion.
    """
    tmp = bin_df[[bin_col, outcome_col]].dropna().copy()
    counts = (
        tmp.groupby([bin_col, outcome_col], observed=True)
        .size()
        .reset_index(name="count")
    )
    counts["bin_total"] = counts.groupby(bin_col)["count"].transform("sum")
    counts["prop"] = counts["count"] / counts["bin_total"]

    dominant = (
        counts.sort_values([bin_col, "prop"], ascending=[True, False])
        .groupby(bin_col, as_index=False)
        .first()
        .rename(columns={outcome_col: "dominant_class", "prop": "dominant_prop"})
    )
    return counts, dominant


def make_bin_class_table(df, outcome_col, bin_col="T_bin"):
    """
    For each outcome class within each T_bin:
    - count
    - proportion
    - Wilson CI
    """
    tmp = df[[bin_col, outcome_col]].dropna().copy()

    counts = (
        tmp.groupby([bin_col, outcome_col], observed=True)
        .size()
        .reset_index(name="count")
    )

    counts["bin_total"] = counts.groupby(bin_col)["count"].transform("sum")
    counts["prop"] = counts["count"] / counts["bin_total"]

    ci = counts.apply(
        lambda r: wilson_ci(r["count"], r["bin_total"]),
        axis=1,
        result_type="expand"
    )
    counts["ci_low"] = ci[0]
    counts["ci_high"] = ci[1]
    counts["ci_width"] = counts["ci_high"] - counts["ci_low"]

    return counts.sort_values([bin_col, outcome_col]).reset_index(drop=True)


def compute_errorbar_overlap(bin_class_table, bin_col="T_bin", class_col=None):
    """
    Computes pairwise overlap of Wilson CIs between classes within each bin.

    overlap_length = max(0, min(high1, high2) - max(low1, low2))
    normalized_overlap = overlap_length / min(width1, width2) if possible

    Returns one row per pair of classes per bin.
    """
    if class_col is None:
        possible = [c for c in bin_class_table.columns if c not in
                    [bin_col, "count", "bin_total", "prop", "ci_low", "ci_high", "ci_width"]]
        if len(possible) != 1:
            raise ValueError("Could not infer class column.")
        class_col = possible[0]

    rows = []

    for b, g in bin_class_table.groupby(bin_col, observed=True):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a = g.loc[i]
                c = g.loc[j]

                overlap = max(0.0, min(a["ci_high"], c["ci_high"]) - max(a["ci_low"], c["ci_low"]))
                min_width = min(a["ci_width"], c["ci_width"])
                norm_overlap = overlap / min_width if (pd.notna(min_width) and min_width > 0) else np.nan

                rows.append({
                    bin_col: b,
                    "class_a": a[class_col],
                    "class_b": c[class_col],
                    "overlap_length": overlap,
                    "normalized_overlap": norm_overlap,
                    "prop_a": a["prop"],
                    "prop_b": c["prop"],
                    "abs_prop_diff": abs(a["prop"] - c["prop"]),
                })

    return pd.DataFrame(rows)


def compute_bin_imbalance(bin_class_table, bin_col="T_bin", class_col=None):
    """
    Computes imbalance metrics for each bin.

    Metrics:
    - max_prop: proportion of dominant class
    - imbalance_l1_uniform: distance to uniform distribution
    - normalized_entropy: entropy / log(K)
    - concentration_index: sum(p_i^2)
    """
    if class_col is None:
        possible = [c for c in bin_class_table.columns if c not in
                    [bin_col, "count", "bin_total", "prop", "ci_low", "ci_high", "ci_width"]]
        if len(possible) != 1:
            raise ValueError("Could not infer class column.")
        class_col = possible[0]

    out = []

    for b, g in bin_class_table.groupby(bin_col, observed=True):
        p = g["prop"].values.astype(float)
        p = p[p > 0]
        K = len(g)

        uniform = np.ones(K) / K
        p_full = g["prop"].values.astype(float)

        entropy = -np.sum(p * np.log(p)) if len(p) > 0 else np.nan
        max_entropy = np.log(K) if K > 1 else np.nan
        normalized_entropy = entropy / max_entropy if pd.notna(max_entropy) and max_entropy > 0 else np.nan

        imbalance_l1_uniform = np.sum(np.abs(p_full - uniform))
        concentration_index = np.sum(p_full ** 2)

        out.append({
            bin_col: b,
            "n_classes_present": K,
            "bin_total": g["bin_total"].iloc[0],
            "max_prop": g["prop"].max(),
            "dominant_class": g.loc[g["prop"].idxmax(), class_col],
            "imbalance_l1_uniform": imbalance_l1_uniform,
            "normalized_entropy": normalized_entropy,
            "concentration_index": concentration_index,
        })

    return pd.DataFrame(out).sort_values(bin_col).reset_index(drop=True)


def summarize_temperature_bins(df, outcome_col, bin_col="T_bin"):
    """
    Wrapper returning:
    1. class table
    2. pairwise CI overlap table
    3. imbalance table
    4. dominant-class table
    """
    class_table = make_bin_class_table(df, outcome_col=outcome_col, bin_col=bin_col)
    overlap_table = compute_errorbar_overlap(class_table, bin_col=bin_col, class_col=outcome_col)
    imbalance_table = compute_bin_imbalance(class_table, bin_col=bin_col, class_col=outcome_col)
    _, dominant_table = classify_bin_dominance(df, outcome_col=outcome_col, bin_col=bin_col)

    return class_table, overlap_table, imbalance_table, dominant_table
