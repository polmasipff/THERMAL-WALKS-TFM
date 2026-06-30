
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COMFORT7_ORDER = [
    "Very comfortable",
    "Comfortable",
    "Slightly comfortable",
    "Neutral",
    "Slightly uncomfortable",
    "Uncomfortable",
    "Very uncomfortable",
]
COMFORT_CODE = {k: i for i, k in enumerate(COMFORT7_ORDER)}

def _infer_spin_value(state) -> float:
    if pd.isna(state):
        return np.nan
    s = str(state).lower().strip()
    if "very uncomfortable" in s:
        return 1.0
    if "uncomfortable" in s:
        return 1.0
    if "neutral" in s:
        return 0.0
    if "comfortable" in s:
        return -1.0
    return np.nan

def prepare_walk_sequence(df: pd.DataFrame, state_col: str = "thermal_comfort") -> pd.DataFrame:
    out = df.copy()
    if "walk_id" not in out.columns or "stop_idx" not in out.columns:
        s = out["space_code"].astype(str).str.extract(r"(?P<walk_prefix>\d{2})(?P<stop_idx>\d)")
        out["walk_id"] = s["walk_prefix"]
        out["stop_idx"] = pd.to_numeric(s["stop_idx"], errors="coerce")
    out["comfort7_code"] = out[state_col].map(COMFORT_CODE)
    if "thermal_comfort_walking" in out.columns:
        out["comfort7_walking_code"] = out["thermal_comfort_walking"].map(COMFORT_CODE)
    return out.sort_values(["walk_id", "ID", "stop_idx"])

def add_spin_variables(df: pd.DataFrame, state_col: str = "thermal_comfort") -> pd.DataFrame:
    out = df.copy()
    out["spin3"] = out[state_col].map(_infer_spin_value)
    out["spin_extreme"] = out[state_col].map(
        lambda x: 1 if str(x).strip().lower() == "very uncomfortable" else (0 if pd.notna(x) else np.nan)
    )
    return out

def local_order_by_stop(df: pd.DataFrame, state_col: str = "thermal_comfort") -> pd.DataFrame:
    out = add_spin_variables(prepare_walk_sequence(df, state_col=state_col), state_col=state_col)
    g = out.groupby(["walk_id", "stop_idx"], dropna=True)
    res = g.agg(
        n=("ID", "size"),
        order_m=("spin3", "mean"),
        susceptibility_like=("spin3", lambda s: len(s) * np.nanvar(s, ddof=0)),
        extreme_prob=("spin_extreme", "mean"),
        temp_mean=("<T-T_fixed+<T>>", "mean"),
        hdx_mean=("<HDX-HDX_fixed+<HDX>>", "mean"),
    ).reset_index()
    # Shannon entropy of comfort categories within stop
    ent = (
        out.groupby(["walk_id", "stop_idx", state_col], dropna=True)
        .size()
        .rename("count")
        .reset_index()
    )
    ent["p"] = ent.groupby(["walk_id", "stop_idx"])["count"].transform(lambda s: s / s.sum())
    ent_sum = ent.groupby(["walk_id", "stop_idx"])["p"].apply(lambda p: -(p * np.log(p)).sum()).reset_index(name="entropy_comfort")
    res = res.merge(ent_sum, on=["walk_id", "stop_idx"], how="left")
    return res.sort_values(["walk_id", "stop_idx"])

def walk_consensus_dynamics(df: pd.DataFrame, state_col: str = "thermal_comfort") -> pd.DataFrame:
    stop_df = local_order_by_stop(df, state_col=state_col).sort_values(["walk_id", "stop_idx"]).copy()
    stop_df["delta_m"] = stop_df.groupby("walk_id")["order_m"].diff()
    stop_df["delta_temp"] = stop_df.groupby("walk_id")["temp_mean"].diff()
    stop_df["delta_hdx"] = stop_df.groupby("walk_id")["hdx_mean"].diff()
    return stop_df

def participant_transition_matrix(df: pd.DataFrame, state_col: str = "thermal_comfort") -> pd.DataFrame:
    out = prepare_walk_sequence(df, state_col=state_col).copy()
    out = out.sort_values(["walk_id", "ID", "stop_idx"])
    out["next_state"] = out.groupby(["walk_id", "ID"])[state_col].shift(-1)
    tr = out.dropna(subset=[state_col, "next_state"]).groupby([state_col, "next_state"]).size().rename("count").reset_index()
    mat = tr.pivot(index=state_col, columns="next_state", values="count").fillna(0)
    probs = mat.div(mat.sum(axis=1), axis=0)
    return probs

def bayesian_transition_summary(df: pd.DataFrame, state_col: str = "thermal_comfort",
                                conditioning_cols=None) -> pd.DataFrame:
    if conditioning_cols is None:
        conditioning_cols = []
    out = prepare_walk_sequence(df, state_col=state_col).copy()
    out = out.sort_values(["walk_id", "ID", "stop_idx"])
    out["next_state"] = out.groupby(["walk_id", "ID"])[state_col].shift(-1)
    keep = [state_col, "next_state"] + conditioning_cols
    tmp = out.dropna(subset=[state_col, "next_state"]).copy()
    grp = tmp.groupby(keep).size().rename("count").reset_index()
    denom_cols = [state_col] + conditioning_cols
    grp["p_next_given_current_and_context"] = grp.groupby(denom_cols)["count"].transform(lambda s: s / s.sum())
    return grp.sort_values(denom_cols + ["p_next_given_current_and_context"], ascending=[True]*len(denom_cols)+[False])

def plot_order_parameter(df_stop: pd.DataFrame, x_col: str = "temp_mean", y_col: str = "order_m",
                         output_path: str | None = None):
    tmp = df_stop.sort_values(x_col).copy()
    plt.figure(figsize=(8,5))
    plt.plot(tmp[x_col], tmp[y_col], marker="o")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=220)

def run_phase_markov_analysis(df: pd.DataFrame, save_prefix: str = "phase", state_col: str = "thermal_comfort"):
    stop_df = local_order_by_stop(df, state_col=state_col)
    dyn_df = walk_consensus_dynamics(df, state_col=state_col)
    trans = participant_transition_matrix(df, state_col=state_col)
    bayes = bayesian_transition_summary(
        df,
        state_col=state_col,
        conditioning_cols=["walk_id", "stop_idx"]
    )
    stop_df.to_csv(f"{save_prefix}_local_order_by_stop.csv", index=False, encoding="utf-8-sig")
    dyn_df.to_csv(f"{save_prefix}_walk_consensus_dynamics.csv", index=False, encoding="utf-8-sig")
    trans.to_csv(f"{save_prefix}_participant_transition_matrix.csv", encoding="utf-8-sig")
    bayes.to_csv(f"{save_prefix}_bayesian_transition_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "stop_df": stop_df,
        "dyn_df": dyn_df,
        "transition_matrix": trans,
        "bayesian_transition_summary": bayes,
    }
