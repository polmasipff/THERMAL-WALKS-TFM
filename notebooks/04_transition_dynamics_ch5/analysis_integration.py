
from __future__ import annotations
import numpy as np
import pandas as pd

COMFORT7_ORDER = [
    "Very comfortable",
    "Comfortable",
    "Slightly comfortable",
    "Neutral",
    "Slightly uncomfortable",
    "Uncomfortable",
    "Very uncomfortable",
]

SENS7_ORDER = [
    "Cool",
    "Slightly cool",
    "Neutral",
    "Slightly warm",
    "Warm",
    "Hot",
    "Very hot",
]


def add_walk_stop_columns(df, space_col="space_code"):
    out = df.copy()

    out["walk_id"] = out[space_col].astype(str).str.extract(r"^(\d+[A-Za-z]+)")
    out["stop_idx"] = (
        out[space_col].astype(str).str.extract(r"(\d+)$")[0].astype(float)
    )

    #Let's also add a row ID to keep track of calculated ML results
    out["row_id"] = np.arange(len(out))

    return out





def add_participant_weights(df: pd.DataFrame, id_col: str = "ID") -> pd.DataFrame:
    out = df.copy()
    vc = out[id_col].value_counts()
    out["sample_weight_id"] = out[id_col].map(lambda x: 1.0 / vc.get(x, 1))
    return out

def add_binary_comf(df:pd.DataFrame):
    out = df.copy()
    out["is_uncomfortable_or_worse"] = out["thermal_comfort"].isin(["Uncomfortable", "Very uncomfortable"])
    out["is_very_uncomfortable"] = out["thermal_comfort"] == "Very uncomfortable"
    out["is_slightly_uncomfortable_or_worse"] = out["thermal_comfort"].isin(["Slightly uncomfortable", "Uncomfortable", "Very uncomfortable"])
    return out

def add_binary_sens(df:pd.DataFrame):
    out = df.copy()
    out["is_warm_or_hot"] = out["thermal_sensation"].isin(["Warm", "Hot", "Very hot"])
    out["is_hot_or_very_hot"] = out["thermal_sensation"].isin(["Hot", "Very hot"])
    out["is_very_hot"] = out["thermal_sensation"].isin([ "Very hot"])
    return out

def add_relatives(df_core:pd.DataFrame):
    temp_col = "<T-T_fixed+<T>>"

    df_core[temp_col] = pd.to_numeric(df_core[temp_col], errors="coerce")

    # global mean reference
    T_mean_global = df_core[temp_col].mean(skipna=True)
    df_core["temp_rel_global"] = df_core[temp_col] - T_mean_global

    # walk mean reference
    df_core["temp_mean_walk"] = df_core.groupby("walk_id")[temp_col].transform("mean")
    df_core["temp_rel_walk"] = df_core[temp_col] - df_core["temp_mean_walk"]

    temp_col = "<HDX-HDX_fixed+<HDX>>"
    # global mean reference
    T_mean_global = df_core[temp_col].mean(skipna=True)
    df_core["HDX_rel_global"] = df_core[temp_col] - T_mean_global

    # walk mean reference
    df_core["HDX_mean_walk"] = df_core.groupby("walk_id")[temp_col].transform("mean")
    df_core["HDX_rel_walk"] = df_core[temp_col] - df_core["HDX_mean_walk"]

    return df_core

    

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_walk_stop_columns(out, "space_code")
    out = add_binary_sens(out)
    out = add_binary_comf(out)
    out = add_relatives(out)
    out = add_participant_weights(out)
    return out
