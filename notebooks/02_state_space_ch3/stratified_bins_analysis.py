
from __future__ import annotations
import numpy as np
import pandas as pd

from candidate_partition_workflow import run_candidate_workflow

def add_temperature_regime(df: pd.DataFrame, temp_col: str = "<T-T_fixed+<T>>", q: int = 3) -> pd.DataFrame:
    out = df.copy()
    out["temp_regime"] = pd.qcut(out[temp_col], q=q, labels=[f"Tbin_{i+1}" for i in range(q)], duplicates="drop")
    return out

def run_temperature_stratified_analysis(df: pd.DataFrame, temp_col: str = "<T-T_fixed+<T>>", q: int = 3,
                                        include_xgb: bool = False, save_prefix="temp_regime"):
    out = add_temperature_regime(df, temp_col=temp_col, q=q)
    parts = []
    for regime, sub in out.groupby("temp_regime", dropna=True):
        if len(sub) < 100:
            continue
        res = run_candidate_workflow(sub, include_xgb=include_xgb, save_prefix=f"{save_prefix}_{regime}")
        res["temp_regime"] = regime
        parts.append(res)
    final = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not final.empty:
        final.to_csv(f"{save_prefix}_all_results.csv", index=False, encoding="utf-8-sig")
    return final
