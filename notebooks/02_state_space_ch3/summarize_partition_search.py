from __future__ import annotations

from pathlib import Path
import pandas as pd

from ORGANISED.thermal_reorg.create_groups_comf import ML_SELECTED_GROUPS

BASE = Path(__file__).resolve().parent
CSVS = BASE / "csvs"
SEARCH_FILES = [
    "comfort_partition_search.csv",
    "comfort_partition_search_NOWalk.csv",
    "comfort_partition_search_WITHWalk.csv",
]

TOKEN_MAP = {
    "VC": "Very comfortable",
    "C": "Comfortable",
    "SC": "Slightly comfortable",
    "N": "Neutral",
    "SU": "Slightly uncomfortable",
    "U": "Uncomfortable",
    "VU": "Very uncomfortable",
}


def expand_group_definition(short_def: str) -> str:
    groups = [g.strip() for g in short_def.split("||")]
    expanded = []
    for g in groups:
        items = [TOKEN_MAP[t.strip()] for t in g.split("+")]
        expanded.append(" + ".join(items))
    return " || ".join(expanded)


def add_group_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    partition_to_outcome = {}
    for outcome_col, meta in ML_SELECTED_GROUPS.items():
        partition_to_outcome[expand_group_definition(meta["group_definition"])] = outcome_col

    out["outcome_col"] = out["partition"].map(partition_to_outcome)
    out["pretty_label"] = out["outcome_col"].map(lambda x: ML_SELECTED_GROUPS[x]["pretty_label"] if pd.notna(x) else pd.NA)
    return out


def summarize_best_features(df: pd.DataFrame, metric: str = "bal_acc_penalized") -> pd.DataFrame:
    tmp = add_group_metadata(df)
    tmp = tmp.dropna(subset=["outcome_col"]).copy()
    if tmp.empty:
        return pd.DataFrame()

    idx = tmp.groupby(["outcome_col", "partition"])[metric].idxmax()
    best = tmp.loc[idx].copy()
    best = best.sort_values(metric, ascending=False)
    best["ranking_metric"] = metric
    return best[
        [
            "pretty_label",
            "outcome_col",
            "partition",
            "feature_set",
            "model",
            "n_groups",
            "largest_class_share",
            "smallest_class_share",
            "bal_acc_mean",
            "bal_acc_adj_chance",
            "bal_acc_penalized",
            "f1_macro_mean",
            "ranking_metric",
        ]
    ]


def main():
    for fname in SEARCH_FILES:
        p = CSVS / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        best = summarize_best_features(df, metric="bal_acc_penalized")
        out_path = p.with_name(p.stem + "_best_features.csv")
        best.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {out_path} ({len(best)} rows)")


if __name__ == "__main__":
    main()
