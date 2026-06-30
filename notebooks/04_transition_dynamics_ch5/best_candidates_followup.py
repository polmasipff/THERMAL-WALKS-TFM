from __future__ import annotations

from pathlib import Path
import pandas as pd

from analysis_integration import prepare_df
from candidate_partition_workflow import plot_confusion_for_candidate
from opinion_phase_markov import run_phase_markov_analysis

BASE = Path(__file__).resolve().parent
THERMAL_REORG = BASE.parent / "thermal_reorg"
CSVS = THERMAL_REORG / "csvs"

RANK_TABLE = CSVS / "candidate_groupings_rank_table.csv"
BEST_FEATURES = CSVS / "comfort_partition_search_best_features.csv"

TOP_LABEL_TO_OUTCOME = {
    "Comfort 3 – option 1": "comfort3_option1",
    "Comfort 3 – option 2": "comfort3_option2",
    "Comfort 4 – option 1": "comfort4_option1",
    "Comfort 3": "comfort3_UNoption",
    "Comfort 4 soft": "comfort4_soft",
}


def load_top_candidates(top_n: int = 3) -> pd.DataFrame:
    rank_df = pd.read_csv(RANK_TABLE).sort_values("global_score", ascending=False)
    top = rank_df.head(top_n).copy()
    top["outcome_col"] = top["pretty_label"].map(TOP_LABEL_TO_OUTCOME)

    if BEST_FEATURES.exists():
        feat = pd.read_csv(BEST_FEATURES)
        feat = feat.sort_values("bal_acc_penalized", ascending=False)
        feat = feat.drop_duplicates(subset=["pretty_label"], keep="first")
        top = top.merge(
            feat[["pretty_label", "feature_set", "model", "bal_acc_penalized"]],
            on="pretty_label",
            how="left",
        )
    else:
        top["feature_set"] = "full_context"
        top["model"] = "hgb"
        top["bal_acc_penalized"] = pd.NA

    top["feature_set"] = top["feature_set"].fillna("full_context")
    top["model"] = top["model"].fillna("hgb")
    return top


def run_followup(df_raw: pd.DataFrame, top_n: int = 3, save_prefix: str = "best_candidate"):
    df = prepare_df(df_raw)
    top = load_top_candidates(top_n=top_n)

    cm_rows = []
    for _, r in top.iterrows():
        outcome_col = r["outcome_col"]
        if pd.isna(outcome_col) or outcome_col not in df.columns:
            continue

        feature_set = str(r.get("feature_set", "full_context"))
        model = str(r.get("model", "hgb"))

        cm_path = f"{save_prefix}_{outcome_col}_{model}_{feature_set}_cm.png"
        cm = plot_confusion_for_candidate(
            df,
            outcome_col=outcome_col,
            feature_set_name=feature_set,
            model_name=model,
            include_xgb=False,
            output_path=cm_path,
        )
        cm_df = cm.reset_index().rename(columns={"index": "true_class"})
        cm_long = cm_df.melt(id_vars=["true_class"], var_name="pred_class", value_name="value")
        cm_long["outcome_col"] = outcome_col
        cm_long["pretty_label"] = r["pretty_label"]
        cm_long["model"] = model
        cm_long["feature_set"] = feature_set
        cm_rows.append(cm_long)

        # Start testing Markov/phase scripts with each selected grouped outcome
        run_phase_markov_analysis(
            df,
            save_prefix=f"{save_prefix}_{outcome_col}_phase",
            state_col=outcome_col,
        )

    cm_out = pd.concat(cm_rows, ignore_index=True) if cm_rows else pd.DataFrame()
    if not cm_out.empty:
        cm_out.to_csv(f"{save_prefix}_top_confusion_matrices.csv", index=False, encoding="utf-8-sig")

    top.to_csv(f"{save_prefix}_top_candidates_used.csv", index=False, encoding="utf-8-sig")
    return {"top_candidates": top, "confusion_long": cm_out}


if __name__ == "__main__":
    # Example usage:
    # raw = pd.read_csv("../data/all_surveys(votes).csv")
    # run_followup(raw, top_n=3)
    print("Load your raw merged dataframe and call run_followup(df_raw, top_n=3)")
