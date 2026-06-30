import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from candidate_partition_workflow import *
from pathlib import Path
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


DEFAULT_FEATURES = {
    "temp_only": ["<T-T_fixed+<T>>"],
    "hdx_only": ["<HDX-HDX_fixed+<HDX>>"],
    "temp_hdx": ["<T-T_fixed+<T>>", "<HDX-HDX_fixed+<HDX>>"],
    "temp_sun_wind_context": [
        "<T-T_fixed+<T>>", "sun", "wind", "SVF_1.5m", "NDVI_1.5m",
        "climate_shelter", "surface_type", "space_category", "IVAC",
    ],
    "temp_sun_wind_personcontext": [
        "<T-T_fixed+<T>>", "sun", "wind", "SVF_1.5m", "NDVI_1.5m",
        "climate_shelter", "surface_type", "space_category", "IVAC",
        "distance_fountain", "distance_drinking_fountain", "distance_green_zone",
    ],
    "full_context": [
        "<T-T_fixed+<T>>", "<HDX-HDX_fixed+<HDX>>", "sun", "wind",
        "SVF_1.5m", "NDVI_1.5m", "climate_shelter", "surface_type",
        "space_category", "IVAC", "distance_fountain",
        "distance_drinking_fountain", "distance_green_zone,particiants_clothing "
    ],
    "hdx_sun_wind_context": [
        "<HDX-HDX_fixed+<HDX>>", "sun", "wind", "SVF_1.5m", "NDVI_1.5m",
        "climate_shelter", "surface_type", "space_category", "IVAC"
    ],
}

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_feature_cols(df, feature_set_name="full_context"):
    cols = [c for c in DEFAULT_FEATURES[feature_set_name] if c in df.columns]
    if len(cols) == 0:
        raise ValueError(f"No columns found for feature_set_name={feature_set_name}")
    return cols


def make_preprocessor_from_X(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe()),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    return pre, num_cols, cat_cols


def choose_threshold_from_oof(
    y_true,
    y_score,
    thresholds=None,
    objective="f1",
):
    from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    rows = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        rows.append({
            "threshold": thr,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        })

    th_df = pd.DataFrame(rows)
    if objective not in th_df.columns:
        raise ValueError(f"objective must be one of {list(th_df.columns)}")

    th_df = th_df.sort_values(
        [objective, "balanced_accuracy", "recall", "precision"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return th_df.iloc[0].to_dict(), th_df

# ---------------------------------------------------------
# 8. Ordinal model:
#    regress on 0/1/2 and round back to nearest state
# ---------------------------------------------------------
def run_grouped_ordinal_model(
    df,
    outcome_col="comfort3_option1",
    feature_set_name="full_context",
    group_col="ID",
    n_splits=5,
    model_name="rf_ord",
    label_order=None,
    output_path=None,
):
    if label_order is None:
        label_order = ["comfortable-neutral", "uncomfortable", "very uncomfortable"]

    order_map = {lab: i for i, lab in enumerate(label_order)}
    inv_map = {i: lab for lab, i in order_map.items()}
    
    cols = get_feature_cols(df, feature_set_name)
    tmp = df[[outcome_col, group_col] + cols].dropna(subset=[outcome_col, group_col]).copy()

    X = tmp[cols].copy()
    y_lab = tmp[outcome_col].copy()
    y = y_lab.map(order_map).astype(int)
    groups = tmp[group_col].copy()

    pre = make_preprocessor_from_X(X)

    if model_name == "rf_ord":
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=5,
            n_jobs=-1,
        )
    else:
        raise ValueError("Supported ordinal models: 'rf_ord'")

    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    gkf = GroupKFold(n_splits=n_splits)
    y_true_all, y_pred_all, y_pred_cont_all = [], [], []

    for tr, te in gkf.split(X, y, groups):
        fitted = clone(pipe)
        fitted.fit(X.iloc[tr], y.iloc[tr])

        pred_cont = fitted.predict(X.iloc[te])
        pred_ord = np.clip(np.rint(pred_cont), 0, len(label_order) - 1).astype(int)

        y_true_all.extend(y.iloc[te].tolist())
        y_pred_all.extend(pred_ord.tolist())
        y_pred_cont_all.extend(pred_cont.tolist())

    y_true_lab = pd.Series([inv_map[int(v)] for v in y_true_all], name="y_true")
    y_pred_lab = pd.Series([inv_map[int(v)] for v in y_pred_all], name="y_pred")

    # Confusion matrices
    cm_counts = confusion_matrix(y_true_lab, y_pred_lab, labels=label_order, normalize=None)
    cm_rowpct = confusion_matrix(y_true_lab, y_pred_lab, labels=label_order, normalize="true")

    support = y_true_lab.value_counts()

    plot_cm_counts_rowpct(
        cm_counts=cm_counts,
        cm_rowpct=cm_rowpct,
        labels=label_order,
        show_support=support,
        title=f"{outcome_col} | ordinal {model_name} | {feature_set_name}",
        output_path=output_path,
    )

    # Ordinal metrics
    yt = np.asarray(y_true_all)
    yp = np.asarray(y_pred_all)

    metrics = {
        "accuracy": np.mean(yt == yp),
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "mae_ordinal": np.mean(np.abs(yt - yp)),
        "mse_ordinal": np.mean((yt - yp) ** 2),
        "far_error_rate": np.mean(np.abs(yt - yp) >= 2),  # jumps across one full state
    }

    return {
        "cm_counts": pd.DataFrame(cm_counts, index=label_order, columns=label_order),
        "cm_rowpct": pd.DataFrame(cm_rowpct, index=label_order, columns=label_order),
        "metrics": pd.DataFrame([metrics]),
        "oof": pd.DataFrame({
            "y_true": y_true_lab,
            "y_pred": y_pred_lab,
            "y_true_ord": yt,
            "y_pred_ord": yp,
            "y_pred_cont": y_pred_cont_all,
        }),
    }


# ---------------------------------------------------------
# 9. Binary detector with threshold tuning
# ---------------------------------------------------------
def run_grouped_binary_detector(
    df,
    target_col,
    feature_set_name="full_context",
    group_col="ID",
    n_splits=5,
    model_name="rf",
    threshold="auto",
    threshold_objective="f1",   # nou
    output_path=None,
):
    cols = get_feature_cols(df, feature_set_name)
    tmp = df[[target_col, group_col] + cols].dropna(subset=[target_col, group_col]).copy()

    X = tmp[cols].copy()
    y = tmp[target_col].astype(int).copy()
    groups = tmp[group_col].copy()

    pre = make_preprocessor_from_X(X)

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    elif model_name == "logreg":
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
        )
    else:
        raise ValueError("Supported binary models: 'rf', 'logreg'")

    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    gkf = GroupKFold(n_splits=n_splits)
    y_true_all, y_score_all = [], []

    for tr, te in gkf.split(X, y, groups):
        fitted = clone(pipe)
        fitted.fit(X.iloc[tr], y.iloc[tr])

        if hasattr(fitted, "predict_proba"):
            score = fitted.predict_proba(X.iloc[te])[:, 1]
        else:
            # fallback: decision-like score unavailable
            score = fitted.predict(X.iloc[te]).astype(float)

        y_true_all.extend(y.iloc[te].tolist())
        y_score_all.extend(score.tolist())

    y_true_all = np.asarray(y_true_all).astype(int)
    y_score_all = np.asarray(y_score_all).astype(float)

    # threshold selection
    if threshold == "auto":
        best_thr, threshold_table = choose_threshold_from_oof(
            y_true_all,
            y_score_all,
            objective=threshold_objective
        )
        thr = best_thr["threshold"]
    elif isinstance(threshold, (float, int)):
        thr = float(threshold)
        _, threshold_table = choose_threshold_from_oof(
            y_true_all,
            y_score_all,
            objective=threshold_objective
        )
    else:
        raise ValueError("threshold must be 'auto' or a numeric value")

    y_pred_all = (y_score_all >= thr).astype(int)

    labels = [0, 1]
    label_order = ["negative", "positive"]
    label_names = {
        "is_very_uncomfortable": ["not very uncomfortable", "very uncomfortable"],
        "is_uncomfortable_or_worse": ["not U-or-worse", "U or very uncomfortable"],
        "is_slightly_uncomfortable_or_worse": ["not SU-or-worse", "SU/U/VU"],
    }
    label_order = label_names.get(target_col, label_order)

    cm_counts = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize=None)
    cm_rowpct = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize="true")

    support = pd.Series(y_true_all).map({0: label_order[0], 1: label_order[1]}).value_counts()

    plot_cm_counts_rowpct(
        cm_counts=cm_counts,
        cm_rowpct=cm_rowpct,
        labels=label_order,
        show_support=support,
        title=f"{target_col} | {model_name} | {feature_set_name} | thr={thr:.2f}",
        output_path=output_path,
    )

    metrics = {
        "threshold_used": thr,
        "prevalence_positive": float(np.mean(y_true_all)),
        "balanced_accuracy": balanced_accuracy_score(y_true_all, y_pred_all),
        "precision_positive": precision_score(y_true_all, y_pred_all, zero_division=0),
        "recall_positive": recall_score(y_true_all, y_pred_all, zero_division=0),
        "f1_positive": f1_score(y_true_all, y_pred_all, zero_division=0),
        "average_precision": average_precision_score(y_true_all, y_score_all),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true_all, y_score_all)
    except Exception:
        metrics["roc_auc"] = np.nan

    return {
        "cm_counts": pd.DataFrame(cm_counts, index=label_order, columns=label_order),
        "cm_rowpct": pd.DataFrame(cm_rowpct, index=label_order, columns=label_order),
        "metrics": pd.DataFrame([metrics]),
        "threshold_table": threshold_table,
        "oof": pd.DataFrame({
            "y_true": y_true_all,
            "y_score": y_score_all,
            "y_pred": y_pred_all,
        }),
    }

# ---------------------------------------------------------
# 4. Plot confusion matrix:
#    colors = row percentages
#    text = counts + row percentages
# ---------------------------------------------------------
def plot_cm_counts_rowpct(
    cm_counts,
    cm_rowpct,
    labels,
    title="",
    output_path=None,
    show_support=None,
    cmap="Blues",
):
    labels = list(labels)

    if show_support is not None:
        display_labels = [f"{lab}\n(n={show_support.get(lab, 0)})" for lab in labels]
    else:
        display_labels = labels

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_rowpct, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = int(cm_counts[i, j])
            pct = cm_rowpct[i, j]
            txt = f"{count}\n({pct:.2f})"
            text_color = "white" if pct > 0.45 else "#123b7a"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------
# 5. Ordinal target creation: comfort3_option1
#    comfortable-neutral < uncomfortable < very uncomfortable
# ---------------------------------------------------------
def build_ordinal_comfort3_option1(df, source_col="thermal_comfort"):
    out = df.copy()

    def recode(x):
        if pd.isna(x):
            return np.nan
        if x in ["Very comfortable", "Comfortable", "Slightly comfortable", "Neutral"]:
            return "comfortable-neutral"
        if x in ["Slightly uncomfortable", "Uncomfortable"]:
            return "uncomfortable"
        if x == "Very uncomfortable":
            return "very uncomfortable"
        return np.nan

    out["comfort3_option1"] = out[source_col].map(recode)
   
    return out



def add_binary_discomfort_targets(df, source_col="thermal_comfort"):
    out = df.copy()

    out["is_very_uncomfortable"] = (out[source_col] == "Very uncomfortable").astype("float")
    out["is_uncomfortable_or_worse"] = (
        out[source_col].isin(["Uncomfortable", "Very uncomfortable"])
    ).astype("float")
    out["is_slightly_uncomfortable_or_worse"] = (
        out[source_col].isin(["Slightly uncomfortable", "Uncomfortable", "Very uncomfortable"])
    ).astype("float")

    na_mask = out[source_col].isna()
    out.loc[na_mask, "is_very_uncomfortable"] = np.nan
    out.loc[na_mask, "is_uncomfortable_or_worse"] = np.nan
    out.loc[na_mask, "is_slightly_uncomfortable_or_worse"] = np.nan

    return out


# ---------------------------------------------------------
# Main interpretation function for binary detector
# ---------------------------------------------------------
def analyze_binary_detector(
    df,
    target_col,
    feature_set_name="full_context",
    group_col="ID",
    n_splits=5,
    model_name="rf",
    threshold="auto",
    threshold_objective="f1",
    top_n_cases=30,
):
    cols = get_feature_cols(df, feature_set_name)
    tmp = df[[target_col, group_col] + cols].dropna(subset=[target_col, group_col]).copy()

    X = tmp[cols].copy()
    y = tmp[target_col].astype(int).copy()
    groups = tmp[group_col].copy()

    pre, num_cols, cat_cols = make_preprocessor_from_X(X)

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    elif model_name == "logreg":
        model = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
        )
    else:
        raise ValueError("Supported models: 'rf', 'logreg'")

    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    gkf = GroupKFold(n_splits=n_splits)

    y_true_all = []
    y_score_all = []
    y_pred_default_all = []
    case_rows = []

    fold_importances = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        fitted = clone(pipe)
        fitted.fit(X_tr, y_tr)

        if hasattr(fitted, "predict_proba"):
            score = fitted.predict_proba(X_te)[:, 1]
        else:
            score = fitted.predict(X_te).astype(float)

        pred_default = (score >= 0.5).astype(int)

        y_true_all.extend(y_te.tolist())
        y_score_all.extend(score.tolist())
        y_pred_default_all.extend(pred_default.tolist())

        # case table
        fold_cases = tmp.iloc[te].copy()
        fold_cases["y_true"] = y_te.values
        fold_cases["y_score"] = score
        fold_cases["y_pred_0p5"] = pred_default
        fold_cases["fold"] = fold
        case_rows.append(fold_cases)

        # permutation importance on validation fold
        try:
            pi = permutation_importance(
                fitted,
                X_te,
                y_te,
                n_repeats=10,
                random_state=42,
                scoring="f1",
            )
            imp_df = pd.DataFrame({
                "feature": X.columns,
                "importance_mean": pi.importances_mean,
                "importance_std": pi.importances_std,
                "fold": fold,
            })
            fold_importances.append(imp_df)
        except Exception:
            pass

    y_true_all = np.asarray(y_true_all).astype(int)
    y_score_all = np.asarray(y_score_all).astype(float)

    # choose threshold
    if threshold == "auto":
        best_thr, threshold_table = choose_threshold_from_oof(
            y_true_all, y_score_all, objective=threshold_objective
        )
        thr = float(best_thr["threshold"])
    else:
        thr = float(threshold)
        _, threshold_table = choose_threshold_from_oof(
            y_true_all, y_score_all, objective=threshold_objective
        )

    y_pred_all = (y_score_all >= thr).astype(int)

    cases = pd.concat(case_rows, ignore_index=True)
    cases["y_pred"] = y_pred_all

    # classify cases
    def tag_case(row):
        if row["y_true"] == 1 and row["y_pred"] == 1:
            return "TP"
        if row["y_true"] == 0 and row["y_pred"] == 0:
            return "TN"
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP"
        if row["y_true"] == 1 and row["y_pred"] == 0:
            return "FN"
        return "other"

    cases["case_type"] = cases.apply(tag_case, axis=1)

    # aggregate permutation importance
    if len(fold_importances) > 0:
        importance_df = pd.concat(fold_importances, ignore_index=True)
        importance_summary = (
            importance_df.groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance_mean", "mean"),
                importance_std=("importance_mean", "std"),
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
    else:
        importance_summary = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    # feature summaries by case type
    numeric_features = [c for c in cols if pd.api.types.is_numeric_dtype(tmp[c])]
    categorical_features = [c for c in cols if c not in numeric_features]

    # numeric summary
    num_rows = []
    for case_type, g in cases.groupby("case_type"):
        for col in numeric_features:
            s = g[col].dropna()
            if len(s) == 0:
                continue
            num_rows.append({
                "case_type": case_type,
                "feature": col,
                "n": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "median": s.median(),
                "q25": s.quantile(0.25),
                "q75": s.quantile(0.75),
            })
    numeric_case_summary = pd.DataFrame(num_rows)

    # categorical summary
    cat_rows = []
    for case_type, g in cases.groupby("case_type"):
        for col in categorical_features:
            vc = g[col].value_counts(dropna=False, normalize=True).reset_index()
            vc.columns = ["value", "prop"]
            vc["case_type"] = case_type
            vc["feature"] = col
            cat_rows.append(vc)
    categorical_case_summary = (
        pd.concat(cat_rows, ignore_index=True) if len(cat_rows) > 0
        else pd.DataFrame(columns=["value", "prop", "case_type", "feature"])
    )

    # top predicted positives
    top_predicted_positive = (
        cases.sort_values("y_score", ascending=False)
        .head(top_n_cases)
        .copy()
    )

    # top true positives
    top_tp = (
        cases[cases["case_type"] == "TP"]
        .sort_values("y_score", ascending=False)
        .head(top_n_cases)
        .copy()
    )

    # false positives and false negatives
    false_positives = (
        cases[cases["case_type"] == "FP"]
        .sort_values("y_score", ascending=False)
        .head(top_n_cases)
        .copy()
    )

    false_negatives = (
        cases[cases["case_type"] == "FN"]
        .sort_values("y_score", ascending=True)
        .head(top_n_cases)
        .copy()
    )

    # overall metrics
    from sklearn.metrics import (
        balanced_accuracy_score, precision_score, recall_score,
        f1_score, average_precision_score, roc_auc_score
    )

    metrics = pd.DataFrame([{
        "target_col": target_col,
        "threshold_used": thr,
        "prevalence_positive": float(np.mean(y_true_all)),
        "balanced_accuracy": balanced_accuracy_score(y_true_all, y_pred_all),
        "precision_positive": precision_score(y_true_all, y_pred_all, zero_division=0),
        "recall_positive": recall_score(y_true_all, y_pred_all, zero_division=0),
        "f1_positive": f1_score(y_true_all, y_pred_all, zero_division=0),
        "average_precision": average_precision_score(y_true_all, y_score_all),
        "roc_auc": roc_auc_score(y_true_all, y_score_all),
    }])

    return {
        "metrics": metrics,
        "threshold_table": threshold_table,
        "importance_summary": importance_summary,
        "cases": cases,
        "numeric_case_summary": numeric_case_summary,
        "categorical_case_summary": categorical_case_summary,
        "top_predicted_positive": top_predicted_positive,
        "top_true_positives": top_tp,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }