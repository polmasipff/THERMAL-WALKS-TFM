
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

from thermal_model_fixes import make_model_pipelines

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
        "distance_drinking_fountain", "distance_green_zone,particiants_clothing"
    ],
    "hdx_sun_wind_context": [
        "<HDX-HDX_fixed+<HDX>>", "sun", "wind", "SVF_1.5m", "NDVI_1.5m",
        "climate_shelter", "surface_type", "space_category", "IVAC"
    ],
}

#CANDIDATES = {
 #   "comfort3": "VC+C+SC || N || SU+U+VU",
  #  "comfort7": "VC || C || SC || N || SU || U || VU",
   # "comfort3_option1": "VC+C+SC+N || SU+U || VU",
    #"comfort3_option2": "VC+C+SC || N+SU+U || VU",
    #"comfort4_soft": "VC+C || SC+N || SU || U+VU",
   # "comfort4_option1": "VC || C+SC+N || SU+U || VU",
   # "comfort3_UNoption": "VC+C+SC+N+SU || U || VU",
#}

def _run_one(df, outcome_col, feature_set_name="full_context", group_col="ID", n_splits=5, include_xgb=False):
    cols = [c for c in DEFAULT_FEATURES[feature_set_name] if c in df.columns]
    tmp = df[[outcome_col, group_col] + cols].dropna(subset=[outcome_col, group_col]).copy()
    X = tmp[cols].copy()
    y = tmp[outcome_col].copy()
    groups = tmp[group_col].copy()

    models = make_model_pipelines(X, include_xgb=include_xgb)
    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for model_name, model in models.items():
        accs, bals, f1s = [], [], []
        le = None
        y_fit = y.copy()
        if model_name == "xgb":
            le = LabelEncoder()
            y_fit = pd.Series(le.fit_transform(y), index=y.index)

        for tr, te in gkf.split(X, y_fit, groups):
            fitted = clone(model)
            fitted.fit(X.iloc[tr], y_fit.iloc[tr])
            pred = fitted.predict(X.iloc[te])

            if model_name == "xgb":
                pred = le.inverse_transform(pred.astype(int))
                y_true = y.iloc[te].to_numpy()
            else:
                y_true = y.iloc[te].to_numpy()

            accs.append(accuracy_score(y_true, pred))
            bals.append(balanced_accuracy_score(y_true, pred))
            f1s.append(f1_score(y_true, pred, average="macro"))

        k = y.nunique()
        ba = float(np.mean(bals))
        rows.append({
            "outcome": outcome_col,
            "candidate_label": CANDIDATES.get(outcome_col, outcome_col),
            "feature_set": feature_set_name,
            "model": model_name,
            "n": len(tmp),
            "n_groups": k,
            "largest_class_share": y.value_counts(normalize=True).max(),
            "accuracy_mean": np.mean(accs),
            "bal_acc_mean": ba,
            "f1_macro_mean": np.mean(f1s),
            "bal_acc_adj_chance": (ba - 1/k) / (1 - 1/k) if k > 1 else np.nan,
        })
    return pd.DataFrame(rows)




def plot_confusiont(
    df,
    outcome_col,
    feature_set_name="full_context",
    model_name="rf",
    group_col="ID",
    n_splits=5,
    include_xgb=False,
    output_path=None,
    label_order=None,
    show_support=True,
    annotate="both",
    cmap="Blues",
):
    cols = [c for c in DEFAULT_FEATURES[feature_set_name] if c in df.columns]
    tmp = df[[outcome_col, group_col] + cols].dropna(subset=[outcome_col, group_col]).copy()

    X = tmp[cols].copy()
    y = tmp[outcome_col].copy().astype(str).str.strip()
    groups = tmp[group_col].copy()

    if label_order is None:
        DEFAULT_LABEL_ORDERS = {
            "comfort7": ["Very comfortable","Comfortable","Slightly comfortable","Neutral","Slightly uncomfortable","Uncomfortable","Very uncomfortable"],
            "comfort3": ["comfortable", "neutral", "uncomfortable"],
            "comfort3_option1": ["comfortable-neutral", "uncomfortable", "very uncomfortable"],
            "comfort3_option2": ["comfortable", "neutral-uncomfortable", "very uncomfortable"],
            "comfort4_soft": ["comfortable", "near_neutral", "slightly_uncomfortable", "uncomfortable"],
            "comfort4_option1": ["very comfortable", "near_neutral", "slightly_uncomfortable", "uncomfortable"],
            "comfort3_UNoption": ["comfortable", "uncomfortable", "very uncomfortable"],
            "sens7": ["Cool", "Slightly cool", "Neutral", "Slightly warm", "Warm", "Hot", "V    ery hot"],
            "sens3": ["cool", "neutral", "warm"],
            "sens3_option1": ["cool", "neutral-warm", "hot"],
            "sens3_option2": ["cool-neutral", "warm-hot", "very hot"],
            "sens3_option3": ["cool", "neutral-slightly warm", "warm-hot"],
            "sens4_option1": ["cool", "slightly-neutral", "warm-hot", "very hot"],
            "sens4_option2": ["cool", "neutral", "warm-hot", "very hot"],
        }
        label_order = DEFAULT_LABEL_ORDERS.get(outcome_col, sorted(y.dropna().unique()))

    label_order = [str(x).strip() for x in label_order]

    print("Label order repr:")
    print([repr(x) for x in label_order])

    models = make_model_pipelines(X, include_xgb=include_xgb)
    model = models[model_name]

    le = None
    y_fit = y.copy()
    if model_name == "xgb":
        le = LabelEncoder()
        le.fit(label_order)
        y_fit = pd.Series(le.transform(y), index=y.index)

    gkf = GroupKFold(n_splits=n_splits)
    y_true_all, y_pred_all = [], []

    for tr, te in gkf.split(X, y_fit, groups):
        fitted = clone(model)
        fitted.fit(X.iloc[tr], y_fit.iloc[tr])
        pred = fitted.predict(X.iloc[te])

        if model_name == "xgb":
            pred = le.inverse_transform(pred.astype(int))

        y_true = y.iloc[te].to_numpy()
        y_true_all.extend([str(v).strip() for v in y_true.tolist()])
        y_pred_all.extend([str(v).strip() for v in pred.tolist()])

    print("Unique y_true_all repr:")
    print(sorted(repr(x) for x in pd.Series(y_true_all).dropna().unique()))

    y_true_set = set(pd.Series(y_true_all).dropna().unique())
    y_pred_set = set(pd.Series(y_pred_all).dropna().unique())
    label_set = set(label_order)

    print("Intersection with y_true:", label_set.intersection(y_true_set))
    print("Intersection with y_pred:", label_set.intersection(y_pred_set))

    # si hi ha labels que no apareixen, els filtrem
    label_order = [lab for lab in label_order if (lab in y_true_set) or (lab in y_pred_set)]

    if len(label_order) == 0:
        raise ValueError(
            f"No labels from label_order appear in y_true/y_pred for outcome_col={outcome_col}.\n"
            f"y_true unique = {sorted(y_true_set)}"
        )

    cm_counts = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=label_order,
        normalize=None
    )

    cm_percent = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=label_order,
        normalize="true"
    )

    support = pd.Series(y_true_all).value_counts()

    if show_support:
        display_labels = [f"{lab}\n(n={support.get(lab, 0)})" for lab in label_order]
    else:
        display_labels = label_order

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_percent, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(label_order)))
    ax.set_yticks(np.arange(len(label_order)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{outcome_col} | {model_name} | {feature_set_name} | counts + row %")

    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = cm_counts[i, j]
            pct = cm_percent[i, j]

            if annotate == "count":
                txt = f"{count}"
            elif annotate == "percent":
                txt = f"{pct:.2f}"
            else:
                txt = f"{count}\n({pct:.2f})"

            text_color = "white" if pct > 0.45 else "#123b7a"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=220, bbox_inches="tight")

    return {
        "cm_counts": pd.DataFrame(cm_counts, index=label_order, columns=label_order),
        "cm_percent": pd.DataFrame(cm_percent, index=label_order, columns=label_order),
    }

def run_candidate_workflow(df, include_xgb=False, save_prefix="candidate"):
    out = []
    for outcome in CANDIDATES:
        for fs in DEFAULT_FEATURES:
            out.append(_run_one(df, outcome, fs, include_xgb=include_xgb))
    res = pd.concat(out, ignore_index=True)
    res = res.sort_values(["bal_acc_adj_chance", "bal_acc_mean", "f1_macro_mean"], ascending=False)
    res.to_csv(f"{save_prefix}_benchmark.csv", index=False, encoding="utf-8-sig")
    return res
