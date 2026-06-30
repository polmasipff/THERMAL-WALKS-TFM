
"""
Compatibility wrapper.

If your notebook imports thermal_model_fixes, keep using it.
This file exposes the same make_model_pipelines interface expected by the new scripts.
"""
from typing import Dict
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def build_preprocessor(X: pd.DataFrame, dense: bool = False):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    if dense:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1))
        ])
    else:
        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

def make_model_pipelines(X: pd.DataFrame, include_xgb: bool = False) -> Dict[str, Pipeline]:
    sparse_pre = build_preprocessor(X, dense=False)
    dense_pre = build_preprocessor(X, dense=True)
    models = {
        "logreg": Pipeline([("pre", sparse_pre), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", multi_class="auto"))]),
        "rf": Pipeline([("pre", dense_pre), ("clf", RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample", min_samples_leaf=3, n_jobs=-1))]),
        "hgb": Pipeline([("pre", dense_pre), ("clf", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=350, max_leaf_nodes=31, min_samples_leaf=20, random_state=42))]),
    }
    if include_xgb and HAS_XGB:
        models["xgb"] = Pipeline([("pre", dense_pre), ("clf", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective="multi:softprob", eval_metric="mlogloss", random_state=42, n_jobs=4))])
    return models
