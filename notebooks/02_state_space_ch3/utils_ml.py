import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    XGBClassifier = None


# ----------------------------
# Outcome builders
# ----------------------------




# ----------------------------
# Feature engineering
# ----------------------------

def add_ordinal_sun_wind(df: pd.DataFrame,
                         sun_col: str = 'sun',
                         wind_col: str = 'wind') -> pd.DataFrame:
    out = df.copy()

    sun_map = {
        'In full shade': 0,
        'In the shade of a tree': 1,
        'Both in the sun and shade': 2,
        'In full sun': 3,
    }
    wind_map = {
        "It's not windy": 0,
        'A little bit windy': 1,
        'A moderate wind': 2,
        'It is very windy': 3,
    }

    if sun_col in out.columns:
        out['sun_ord'] = out[sun_col].map(sun_map)
    if wind_col in out.columns:
        out['wind_ord'] = out[wind_col].map(wind_map)
    return out
