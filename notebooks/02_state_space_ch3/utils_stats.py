
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, chi2_contingency

TSV7_ORDER = [
    "Cool", "Slightly cool", "Neutral", "Slightly warm", "Warm", "Hot", "Very hot"
]

TCV7_ORDER = [
    "Very comfortable", "Comfortable", "Slightly comfortable",
    "Neutral", "Slightly uncomfortable", "Uncomfortable", "Very uncomfortable"
]

def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(m, dtype=float)
    for i, p in enumerate(ranked):
        adj[i] = min((m - i) * p, 1.0)
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i-1])
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def kruskal_test(data, outcome_col, predictor_col):
    tmp = data[[outcome_col, predictor_col]].dropna().copy()
    groups = [g[predictor_col].values for _, g in tmp.groupby(outcome_col, observed=True)]
    if len(groups) < 2 or any(len(g) == 0 for g in groups):
        return None
    H, p = kruskal(*groups)
    return {"outcome": outcome_col, "predictor": predictor_col, "H": H, "p_value": p, "n": len(tmp), "k": len(groups)}

def spearman_monotonicity(data, outcome_col, predictor_col, order):
    tmp = data[[outcome_col, predictor_col]].dropna().copy()
    tmp = tmp[tmp[outcome_col].isin(order)]
    if tmp.empty: return None
    codes = {lab:i for i,lab in enumerate(order)}
    rho, p = spearmanr(tmp[outcome_col].map(codes), tmp[predictor_col])
    return {"outcome": outcome_col, "predictor": predictor_col, "spearman_rho": rho, "p_value": p, "n": len(tmp)}

def pairwise_adjacent_mwu(data, outcome_col, predictor_col, order):
    tmp = data[[outcome_col, predictor_col]].dropna().copy()
    tmp = tmp[tmp[outcome_col].isin(order)]
    results, raw_pvals, valid_pairs = [], [], []
    for a, b in zip(order[:-1], order[1:]):
        xa = tmp.loc[tmp[outcome_col] == a, predictor_col].values
        xb = tmp.loc[tmp[outcome_col] == b, predictor_col].values
        if len(xa) == 0 or len(xb) == 0:
            continue
        U, p = mannwhitneyu(xa, xb, alternative='two-sided')
        r_rb = 1 - (2 * U) / (len(xa) * len(xb))
        results.append({
            "outcome": outcome_col, "predictor": predictor_col,
            "group_a": a, "group_b": b, "U": U,
            "p_value": p, "effect_rank_biserial": r_rb,
            "n_a": len(xa), "n_b": len(xb)
        })
        raw_pvals.append(p)
        valid_pairs.append((a,b))
    if not results:
        return pd.DataFrame()
    adj = holm_adjust(raw_pvals)
    for row, p_adj in zip(results, adj):
        row["p_holm"] = p_adj
    return pd.DataFrame(results)

def chisq_test(data, outcome_col, predictor_col):
    tmp = data[[outcome_col, predictor_col]].dropna().copy()
    if tmp.empty: return None
    table = pd.crosstab(tmp[outcome_col], tmp[predictor_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.values.sum()
    r, c = table.shape
    cramers_v = np.sqrt((chi2 / n) / (min(r - 1, c - 1)))
    return {"outcome": outcome_col, "predictor": predictor_col, "chi2": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v, "n": n}
