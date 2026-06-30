
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional



# ----------------------------
# Temperature / HDX overlap tables and plots
# ----------------------------

def make_binned_overlap_table(df: pd.DataFrame,
                              value_col: str,
                              outcome_col: str,
                              bins=10,
                              strategy: str = 'quantile',
                              dropna: bool = True) -> pd.DataFrame:
    """
    Returns counts and proportions of outcome categories inside bins of value_col.
    strategy: 'quantile' or 'equal_width'
    bins: int or sequence
    """
    tmp = df[[value_col, outcome_col]].copy()
    if dropna:
        tmp = tmp.dropna(subset=[value_col, outcome_col])

    if tmp.empty:
        raise ValueError(f'No rows available after dropping NA for {value_col} and {outcome_col}.')

    if isinstance(bins, int):
        if strategy == 'quantile':
            # duplicates='drop' avoids failure when repeated values are heavy
            tmp['_bin'] = pd.qcut(tmp[value_col], q=bins, duplicates='drop')
        elif strategy == 'equal_width':
            tmp['_bin'] = pd.cut(tmp[value_col], bins=bins, include_lowest=True)
        else:
            raise ValueError("strategy must be 'quantile' or 'equal_width'")
    else:
        tmp['_bin'] = pd.cut(tmp[value_col], bins=bins, include_lowest=True)

    tmp['_bin'] = tmp['_bin'].astype(str)

    counts = (
        tmp.groupby(['_bin', outcome_col], dropna=False)
        .size()
        .reset_index(name='count')
    )

    total = counts.groupby('_bin')['count'].transform('sum')
    counts['prop'] = counts['count'] / total

    # Bin summary stats to aid interpretation
    bin_summary = (
        tmp.groupby('_bin')[value_col]
        .agg(bin_n='size', bin_min='min', bin_max='max', bin_mean='mean', bin_median='median')
        .reset_index()
    )

    out = counts.merge(bin_summary, on='_bin', how='left')
    out = out.sort_values(['bin_mean', outcome_col]).reset_index(drop=True)
    out = out.rename(columns={'_bin': 'bin'})
    return out


def plot_binned_overlap(overlap_df: pd.DataFrame,
                        outcome_col: str,
                        normalize: bool = True,
                        figsize=(12, 6),
                        title: Optional[str] = None,
                        output_path: Optional[str] = None):
    """Stacked bar plot for overlap_df returned by make_binned_overlap_table."""
    value_name = 'prop' if normalize else 'count'
    pivot = overlap_df.pivot_table(index='bin', columns=outcome_col, values=value_name, fill_value=0)

    # Order bins by mean if available
    if 'bin_mean' in overlap_df.columns:
        order = (
            overlap_df[['bin', 'bin_mean']]
            .drop_duplicates()
            .sort_values('bin_mean')['bin']
            .tolist()
        )
        pivot = pivot.reindex(order)

    ax = pivot.plot(kind='bar', stacked=True, figsize=figsize)
    ax.set_ylabel('Proportion' if normalize else 'Count')
    ax.set_xlabel('Bin')
    ax.set_title(title or f'Overlap of {outcome_col} across bins')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
    return ax


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


def binomial_band(p, n, z=1.0):
    se = np.sqrt((p * (1 - p)) / n)
    low = np.clip(p - z * se, 0, 1)
    high = np.clip(p + z * se, 0, 1)
    return low, high

def plot_bin_overlap_with_errorbars(overlap_df, outcome_col, xcol="bin_mean", z=1.0, ylim=(0,1), title=None, output_path=None):
    df = overlap_df.copy()
    cats = list(df[outcome_col].dropna().unique())
    plt.figure(figsize=(10, 6))
    for cat in cats:
        sub = df[df[outcome_col] == cat].sort_values(xcol).copy()
        p = sub["prop"].values
        n = sub["bin_n"].values
        lo, hi = binomial_band(p, n, z=z)
        yerr = np.vstack([p-lo, hi-p])
        plt.errorbar(sub[xcol].values, p, yerr=yerr, marker="o", capsize=3, label=cat)
    plt.xlabel(xcol)
    plt.ylabel("Proportion within bin")
    plt.ylim(*ylim)
    plt.grid(True, alpha=0.25)
    plt.title(title or f"{outcome_col} by bin")
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=220, bbox_inches='tight')
    return plt.gca()


def plot_grouping_scorecard(scorecard_df, metric="bal_acc_adj_chance", title=None, output_path=None):
    plot_df = scorecard_df.copy().sort_values(metric, ascending=True).reset_index(drop=True)
    plot_df["rank"] = range(len(plot_df))
    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df)*0.35)))
    fig.patch.set_facecolor("#F7F7F9"); ax.set_facecolor("#F7F7F9")
    colors = plt.cm.RdYlGn([i / max(1,len(plot_df)-1) for i in range(len(plot_df))])
    ax.plot(plot_df[metric], plot_df["rank"], "-o", color="#cccccc", linewidth=1.5, zorder=1)
    for _, row in plot_df.iterrows():
        idx = int(row["rank"])
        ax.scatter(row[metric], idx, color=colors[idx], s=95, zorder=3, edgecolors="white", linewidths=1.0)
        ax.text(row[metric] + 0.002, idx, row["partition"] if "partition" in row else row["outcome"], fontsize=8, va="center", color=colors[idx])
    ax.set_yticks(plot_df["rank"]); ax.set_yticklabels(plot_df["model"] if "model" in plot_df else plot_df["outcome"], fontsize=8)
    ax.set_xlabel(metric); ax.set_ylabel("scheme / model")
    ax.set_title(title or f"Grouping comparison sorted by {metric}")
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=220, bbox_inches='tight')
    return ax
