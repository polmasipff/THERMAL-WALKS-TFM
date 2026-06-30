"""Shared style + data loading for Chapter 3 figures."""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- paths ----
HERE = os.path.dirname(os.path.abspath(__file__))
# repo root holds the data folder
DATA = os.path.join(HERE, "..","data",
                    "votes_with_Tenv_comfort.csv")
OUT = HERE  # figures written next to the scripts

CM = 1 / 2.54  # cm -> inch

# coordinate colours (pipeline §2)
COL = {
    "T_corr": "#1f77b4",   # blue
    "HDX":    "#7f7f7f",   # grey
    "T_rad":  "#ff7f0e",   # orange
    "T_env":  "#d62728",   # red
}
# state colours (F5)
COL_STATE = {"comfortable": "#1f77b4", "neutral": "#7f7f7f", "uncomfortable": "#d62728"}

# nice display names
DISP = {"T_corr": r"$T_{\rm corr}$", "HDX": "HDX",
        "T_rad": r"$T_{\rm rad}$", "T_env": r"$T_{\rm env}$"}


def set_style():
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "font.family": "DejaVu Sans",
    })


def load():
    df = pd.read_csv(DATA)
    # HDX coordinate name in this file is HDX_corr
    if "HDX" not in df.columns and "HDX_corr" in df.columns:
        df = df.rename(columns={"HDX_corr": "HDX"})
    return df


def save(fig, name):
    png = os.path.join(OUT, name + ".png")
    pdf = os.path.join(OUT, name + ".pdf")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    print("wrote", png, "and", pdf)


def set_paper_style():
    """Paper-style: larger legible fonts, no grid, clean spines. Titles go in the caption,
    legends have no frame. Call after set_style()."""
    mpl.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.grid": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })
