# `data/` — input datasets (not tracked in git)

Everything in this folder **except this file is git-ignored** (see the
repository `.gitignore`). The repository ships *code and figures*, not data.
This file documents which inputs the code expects, where each comes from, and
the exact definitions of the derived environmental coordinates, so the
analysis is reproducible.

Place the files listed below directly in this folder (paths in the notebooks
and figure scripts resolve to `../data/` or `../../data/`).

---

## 1. Canonical files expected here

| File | Used by | Rows | Role |
|------|---------|------|------|
| `votes_with_Tenv_comfort.csv` | `figures/` (Ch 3–4, 6), `src/build_T_env.py` | 1855 | **Static state-space dataset.** One row per comfort vote with the bare and effective temperatures. |
| `votes_with_Tenv_sensation.csv` | sensation robustness checks | — | Same construction for the TSV (sensation) scale. |
| `markovian_analysis_baseline.csv` | `figures/` (Ch 5), Ch 5 notebooks | 1857 | **Transition dataset.** One row per vote with `ID`, `walk_id`, `stop_idx`, `comfort3/comfort7`, `T_corr`, `T_env`, demographics — the basis for stop-to-stop transitions. |
| `df_dynamics_with_oof.csv` | Ch 5 notebooks | — | Dynamics dataframe with out-of-fold predictions. |
| `transitions_events.csv` | onset / trap analyses | — | Event-level transition records. |
| `stops_with_Trad.csv` | T_rad construction | — | Per-stop radiative-temperature inputs. |
| `candidate_groupings_rank_table.csv` | `figures/FA_partition_scorecard.py` | — | Grouping-diagnostics scorecard (Appendix A). |
| `all_surveys(ID).csv`, `all_surveys(stops).csv`, `all_surveys(votes).csv` | Ch 3 prepare/statistics notebooks, exploratory notebooks | — | Raw survey exports (participants / stops / votes). |
| `calibration/` | `src/build_T_env.py`, calibration discussion | — | Frozen grid-search coefficients and walk-CV metrics. |

> The three `all_surveys(*).csv` files are the aggregated exports of the
> **Cròniques de la calor / Heat Chronicles** citizen-science campaign. Obtain
> them from the original data repository (Dataverse) and drop them here.

---

## 2. Effective environmental coordinates (frozen definitions)

Both derived temperatures are reconstructed from columns already present in the
files, so they can be verified to machine precision:

```
T_rad = T_corr + 3.0*f_mix + 10.5*f_sun
T_env = T_corr + 3.0*f_mix +  9.5*f_sun - 16.0*wind_sc
```

with coefficients `k_mix = 3.0 °C`, `k_sun = 9.5 °C`, `k_wind = 16.0 °C`
(grid-calibrated against the comfort vote; `k_wind` sits at the top of the
search grid and should be read as "wind has a large local cooling effect",
not as a literal 16 °C equivalent).

Verification (run from the repo root):

```python
import pandas as pd, numpy as np
df = pd.read_csv("data/votes_with_Tenv_comfort.csv")
rec = df.T_corr + 3.0*df.f_mix + 9.5*df.f_sun - 16.0*df.wind_sc
assert np.nanmax(np.abs(rec - df.T_env)) < 1e-10
```

`src/build_T_env.py` documents and re-runs this construction end to end.

---

## 3. Key columns

`votes_with_Tenv_comfort.csv`: `walk`, `city`, `category` (7-pt TCV),
`state3` (C/N/U), `T_corr`, `HDX_corr`, `f_mix`, `f_sun`, `wind_sc`,
`T_rad`, `T_env`.

`markovian_analysis_baseline.csv`: `ID`, `walk_id`, `stop_idx`,
`comfort7`, `comfort3`, `T_corr`, `T_env`, `f_mix`, `f_sun`, `wind_sc`,
`gender`, `age`, `city`, `date`.

Do not silently switch between `T`, `T_corr` and `T_env`; every figure/notebook
states which coordinate it uses.
