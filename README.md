# Urban thermal walks as a forced stochastic state system

**Effective environmental coordinates and comfort-transition dynamics in
citizen-science thermal walks.**

Code and figures for a Master's thesis (TFM) in complex-systems physics. The
project treats urban thermal comfort not as a deterministic function of
temperature but as a **noisy, heterogeneous, forced state process**: static
occupation probabilities reveal a population-level comfort↔discomfort
crossover, while the *transition dynamics* reveal how comfort is generated,
lost, recovered, trapped, and forced by the urban microclimate.

---

## Scientific summary

The campaign ("Cròniques de la calor / Heat Chronicles") records ~1,860
comfort votes across 48 warm-season walks (~210 stops, five neighbourhoods),
each paired with on-the-spot microclimate measurements. From this the thesis:

1. **Reduces the state space.** The seven-point thermal-comfort vote (TCV) is
   *locally over-resolved*: category distributions overlap strongly along
   temperature and adjacent steps are mostly not separable. The analysis
   reduces it to a three-state **Comfort / Neutral / Uncomfortable (C/N/U)**
   spin-1 space.

2. **Builds an effective coordinate.** A grid-calibrated environmental
   temperature
   `T_env = T_corr + 3.0·f_mix + 9.5·f_sun − 16.0·wind_sc`
   organises comfort better than bare air temperature, radiative temperature,
   or Humidex by combining slow regime forcing (sun) with local ventilation
   (wind).

3. **Reads comfort as non-equilibrium dynamics.** Stop-to-stop C/N/U
   transitions are approximately first-order Markovian in the bulk, with
   regime-dependent transition matrices, a **cycle current that reverses sign**
   across the crossover, a cycle **affinity** that standardises the small net
   current into a scale-free irreversibility measure, **trap-like memory** of
   the very-uncomfortable state, and signatures of **incomplete local
   adaptation**.

The thesis claim, in one line: *urban thermal comfort behaves like a
heterogeneous, forced, stochastic state system — static occupation shows a
broad crossover, but the transition dynamics reveal the mechanisms of onset,
recovery, trapping, memory and incomplete adaptation.*

---

## Repository layout

```
thermal-walks-repo/
├── README.md            ← you are here
├── LICENSE              ← MIT (code only; data licensed separately)
├── CITATION.cff         ← how to cite this work
├── requirements.txt     ← Python dependencies
├── .gitignore           ← excludes all data and pipeline intermediates
│
├── data/                ← input datasets (NOT tracked; only data/README.md is)
│   └── README.md           data dictionary + how to obtain / verify the inputs
│
├── src/
│   └── build_T_env.py      transparent, verifiable construction of T_env
│
├── figures/             ← scripts that regenerate every thesis figure
│   ├── _common.py, _cap5_data.py, _sep_core.py   shared loaders/helpers
│   ├── F*.py               one script per figure  (→ figures/README.md map)
│   ├── *.pdf / *.png       rendered figures (tracked)
│   └── README.md           figure → chapter mapping
│
└── notebooks/           ← analysis record, organised by chapter
    ├── 01_effective_temperature/
    ├── 02_state_space_ch3/
    ├── 03_blume_capel_ch4/
    ├── 04_transition_dynamics_ch5/
    ├── 05_exploratory/
    └── README.md           run convention + canonical notebooks
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt

# 1. add the data (see data/README.md)
#    -> data/votes_with_Tenv_comfort.csv, data/markovian_analysis_baseline.csv, ...

# 2. verify the effective-temperature construction
python src/build_T_env.py

# 3. regenerate a thesis figure
cd figures && python F5_pstate_Tenv.py
```

## Reproducibility

- **Figures** are turnkey: each script in `figures/` reads `../data/` and
  writes its own `.pdf`/`.png`. The figure → chapter map is in
  [`figures/README.md`](figures/README.md).
- **`src/build_T_env.py`** reconstructs `T_env` (and `T_rad`) from their inputs
  and checks them against the stored columns to machine precision
  (max error ≈ 1e-15).
- **Notebooks** are the authored analysis record; see
  [`notebooks/README.md`](notebooks/README.md) for the run convention.
- **Data is never committed.** The repository ships code and figures; the
  measurement data are obtained separately (see
  [`data/README.md`](data/README.md)).

## Data availability

The thermal-walk data come from the *Cròniques de la calor / Heat Chronicles*
citizen-science campaign and are governed by the original providers' licence.
They are not redistributed here. `data/README.md` lists exactly which files the
code expects and documents every derived column.

## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff).

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)). The data are
**not** covered by this license and must be obtained from their original
source.
