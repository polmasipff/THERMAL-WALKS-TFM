# `figures/` — thesis figure pipeline

Self-contained scripts that regenerate the figures used in the thesis. Each
script reads the canonical datasets from `../data/` (see `data/README.md`) plus
a few small derived inputs kept alongside the scripts (`_*.csv`), and writes a
`.pdf` + `.png` next to itself.

Shared helpers: `_common.py` (style + loader for the static dataset),
`_cap5_data.py` (stop-to-stop C/N/U transition builder + T_env regimes),
`_sep_core.py` (separability metrics).

Run any figure from this folder, e.g.:

```bash
cd figures
python F1_overlap_7cat.py
python F5_pstate_Tenv.py
python F_affinity.py
```

## Figure → thesis map

| Script | Output | Chapter / section |
|--------|--------|-------------------|
| `F1_overlap_7cat.py`, `F1b_means_vs_spread.py` | `F1_overlap_7cat*`, `F1b_*` | Ch 3 §3.1 — 7-category overlap (violin + mean±95% CI) |
| `F2_scale_separability.py`, `F2_withinwalk_rho.py`, `F2b_scale_std.py` | `F2*`, `F2b*` | Ch 3 — local separability vs window size Δ |
| `F3_intrawalk_vs_threshold.py` | `F3_*` | Ch 3 — within-walk contrast vs separability threshold |
| `F4_scanG_barplot.py` | `F4_*` | Ch 3 — scan-G, two-vs-three regimes |
| `F5_pstate_Tenv.py` | `F5_*` | Ch 3 — P(C/N/U \| T_env) with Wilson CI + regime cuts |
| `F6_m_q_vs_T.py`, `F_bc_mq.py`, `F_bc_fields.py`, `F_bc_alpha_heff.py` | `F6_*`, `F_bc_*` | Ch 4 — Blume-Capel m(T), q(T), fields |
| `F9_Fflux_build.py`, `F_cap5_matrices.py`, `F_dynamics.py` | `F9_regime_matrices`, `F_flux_reversal`, `F_dynamics` | Ch 5 — C/N/U and T_env-regime transition matrices, flux reversal |
| `F_affinity.py`, `F_cap5_affinity.py` | `F_affinity`, `F_cap5_affinity` | Ch 5 — cycle affinity by regime |
| `F_cap5_memory.py`, `F_recovery_onset.py` | `F_cap5_memory`, `F_recovery_onset` | Ch 5 — very-uncomfortable trap memory, recovery/onset |
| `F11_displacement.py`, `g1_displacement.py` | `F11_displacement` | Ch 5 — local stationary vs observed occupation (incomplete adaptation) |
| `F_simplex_trajectory.py` | `F_simplex_trajectory` | Ch 4/5 — simplex trajectory |
| `F_sun_wind_scales.py`, `F_forcing_scales.py`, `F_forcing_threshold.py` | `F_sun_wind_scales`, `F_forcing_*` | Ch 6 — sun (regime) vs wind (local) forcing |
| `F_occupation_regimes.py`, `F_regime_count.py` | `F_occupation_regimes`, `F_regime_count*` | Ch 3/6 — occupation regimes by coordinate |
| `F_heterogeneity*.py` | `F_heterogeneity*` | End of Ch 3 — comfort/discomfort threshold heterogeneity (incl. binary age split) |
| `FA1_scalespace_heatmap.py` | `FA1_*` | Appendix — scale-space heatmap (exploratory regime view) |
| `FA_partition_scorecard.py`, `F_cardinality.py`, `F_ml_importance.py` | `FA_*`, `F_cardinality`, `F_ml_importance` | Appendix A — grouping diagnostics, ML separability |
| `F_tsv_tcv.py`, `F_field_additivity.py` | `F_tsv_tcv*`, `F_field_additivity` | Appendix — TSV/TCV checks, field additivity |

`compute_*.py` and `make_occupation_csv.py` (re)build the small `_*.csv`
derived inputs that the figure scripts consume; the produced CSVs are committed
so the figures render without rerunning the upstream pipeline.

> Naming note: scripts are kept flat (not split into per-chapter folders) so
> the `from _common import …` imports and the `../data/` relative paths keep
> working unchanged. The chapter mapping above replaces a folder hierarchy.
