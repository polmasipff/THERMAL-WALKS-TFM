"""
build_T_env.py  --  How the effective environmental temperature T_env is constructed.

This file has ONE job: document, transparently and reproducibly, how T_env is built from
the bare corrected air temperature T_corr plus three survey-derived microclimate factors.
It reconstructs T_env from its inputs and checks it against the stored column to machine
precision. Run:  python build_T_env.py

--------------------------------------------------------------------------------------------
1. THE INPUTS  (all already present in votes_with_Tenv_comfort.csv)
--------------------------------------------------------------------------------------------
T_corr   : corrected air temperature (degC). Raw MeteoTracker reading referenced to a fixed
           station so stops visited at different times of day are comparable:
               T_corr(stop) = T_raw(stop) - T_fixed(t) + <T_fixed>.
           This is the BARE air-temperature field used in the Blume-Capel chapter.

The next three are dimensionless in [0,1], built from the per-stop perception ballots
(upstream notebook: ORGANISED/temperatura_efectiva_sol.ipynb):

f_sun    : fraction of participants who reported being "In full sun" at the stop
               f_sun = (# "In full sun") / (# sun-exposure votes).
f_mix    : fraction who reported "In a mixture of sun and shadow"
               f_mix = (# "mixture of sun and shadow") / (# sun-exposure votes).
wind_sc  : normalised wind intensity from the 5-point wind scale
           ["not windy","very light","light","moderate","strong"] with weights [0,1,2,3,4]:
               wind_sc = sum(weight * votes) / (4 * total_votes)   in [0,1].

--------------------------------------------------------------------------------------------
2. THE COEFFICIENTS  (frozen, from grid_coefficients_comfort.csv)
--------------------------------------------------------------------------------------------
        k_mix = 3.0 degC ,  k_sun = 9.5 degC ,  k_wind = 16.0 degC

They were NOT assumed: they come from a grid search that MAXIMISES |Spearman(T_env, comfort)|
(objective rho = 0.363). Two honest caveats that matter downstream:
  * Because the coefficients are calibrated against the comfort vote, T_env is comfort-tuned.
    That is why the Blume-Capel chapter uses the comfort-blind T_corr as the external field
    (using T_env there would be circular).
  * k_wind = 16.0 sits at the top of the search grid, so it should be read as "wind has a
    large local cooling effect", not as a literal 16 degC equivalent.

(The companion coordinate T_rad = T_corr + 3.0*f_mix + 10.5*f_sun is the no-wind version.)

--------------------------------------------------------------------------------------------
3. THE FORMULA
--------------------------------------------------------------------------------------------
        T_env = T_corr + k_mix*f_mix + k_sun*f_sun - k_wind*wind_sc
              = T_corr + 3.0*f_mix   + 9.5*f_sun   - 16.0*wind_sc

Reading: start from the bare air temperature, add a little for partial sun, more for full
sun (solar load), and subtract for wind (convective cooling). Sun and wind enter with
opposite sign because they are opposing forcings.
"""

import os
import numpy as np
import pandas as pd

# Frozen, calibrated coefficients (degC)
K_MIX = 3.0
K_SUN = 9.5
K_WIND = 16.0

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "votes_with_Tenv_comfort.csv")


def build_T_env(df):
    """Construct T_env from T_corr and the three microclimate factors."""
    return (df["T_corr"]
            + K_MIX * df["f_mix"]
            + K_SUN * df["f_sun"]
            - K_WIND * df["wind_sc"])


def main():
    df = pd.read_csv(DATA)

    T_env_rebuilt = build_T_env(df)

    # Verify against the stored column
    err = np.nanmax(np.abs(T_env_rebuilt - df["T_env"]))
    print("Reconstructed T_env vs stored column:")
    print(f"  max abs error = {err:.2e}   ({'OK' if err < 1e-9 else 'MISMATCH'})")

    print("\nContribution of each term (mean over all votes, degC):")
    print(f"  T_corr base        : {df['T_corr'].mean():6.2f}")
    print(f"  + {K_MIX:>4} * f_mix   : {K_MIX*df['f_mix'].mean():+6.2f}   (partial sun)")
    print(f"  + {K_SUN:>4} * f_sun   : {K_SUN*df['f_sun'].mean():+6.2f}   (full sun)")
    print(f"  - {K_WIND:>4} * wind_sc : {-K_WIND*df['wind_sc'].mean():+6.2f}   (wind cooling)")
    print(f"  = T_env mean       : {T_env_rebuilt.mean():6.2f}")

    print(f"\nRanges: T_corr [{df.T_corr.min():.1f}, {df.T_corr.max():.1f}]  "
          f"T_env [{df.T_env.min():.1f}, {df.T_env.max():.1f}] degC  "
          f"-> T_env spreads wider because it adds local sun/wind contrast.")


if __name__ == "__main__":
    main()
