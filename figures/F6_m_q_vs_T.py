"""F6 (Chapter 4) - Order parameter m(T) and quadrupolar density q(T).
Sign convention: S=-1 comfortable, S=0 neutral, S=+1 uncomfortable.
m(T) = P(U) - P(C);  q(T) = P(U) + P(C) = 1 - P(N).
Walk-stratified bootstrap bands. m=0 crossing marked. Critical window shaded.
Size 14x5 cm. PNG 300dpi + PDF."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import load, set_style, save, CM

set_style()
df = load()

XCOL = "T_corr"
BIN_W = 1.5
MIN_N = 25
XMIN, XMAX = 22, 34
N_BOOT = 600
RNG = np.random.default_rng(0)

edges = np.arange(XMIN, XMAX + BIN_W, BIN_W)
centers = 0.5 * (edges[:-1] + edges[1:])

is_C = (df.state3 == "comfortable").to_numpy()
is_U = (df.state3 == "uncomfortable").to_numpy()
x = df[XCOL].to_numpy()
walks = df.walk.to_numpy()
uw = np.unique(walks)


def m_q_profiles(idx):
    xb = x[idx]; cb = is_C[idx]; ub = is_U[idx]
    m = np.full(len(centers), np.nan); q = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        sel = (xb >= edges[i]) & (xb < edges[i + 1])
        n = sel.sum()
        if n < MIN_N:
            continue
        pC = cb[sel].mean(); pU = ub[sel].mean()
        m[i] = pU - pC; q[i] = pU + pC
    return m, q


m0, q0 = m_q_profiles(np.arange(len(df)))

# walk-stratified bootstrap
mb = np.full((N_BOOT, len(centers)), np.nan)
qb = np.full((N_BOOT, len(centers)), np.nan)
walk_idx = {w: np.where(walks == w)[0] for w in uw}
for b in range(N_BOOT):
    samp = np.concatenate([RNG.choice(walk_idx[w], size=len(walk_idx[w]), replace=True)
                           for w in RNG.choice(uw, size=len(uw), replace=True)])
    mb[b], qb[b] = m_q_profiles(samp)

m_lo, m_hi = np.nanpercentile(mb, [2.5, 97.5], axis=0)
q_lo, q_hi = np.nanpercentile(qb, [2.5, 97.5], axis=0)


def crossing(c, y):
    good = ~np.isnan(y)
    c, y = c[good], y[good]
    for i in range(len(c) - 1):
        if y[i] == 0:
            return c[i]
        if y[i] * y[i + 1] < 0:
            t = y[i] / (y[i] - y[i + 1])
            return c[i] + t * (c[i + 1] - c[i])
    return np.nan


m0_cross = crossing(centers, m0)
boot_cross = np.array([crossing(centers, mb[b]) for b in range(N_BOOT)])
cross_lo, cross_hi = np.nanpercentile(boot_cross, [2.5, 97.5])
print(f"m=0 crossing = {m0_cross:.2f} C  CI [{cross_lo:.2f}, {cross_hi:.2f}]")
print(f"q median (binned) = {np.nanmedian(q0):.3f}")

fig, (axm, axq) = plt.subplots(1, 2, figsize=(14 * CM, 5 * CM))

# critical window (from frozen numbers, chi-tilde >= 0.95)
for ax in (axm, axq):
    ax.axvspan(27.1, 30.1, color="0.88", zorder=0)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xlabel(r"Air temperature $T_{\rm corr}$ (°C)")

good = ~np.isnan(m0)
axm.fill_between(centers[good], m_lo[good], m_hi[good], color="#d62728", alpha=0.2, lw=0)
axm.plot(centers[good], m0[good], color="#d62728", lw=1.5, marker="o", ms=2.5)
axm.axhline(0, ls="--", color="0.4", lw=0.8)
axm.set_ylabel(r"$m(T) = P(U) - P(C)$")
axm.set_title("Order parameter", fontsize=8)
axm.annotate("$m=0$ in critical\nwindow (~29 °C)", (28.6, 0),
             textcoords="offset points", xytext=(2, 26), fontsize=6, color="#d62728")

goodq = ~np.isnan(q0)
axq.fill_between(centers[goodq], q_lo[goodq], q_hi[goodq], color="#1f77b4", alpha=0.2, lw=0)
axq.plot(centers[goodq], q0[goodq], color="#1f77b4", lw=1.5, marker="o", ms=2.5)
axq.axhline(0.72, ls="--", color="0.4", lw=0.8)
axq.set_ylabel(r"$q(T) = P(U) + P(C)$")
axq.set_ylim(0, 1)
axq.set_title("Quadrupolar density", fontsize=8)
axq.annotate(r"$q \approx 0.72$ (constant)", (XMIN + 0.5, 0.78), fontsize=6, color="#1f77b4")

save(fig, "F6_m_q_vs_T")
