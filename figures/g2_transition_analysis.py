"""G2: C/N/U transition matrices, stationary distributions, net fluxes and cycle
currents resolved by T_env regime, with walk-stratified bootstrap. Also flows / R_D
vs T_env. Saves results to g2_results/ and prints a summary."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..","data", "markovian_analysis_baseline.csv")
OUT = os.path.join(HERE, "g2_results"); os.makedirs(OUT, exist_ok=True)

ORDER = ["comfortable", "neutral", "uncomfortable"]
NUM = {s: i for i, s in enumerate(ORDER)}
CUTS = [24.6, 29.8]
RNG = np.random.default_rng(0)

d = pd.read_csv(BASE).dropna(subset=["comfort3", "T_env", "ID", "walk_id", "stop_idx"]).copy()
d["s"] = d["comfort3"].map(NUM)
d = d.sort_values(["ID", "walk_id", "stop_idx"])
g = d.groupby(["ID", "walk_id"])
d["s_next"] = g["s"].shift(-1)
d["stop_next"] = g["stop_idx"].shift(-1)
tr = d[d["s_next"].notna()].copy()
tr["s_next"] = tr["s_next"].astype(int)
tr["reg"] = np.where(tr["T_env"] < CUTS[0], "cold",
                     np.where(tr["T_env"] < CUTS[1], "central", "hot"))


def count_matrix(frame):
    M = np.zeros((3, 3))
    for a, b in zip(frame["s"], frame["s_next"]):
        M[a, b] += 1
    return M


def P_from_counts(M):
    rs = M.sum(1, keepdims=True)
    return np.divide(M, rs, where=rs > 0)


def stationary(P):
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def net_fluxes(P, pi):
    # F_ij = pi_i P_ij - pi_j P_ji ; cycle current around 0->1->2->0
    F = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            F[i, j] = pi[i] * P[i, j] - pi[j] * P[j, i]
    Jc = (F[0, 1] + F[1, 2] + F[2, 0]) / 3.0  # signed cycle current (C->N->U->C)
    return F, Jc


def analyse(frame):
    M = count_matrix(frame)
    P = P_from_counts(M)
    pi = stationary(P)
    F, Jc = net_fluxes(P, pi)
    return M, P, pi, F, Jc


walks = tr["walk_id"].unique()
widx = {w: tr.index[tr.walk_id == w].to_numpy() for w in walks}

def boot_Jc(frame_reg, n=600):
    sub = tr.loc[tr["reg"] == frame_reg] if frame_reg else tr
    wl = sub["walk_id"].unique()
    wi = {w: sub.index[sub.walk_id == w].to_numpy() for w in wl}
    out = []
    for _ in range(n):
        chosen = RNG.choice(wl, size=len(wl), replace=True)
        idx = np.concatenate([wi[w] for w in chosen])
        _, _, _, _, Jc = analyse(tr.loc[idx])
        out.append(Jc)
    return np.array(out)


print("=== Aggregate ===")
M, P, pi, F, Jc = analyse(tr)
print("P=\n", np.round(P, 3), "\npi=", np.round(pi, 3), " Jc=", round(Jc, 5))
bJ = boot_Jc(None); print(f"  Jc boot: median={np.median(bJ):.5f} CI[{np.percentile(bJ,2.5):.5f},{np.percentile(bJ,97.5):.5f}] sign+={np.mean(bJ>0):.2f}")

rows = []
for reg in ["cold", "central", "hot"]:
    sub = tr[tr["reg"] == reg]
    M, P, pi, F, Jc = analyse(sub)
    bJ = boot_Jc(reg)
    # key channels: recovery U->C (P[2,0]), trap U->U (P[2,2]), onset N->U (P[1,2])
    print(f"\n=== {reg}  (n={len(sub)}) ===")
    print("P=\n", np.round(P, 3))
    print(f"  U->C recovery={P[2,0]:.3f}  U->U persist={P[2,2]:.3f}  N->U onset={P[1,2]:.3f}  C->U={P[0,2]:.3f}")
    print(f"  pi={np.round(pi,3)}  net C->U flux={F[0,2]:.4f}  Jc={Jc:.5f}  bootCI[{np.percentile(bJ,2.5):.5f},{np.percentile(bJ,97.5):.5f}] sign+={np.mean(bJ>0):.2f}")
    rows.append(dict(regime=reg, n=len(sub), UtoC=P[2,0], UtoU=P[2,2], NtoU=P[1,2], CtoU=P[0,2],
                     pi_C=pi[0], pi_N=pi[1], pi_U=pi[2], netCU=F[0,2], netNU=F[1,2], netUC=F[2,0],
                     Jc=Jc, Jc_lo=np.percentile(bJ,2.5), Jc_hi=np.percentile(bJ,97.5), signpos=np.mean(bJ>0)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "regime_summary.csv"), index=False)

# ---- flows and R_D vs T_env (continuous bins) ----
def flows_bin(frame):
    fromU = frame[frame.s == 2]; fromN = frame[frame.s == 1]; fromC = frame[frame.s == 0]
    rec = np.mean(fromU.s_next < 2) if len(fromU) else np.nan      # U -> C or N
    ons = np.mean(fromN.s_next == 2) if len(fromN) else np.nan      # N -> U
    closs = np.mean(fromC.s_next > 0) if len(fromC) else np.nan     # C -> N or U
    return rec, ons, closs, len(fromU), len(fromN)

bins = np.arange(18, 36, 2.0)
print("\n=== flows vs T_env ===  Tc  recov  onset  RD")
fr = []
for i in range(len(bins) - 1):
    sub = tr[(tr.T_env >= bins[i]) & (tr.T_env < bins[i+1])]
    if len(sub) < 25: continue
    rec, ons, closs, nu, nn = flows_bin(sub)
    rd = ons / rec if rec and rec > 0 else np.nan
    c = (bins[i]+bins[i+1])/2
    fr.append(dict(Tenv=c, recovery=rec, onset=ons, comfort_loss=closs, RD=rd, nU=nu, nN=nn))
    print(f"  {c:4.1f}  rec={rec:.2f} ons={ons:.2f} RD={rd:.2f}")
pd.DataFrame(fr).to_csv(os.path.join(OUT, "flows_vs_Tenv.csv"), index=False)
print("\nsaved to", OUT)
