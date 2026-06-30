"""analysis_field_additivity.py
Tests whether the Blume-Capel effective field beta*h is an ADDITIVE linear function of the raw
per-vote microclimate variables, i.e. whether a single scalar discomfort coordinate exists.

beta*h = 0.5 * ln[P(U)/P(C)]  is, at the vote level, half the log-odds of discomfort-vs-comfort.
So a logistic regression of [U vs C] (neutrals dropped) on (T_corr, f_mix, f_sun, wind_sc) has
linear predictor eta = ln[P(U)/P(C)] = 2*beta*h. Its coefficients gamma give the per-variable
field gradients (gamma/2), and the RATIOS recover the T_env weights independently of the grid
calibration:
    k_mix_hat  =  gamma_mix  / gamma_Tcorr
    k_sun_hat  =  gamma_sun  / gamma_Tcorr
    k_wind_hat = -gamma_wind / gamma_Tcorr      (compare to calibrated 3.0, 9.5, 16.0)

We also ask whether quadratic/interaction terms improve out-of-sample fit (walk-grouped CV).
If they don't, the variables genuinely sum into one additive coordinate = T_env (up to scale).
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")
RNG=np.random.default_rng(0)
CAL=dict(f_mix=3.0,f_sun=9.5,wind_sc=16.0)  # grid-calibrated weights

def irls(X,y,ridge=1e-6,iters=100):
    """Newton-Raphson logistic. X includes intercept column. Returns beta."""
    b=np.zeros(X.shape[1])
    for _ in range(iters):
        eta=np.clip(X@b,-30,30); p=1/(1+np.exp(-eta)); W=np.maximum(p*(1-p),1e-9)
        H=(X*W[:,None]).T@X+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: break
        b+=step
        if np.max(np.abs(step))<1e-10: break
    return b

def design(df,cols,center,means=None):
    M={}
    if means is None: means={c:df[c].mean() for c in center}
    parts=[np.ones(len(df))]
    for c in cols:
        x=df[c].to_numpy(float)
        if c in means: x=x-means[c]
        parts.append(x)
    return np.column_stack(parts),means

d=pd.read_csv(D).dropna(subset=["state3","T_corr","f_mix","f_sun","wind_sc","T_env","score","walk"]).copy()
d=d[d.state3.isin(["comfortable","uncomfortable"])].copy()   # field h is exactly the C/U ratio; drop N
d["y"]=(d.state3=="uncomfortable").astype(float)
print("n votes (C/U only) =",len(d),"| walks =",d.walk.nunique(),
      "| U fraction =%.3f"%d.y.mean())

BASE=["T_corr","f_mix","f_sun","wind_sc"]
X,means=design(d,BASE,BASE)
b=irls(X,d.y.to_numpy(float))
names=["intercept"]+BASE
gam=dict(zip(names,b))
gT=gam["T_corr"]
print("\n--- linear multivariate logistic  eta = ln[P(U)/P(C)] = 2*beta*h ---")
for nm in names: print(f"  gamma[{nm:8s}] = {gam[nm]:+.4f}")
print("\n--- recovered weights k_hat = gamma_x/gamma_Tcorr  (calibrated in parens) ---")
ratios={}
for c,sign in [("f_mix",1),("f_sun",1),("wind_sc",-1)]:
    k=sign*gam[c]/gT; ratios[c]=k
    print(f"  k[{c:8s}] = {k:6.2f} degC   (calibrated {CAL[c]:.1f})")

# walk-stratified bootstrap CI on ratios
walks=d.walk.unique(); widx={w:d.index[d.walk==w].to_numpy() for w in walks}
boot={c:[] for c in ratios}
for _ in range(800):
    samp=d.loc[np.concatenate([widx[w] for w in RNG.choice(walks,len(walks),replace=True)])]
    Xb,_=design(samp,BASE,BASE,means)
    bb=irls(Xb,samp.y.to_numpy(float)); gb=dict(zip(names,bb))
    if abs(gb["T_corr"])<1e-6: continue
    for c,sign in [("f_mix",1),("f_sun",1),("wind_sc",-1)]:
        boot[c].append(sign*gb[c]/gb["T_corr"])
print("\n--- walk-stratified bootstrap 95% CI on the recovered weights ---")
for c in ratios:
    lo,hi=np.percentile(boot[c],[2.5,97.5])
    inside = lo<=CAL[c]<=hi
    print(f"  k[{c:8s}] = {ratios[c]:6.2f}  CI[{lo:6.2f},{hi:6.2f}]  calibrated {CAL[c]:4.1f}  {'OK inside' if inside else 'OUTSIDE'}")

# implied index vs calibrated T_env: Spearman with comfort score
from scipy.stats import spearmanr
d["T_env_hat"]=d.T_corr+ratios["f_mix"]*d.f_mix+ratios["f_sun"]*d.f_sun-(-1)*0  # build properly:
d["T_env_hat"]=d.T_corr+ratios["f_mix"]*d.f_mix+ratios["f_sun"]*d.f_sun-ratios["wind_sc"]*d.wind_sc
print("\n--- implied coordinate check (Spearman |rho| with comfort score, C/U subset) ---")
for c in ["T_corr","T_env","T_env_hat"]:
    print(f"  |rho|({c:9s}, score) = {abs(spearmanr(d[c],d.score).correlation):.3f}")
