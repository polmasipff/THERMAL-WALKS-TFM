"""analysis_field_cv.py -- does a single ADDITIVE coordinate suffice?
Walk-grouped cross-validation (no walk in both train and test -> no leakage). Compares
out-of-sample log-loss and AUC of nested models for discomfort-vs-comfort:
  M0      : T_corr only
  M_env   : T_env only (the grid-calibrated additive index)
  M_lin   : T_corr + f_mix + f_sun + wind_sc (free linear, re-fit)
  M_nl    : M_lin + T_corr^2 + T_corr:f_sun + T_corr:wind_sc + f_sun:wind_sc
If M_nl does not beat M_lin/M_env out of sample, the variables sum into ONE additive
discomfort coordinate (the field is a single scalar), validating T_env structurally.
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")
RNG=np.random.default_rng(1)

def irls(X,y,ridge=1e-4,iters=200):
    b=np.zeros(X.shape[1])
    for _ in range(iters):
        eta=np.clip(X@b,-30,30); p=1/(1+np.exp(-eta)); W=np.maximum(p*(1-p),1e-9)
        H=(X*W[:,None]).T@X+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: break
        b+=step
        if np.max(np.abs(step))<1e-10: break
    return b

def feats(df,spec,means):
    """Build feature columns (no intercept) for a model spec, centering by given means."""
    c={k:df[k].to_numpy(float)-means[k] for k in ["T_corr","f_mix","f_sun","wind_sc"]}
    cE=df["T_env"].to_numpy(float)-means["T_env"]
    cols=[]
    if spec=="M0": cols=[c["T_corr"]]
    elif spec=="M_env": cols=[cE]
    elif spec=="M_lin": cols=[c["T_corr"],c["f_mix"],c["f_sun"],c["wind_sc"]]
    elif spec=="M_nl": cols=[c["T_corr"],c["f_mix"],c["f_sun"],c["wind_sc"],
                             c["T_corr"]**2,c["T_corr"]*c["f_sun"],
                             c["T_corr"]*c["wind_sc"],c["f_sun"]*c["wind_sc"]]
    return np.column_stack([np.ones(len(df))]+cols)

def auc(y,p):
    o=np.argsort(p); r=np.empty(len(p)); r[o]=np.arange(1,len(p)+1)
    n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)

d=pd.read_csv(D).dropna(subset=["state3","T_corr","f_mix","f_sun","wind_sc","T_env","walk"]).copy()
d=d[d.state3.isin(["comfortable","uncomfortable"])].copy()
d["y"]=(d.state3=="uncomfortable").astype(float)

walks=np.array(sorted(d.walk.unique())); RNG.shuffle(walks)
K=8; folds=np.array_split(walks,K)
SPECS=["M0","M_env","M_lin","M_nl"]
ll={s:[] for s in SPECS}; ag={s:[] for s in SPECS}
for f in folds:
    te=d[d.walk.isin(f)]; tr=d[~d.walk.isin(f)]
    means={k:tr[k].mean() for k in ["T_corr","f_mix","f_sun","wind_sc","T_env"]}
    for s in SPECS:
        Xtr=feats(tr,s,means); Xte=feats(te,s,means)
        b=irls(Xtr,tr.y.to_numpy(float))
        p=np.clip(1/(1+np.exp(-np.clip(Xte@b,-30,30))),1e-6,1-1e-6)
        yte=te.y.to_numpy(float)
        ll[s].append(-np.mean(yte*np.log(p)+(1-yte)*np.log(1-p)))
        ag[s].append(auc(yte,p))
print("walk-grouped %d-fold CV (lower log-loss = better; higher AUC = better)\n"%K)
print(f"{'model':7s} {'CV log-loss':>14s} {'CV AUC':>10s}   params")
P={"M0":1,"M_env":1,"M_lin":4,"M_nl":8}
for s in SPECS:
    print(f"{s:7s} {np.mean(ll[s]):14.4f} {np.mean(ag[s]):10.3f}   {P[s]} pred")
print("\nDelta log-loss vs M_lin (positive = WORSE than linear additive):")
base=np.mean(ll["M_lin"])
for s in SPECS: print(f"  {s:7s} {np.mean(ll[s])-base:+.4f}")
