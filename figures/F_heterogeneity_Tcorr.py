"""F_heterogeneity_Tcorr -- minimal body version: PERSONAL threshold heterogeneity in T_corr
(the Blume-Capel field coordinate). Logistic [U vs C] ~ T_corr, threshold -b0/b1,
walk-stratified bootstrap. T_env robustness check: F_heterogeneity_Tenv / _sidebyside."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__)); DEG=r"$^{\circ}$C"; C="T_corr"
B=os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")
RNG=np.random.default_rng(2); NB=2000
d=pd.read_csv(B).dropna(subset=["comfort3",C,"walk_id"]).copy()
d=d[d.comfort3.isin(["comfortable","uncomfortable"])].copy(); d["y"]=(d.comfort3=="uncomfortable").astype(float)
YOUNG={"Less than 10","10-12","13-15"}; ELD={"65-74","75-84"}
d["agebin"]=d.age.map(lambda a:"le15" if a in YOUNG else ("ge16" if pd.notna(a) and a not in ELD else np.nan))
def fit(sub):
    if len(sub)<40 or sub.y.nunique()<2: return np.nan
    x=sub[C].to_numpy(float); y=sub.y.to_numpy(float); X=np.column_stack([np.ones_like(x),x]); b=np.zeros(2)
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(X@b,-30,30))); W=np.maximum(p*(1-p),1e-9)
        try: st=np.linalg.solve((X*W[:,None]).T@X+1e-8*np.eye(2),X.T@(y-p))
        except: return np.nan
        b+=st
        if np.max(np.abs(st))<1e-9: break
    return -b[0]/b[1] if b[1]>0 else np.nan
def boot(sub):
    wl=sub.walk_id.unique(); wi={w:sub.index[sub.walk_id==w].to_numpy() for w in wl}; o=[]
    for _ in range(NB):
        t=fit(sub.loc[np.concatenate([wi[w] for w in RNG.choice(wl,len(wl),replace=True)])])
        if not np.isnan(t): o.append(t)
    return (np.median(o),np.percentile(o,2.5),np.percentile(o,97.5)) if o else (np.nan,)*3
g=d[d.gender.isin(["Man","Woman"])]
rows=[("All votes",d,"#152F61"),("Women",g[g.gender=="Woman"],"#d62728"),("Men",g[g.gender=="Man"],"#2C7AB5"),
      (r"Age $\leq$15",d[d.agebin=="le15"],"#E8902E"),(r"Age $\geq$16",d[d.agebin=="ge16"],"#E8902E")]
res=[(lab,*boot(sub),len(sub),col) for lab,sub,col in rows]; gmed=fit(d)
fig,ax=plt.subplots(figsize=(10*CM,6*CM)); y=np.arange(len(res))[::-1]
ax.axvline(gmed,color="0.6",ls="--",lw=0.9,zorder=0)
ax.text(gmed,len(res)-0.45,"pooled %.1f"%gmed+DEG,color="0.45",fontsize=7,va="bottom",ha="center")
for yi,(lab,m,lo,hi,nn,col) in zip(y,res):
    if np.isnan(m): continue
    ax.plot([lo,hi],[yi,yi],color=col,lw=2.2,solid_capstyle="round",zorder=2)
    ax.scatter([m],[yi],color=col,s=40,zorder=4,edgecolor="white",linewidth=0.5)
    ax.text(hi+0.12,yi,"n=%d"%nn,va="center",ha="left",fontsize=6.5,color="0.5")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in res])
ax.set_xlabel(r"comfort$\leftrightarrow$discomfort threshold  $T(m{=}0)$  in $T_{\rm corr}$  ("+DEG+")")
ax.set_xlim(24,34)
save(fig,"F_heterogeneity_Tcorr"); print("done Tcorr")
