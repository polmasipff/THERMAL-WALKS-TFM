"""F_forcing_threshold -- the comfort<->discomfort threshold T(m=0) shifts with sun and wind,
and the shift VANISHES in T_env. Same method as F_heterogeneity but stratified by forcing:
logistic [U vs C] ~ coordinate, threshold = -b0/b1, walk-stratified bootstrap CI.
LEFT: threshold in T_corr -> sun lowers it, wind raises it (the forcings are visible).
RIGHT: threshold in T_env -> strata collapse to one value (T_env absorbs sun & wind).
Data: votes_with_Tenv_comfort.csv (C/U votes). Paper-style."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__)); DEG=r"$^{\circ}$C"
D=os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")
RNG=np.random.default_rng(3); NB=1500
d=pd.read_csv(D).dropna(subset=["state3","T_corr","T_env","f_sun","f_mix","wind_sc","walk"]).copy()
d=d[d.state3.isin(["comfortable","uncomfortable"])].copy()
d["y"]=(d.state3=="uncomfortable").astype(float)
d["solar"]=0.5*d.f_mix+d.f_sun

def thr(sub,col,iters=40):
    if len(sub)<40 or sub.y.nunique()<2: return np.nan
    x=sub[col].to_numpy(float); y=sub.y.to_numpy(float); X=np.column_stack([np.ones_like(x),x]); b=np.zeros(2)
    for _ in range(iters):
        p=1/(1+np.exp(-np.clip(X@b,-30,30))); W=np.maximum(p*(1-p),1e-9)
        try: s=np.linalg.solve((X*W[:,None]).T@X+1e-8*np.eye(2),X.T@(y-p))
        except np.linalg.LinAlgError: return np.nan
        b+=s
        if np.max(np.abs(s))<1e-9: break
    return -b[0]/b[1] if b[1]>0 else np.nan

def boot(sub,col):
    wl=sub.walk.unique(); wi={w:sub.index[sub.walk==w].to_numpy() for w in wl}; out=[]
    for _ in range(NB):
        r=sub.loc[np.concatenate([wi[w] for w in RNG.choice(wl,len(wl),replace=True)])]
        t=thr(r,col)
        if not np.isnan(t): out.append(t)
    return (np.median(out),np.percentile(out,2.5),np.percentile(out,97.5)) if out else (np.nan,)*3

strata=[("All votes",         d,                       "ref"),
        ("Shade",             d[d.solar<0.30],         "sun"),
        ("Sun",               d[d.solar>=0.46],        "sun"),
        ("Calm",              d[d.wind_sc<=0.29],      "wind"),
        ("Windy",             d[d.wind_sc>=0.44],      "wind")]
res={}
for col in ["T_corr","T_env"]:
    res[col]=[(lab,*boot(sub,col),len(sub),kind) for lab,sub,kind in strata]
    print(f"\n=== threshold T(m=0) in {col} ===")
    for lab,m,lo,hi,n,k in res[col]: print(f"  {lab:10s} {m:6.2f} [{lo:6.2f},{hi:6.2f}]  n={n}")

COL={"ref":"#152F61","sun":"#E8902E","wind":"#2C7AB5"}
fig,(axL,axR)=plt.subplots(1,2,figsize=(17*CM,6.6*CM),sharey=True)
fig.subplots_adjust(wspace=0.08,left=0.11,right=0.985,bottom=0.17,top=0.88)
y=np.arange(len(strata))[::-1]
for ax,col,title,xlim in [(axL,"T_corr","threshold in $T_{\\rm corr}$ (forcings visible)",(23.5,34.5)),
                          (axR,"T_env","threshold in $T_{\\rm env}$ (forcings absorbed)",(23.5,30.5))]:
    x0,x1=xlim; gmed=res[col][0][1]
    ax.axvline(gmed,color="0.6",ls="--",lw=0.8,zorder=0)
    for yi,(lab,m,lo,hi,n,k) in zip(y,res[col]):
        if np.isnan(m): continue
        loc,hic=max(lo,x0),min(hi,x1)
        ax.plot([loc,hic],[yi,yi],color=COL[k],lw=2.2,solid_capstyle="round",zorder=2)
        if hi>x1: ax.plot([x1],[yi],marker=">",color=COL[k],ms=4,zorder=3)
        if lo<x0: ax.plot([x0],[yi],marker="<",color=COL[k],ms=4,zorder=3)
        ax.scatter([m],[yi],color=COL[k],s=40,zorder=4,edgecolor="white",linewidth=0.5)
        ax.text(x1-0.1,yi+0.30,"n=%d"%n,va="bottom",ha="right",fontsize=5.5,color="0.55")
    sp=max(r[1] for r in res[col][1:])-min(r[1] for r in res[col][1:])
    ax.text(0.03,0.05,"strata spread\n%.1f"%sp+DEG,transform=ax.transAxes,fontsize=8,
            color="0.2",va="bottom",ha="left",
            bbox=dict(boxstyle="round,pad=0.3",fc="0.95",ec="0.7",lw=0.5))
    ax.set_xlim(*xlim); ax.set_xlabel(r"$T(m{=}0)$  ("+DEG+")")
    ax.set_title(title,fontsize=9)
axL.set_yticks(y); axL.set_yticklabels([s[0] for s in strata])
from matplotlib.lines import Line2D
axL.legend(handles=[Line2D([0],[0],color="#E8902E",lw=2.2,label="sun strata"),
                    Line2D([0],[0],color="#2C7AB5",lw=2.2,label="wind strata")],
           frameon=False,fontsize=7,loc="lower right")
save(fig,"F_forcing_threshold")
# quantify the collapse
def spread(col): 
    vals=[r[1] for r in res[col][1:]]; return max(vals)-min(vals)
print("\nthreshold spread across strata: T_corr=%.2f C  ->  T_env=%.2f C"%(spread("T_corr"),spread("T_env")))
