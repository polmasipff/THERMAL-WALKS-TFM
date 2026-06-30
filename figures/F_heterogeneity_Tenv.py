"""F_heterogeneity_Tenv -- threshold T(m=0) measured in T_env (sun/wind already absorbed), so
only NON-environmental structure should remain. Three blocks:
  (1) environmental check (Shade/Sun/Calm/Windy): if T_env truly absorbs the forcing these
      collapse onto the pooled line -- a within-figure proof we 'ate' sun and wind;
  (2) personal (Women/Men, Age<=15/>=16): the microscopic face of the Blume-Capel beta^-1;
  (3) place (city/neighbourhood): residual spatial inequality (confounded with walk/day).
Method: logistic [U vs C] ~ T_env, threshold -b0/b1, walk-stratified bootstrap CI.
Data: markovian_analysis_baseline.csv. Paper-style."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__)); DEG=r"$^{\circ}$C"
B=os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")
RNG=np.random.default_rng(2); NB=2000; C="T_env"
d=pd.read_csv(B).copy()
d["city"]=d["city"].replace({"L'Hospitalet de Llobreat":"L'Hospitalet de Llobregat"})
d=d.dropna(subset=["comfort3",C,"walk_id"]).copy()
d=d[d.comfort3.isin(["comfortable","uncomfortable"])].copy(); d["y"]=(d.comfort3=="uncomfortable").astype(float)
YOUNG={"Less than 10","10-12","13-15"}; ELD={"65-74","75-84"}
d["agebin"]=d.age.map(lambda a:"le15" if a in YOUNG else ("ge16" if pd.notna(a) and a not in ELD else np.nan))
CITYAB={"L'Hospitalet de Llobregat":"L'Hospitalet","Montcada i Reixac":"Montcada",
        "Barri Sant Pere / La Ribera - Barcelona":"Sant Pere","Barri Congrés / Els Indians  - Barcelona":"Congrés",
        "Sant Vicenç dels Horts":"Sant Vicenç"}

def fit(sub):
    if len(sub)<40 or sub.y.nunique()<2: return np.nan
    x=sub[C].to_numpy(float); y=sub.y.to_numpy(float); X=np.column_stack([np.ones_like(x),x]); b=np.zeros(2)
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(X@b,-30,30))); W=np.maximum(p*(1-p),1e-9)
        try: s=np.linalg.solve((X*W[:,None]).T@X+1e-8*np.eye(2),X.T@(y-p))
        except: return np.nan
        b+=s
        if np.max(np.abs(s))<1e-9: break
    return -b[0]/b[1] if b[1]>0 else np.nan
def boot(sub):
    wl=sub.walk_id.unique(); wi={w:sub.index[sub.walk_id==w].to_numpy() for w in wl}; o=[]
    for _ in range(NB):
        t=fit(sub.loc[np.concatenate([wi[w] for w in RNG.choice(wl,len(wl),replace=True)])])
        if not np.isnan(t): o.append(t)
    return (np.median(o),np.percentile(o,2.5),np.percentile(o,97.5)) if o else (np.nan,np.nan,np.nan)

g=d[d.gender.isin(["Man","Woman"])]
items=[("All votes",d,"ref")]
items+=[("ENV","header",None)]
items+=[("Shade",d[d.sun_s==0],"env"),("Sun",d[d.sun_s>=0.5],"env"),
        ("Calm",d[d.wind_s<=0.25],"env"),("Windy",d[d.wind_s>=0.5],"env")]
items+=[("PERSONAL","header",None)]
items+=[("Women",g[g.gender=="Woman"],"per"),("Men",g[g.gender=="Man"],"per"),
        (r"Age $\leq$15",d[d.agebin=="le15"],"per"),(r"Age $\geq$16",d[d.agebin=="ge16"],"per")]
items+=[("PLACE","header",None)]
for full,ab in sorted(CITYAB.items(),key=lambda kv: -len(d[d.city==kv[0]])):
    sub=d[d.city==full]; frag=" *" if sub.walk_id.nunique()<4 else ""
    items.append((ab+frag,sub,"place"))

gmed=fit(d); rows=[]
for lab,sub,kind in items:
    if isinstance(sub,str): rows.append((lab,None,None,None,None,"header")); continue
    m,lo,hi=boot(sub); rows.append((lab,m,lo,hi,len(sub),kind))
COL={"ref":"#152F61","env":"#2C9E8F","per":"#E8902E","place":"#7B4DA0"}
HEAD={"ENV":"environmental check  (should collapse → forcing absorbed)",
      "PERSONAL":"personal  ($\\beta^{-1}$ heterogeneity)","PLACE":"place  (confounded with walk)"}
n=len(rows); fig,ax=plt.subplots(figsize=(12*CM,12.5*CM)); y=np.arange(n)[::-1]
ax.axvline(gmed,color="0.6",ls="--",lw=0.9,zorder=0)
ax.text(gmed,n-0.2,"pooled %.1f"%gmed+DEG,color="0.45",fontsize=6.8,ha="center",va="bottom")
for yi,(lab,m,lo,hi,nn,kind) in zip(y,rows):
    if kind=="header":
        ax.text(20.05,yi,HEAD[lab],fontsize=7,fontstyle="italic",color="0.35",va="center",ha="left")
        continue
    if np.isnan(m): continue
    lo2,hi2=max(lo,20.2),min(hi,33.8)
    ax.plot([lo2,hi2],[yi,yi],color=COL[kind],lw=2.1,solid_capstyle="round",zorder=2)
    if hi>33.8: ax.plot([33.8],[yi],marker=">",color=COL[kind],ms=3.6)
    ax.scatter([m],[yi],color=COL[kind],s=33,zorder=4,edgecolor="white",linewidth=0.5)
    ax.text(hi2+0.15,yi,"n=%d"%nn,va="center",ha="left",fontsize=5.6,color="0.55")
ax.set_yticks([yi for yi,(l,m,lo,hi,nn,k) in zip(y,rows) if k!="header"])
ax.set_yticklabels([l for (l,m,lo,hi,nn,k) in rows if k!="header"],fontsize=7)
ax.set_xlabel(r"comfort$\leftrightarrow$discomfort threshold  $T(m{=}0)$  in $T_{\rm env}$  ("+DEG+")")
ax.set_xlim(20,34); ax.set_ylim(-0.7,n-0.3)
ax.text(0.99,0.01,"* <4 walks: fragile",transform=ax.transAxes,fontsize=5.5,color="0.5",ha="right",va="bottom")
save(fig,"F_heterogeneity_Tenv")
print("pooled T_env=%.2f"%gmed)
for lab,m,lo,hi,nn,k in rows:
    if k!="header" and not np.isnan(m): print(f"  {lab:14s}[{k}] {m:5.1f} [{lo:5.1f},{hi:5.1f}] n={nn}")
