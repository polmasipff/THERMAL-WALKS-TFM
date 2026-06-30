"""F_bc_alpha_heff -- two regime-coordinate figures on T_corr (C/N/U), tied to the Chapter-IV
T_corr regime cuts (26.10, 30.77 C). Left: mixture weight alpha(T), the convex weight that writes
the local composition as alpha*p_cold + (1-alpha)*p_hot, with p_cold/p_hot the pure compositions
of the cold/hot Chapter-IV regimes. Right: effective field h(T)=1/2 ln[P(U)/P(C)] (increases with
heat; sign is a convention). Walk-stratified bootstrap (resample whole walks). Paper-style:
no title/grid, legend without box, large axis labels, formulas live in the text."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import optimize
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")
CUT_LO, CUT_HI = 26.10, 30.77      # Chapter-IV T_corr regime breakpoints
DEG=r"$^{\circ}$C"; LAB={"comfortable":0,"neutral":1,"uncomfortable":2}
d=pd.read_csv(DATA).dropna(subset=["state3","T_corr","walk"]).copy()
d["s"]=d.state3.map(LAB)
EDGES=np.array([22,24,25.5,26.5,27.5,28.5,29.5,30.5,31.5,33.5]); MIDS=(EDGES[:-1]+EDGES[1:])/2

def comp(sub):  # 3-vector composition
    return np.array([(sub.s==k).mean() for k in range(3)])
def fit_alpha(p_obs,pc,ph):
    r=optimize.minimize_scalar(lambda a:np.sum((p_obs-(a*pc+(1-a)*ph))**2),bounds=(0,1),method="bounded")
    return r.x
def curves(data):
    pc=comp(data[data.T_corr<CUT_LO]); ph=comp(data[data.T_corr>CUT_HI])
    a=np.full(len(MIDS),np.nan); h=np.full(len(MIDS),np.nan)
    b=pd.cut(data.T_corr,EDGES,labels=False)
    for i in range(len(MIDS)):
        sub=data[b==i]
        if len(sub)<12: continue
        po=comp(sub); a[i]=fit_alpha(po,pc,ph)
        pC,pU=po[0],po[2]
        if pC>0 and pU>0: h[i]=0.5*np.log(pU/pC)
    return a,h
a_obs,h_obs=curves(d)
# walk-stratified bootstrap
walks=d.walk.unique(); widx={w:d.index[d.walk==w].to_numpy() for w in walks}
rng=np.random.default_rng(0); A=[];H=[]
for _ in range(600):
    samp=d.loc[np.concatenate([widx[w] for w in rng.choice(walks,len(walks),replace=True)])]
    aa,hh=curves(samp); A.append(aa); H.append(hh)
A=np.array(A);H=np.array(H)
a_lo,a_hi=np.nanpercentile(A,2.5,0),np.nanpercentile(A,97.5,0)
h_lo,h_hi=np.nanpercentile(H,2.5,0),np.nanpercentile(H,97.5,0)
# alpha=0.5 crossing and h=0 crossing
ok=~np.isnan(a_obs); aT=np.interp(0.5,a_obs[ok][::-1],MIDS[ok][::-1])
okh=~np.isnan(h_obs); mh,ch=np.polyfit(MIDS[okh],h_obs[okh],1); hT=-ch/mh

PURPLE="#6a3d9a"
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17.4*CM,7.8*CM))
fig.subplots_adjust(wspace=0.28,left=0.10,right=0.985,bottom=0.16,top=0.97)
def regimes(ax):
    ax.axvline(CUT_LO,color="0.55",ls="--",lw=1.0,zorder=1)
    ax.axvline(CUT_HI,color="0.55",ls="--",lw=1.0,zorder=1)
    ax.axvspan(CUT_LO,CUT_HI,color="0.94",zorder=0)
# alpha panel
regimes(ax1)
ax1.fill_between(MIDS[ok],a_lo[ok],a_hi[ok],color=PURPLE,alpha=0.16,zorder=2)
ax1.plot(MIDS[ok],a_obs[ok],"o-",color=PURPLE,ms=5,lw=1.8,zorder=4)
ax1.axhline(0.5,color="0.45",ls=":",lw=1.0,zorder=3)
ax1.axvline(aT,color="#d62728",ls="-.",lw=1.0,zorder=3)
ax1.text(aT-0.2,0.04,r"%.1f%s"%(aT,DEG),fontsize=8.5,color="#d62728",ha="right")
ax1.text(23.7,0.40,"cold\nregime",fontsize=8,color="0.35",ha="center")
ax1.text(32.1,0.62,"hot\nregime",fontsize=8,color="0.35",ha="center")
ax1.set_ylabel("cold-regime fraction")
ax1.set_xlabel(r"corrected air temperature  ("+DEG+")")
ax1.set_ylim(-0.05,1.08); ax1.set_xlim(22.5,33)
# h_eff panel
regimes(ax2)
xx=np.linspace(MIDS[okh].min(),MIDS[okh].max(),50)
ax2.fill_between(MIDS[okh],h_lo[okh],h_hi[okh],color=PURPLE,alpha=0.16,zorder=2)
ax2.plot(xx,mh*xx+ch,color=PURPLE,lw=1.2,ls=(0,(4,2)),zorder=3)
ax2.plot(MIDS[okh],h_obs[okh],"o",color=PURPLE,ms=5,zorder=4)
ax2.axhline(0,color="0.45",ls=":",lw=1.0,zorder=3)
ax2.axvline(hT,color="#d62728",ls="-.",lw=1.0,zorder=3)
ax2.text(hT+0.15,h_lo[okh].min()*0.85,r"$h=0$ at %.1f%s"%(hT,DEG),fontsize=8.5,color="#d62728")
ax2.text(23.4,h_obs[okh].max()*0.7,"comfort\nfavoured",fontsize=8,color="0.35")
ax2.text(31.2,h_obs[okh].min()*0.7,"discomfort\nfavoured",fontsize=8,color="0.35")
ax2.set_ylabel("comfort\u2013discomfort field")
ax2.set_xlabel(r"corrected air temperature  ("+DEG+")"); ax2.set_xlim(22.5,33)
save(fig,"F_bc_alpha_heff")
print("alpha=0.5 at %.2f | h=0 at %.2f | slope %.3f"%(aT,hT,mh))
