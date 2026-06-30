"""F_recovery_onset -- the behavioural change in the comfort kinetics. Heat does not push people
out of comfort faster; it closes the RECOVERY channel out of discomfort. (A) discomfort boundary:
recovery R=P(exit U | in U) collapses with T_corr (elbow ~27-28C) while onset O=P(enter U | not U)
stays flat; recovery decline p=0.0003 (walk permutation), robust to controls. (B) extreme boundary:
onset into very-uncomfortable rises with T. 1.5C bins, walk-stratified bootstrap bands. Dashed line:
static crossover m=0 at ~29C. Paper-style. Data: markovian_analysis_baseline.csv."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__)); DEG=r"$^{\circ}$C"
B=os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")
d=pd.read_csv(B).dropna(subset=['ID','walk_id','stop_idx','comfort7','T_corr']).copy(); d['stop_idx']=d.stop_idx.astype(int)
order=['Very comfortable','Comfortable','Slightly comfortable','Neutral','Slightly uncomfortable','Uncomfortable','Very uncomfortable']
rk={c:i for i,c in enumerate(order)}; d['r7']=d.comfort7.map(rk)
rows=[]
for (w,i),g in d.sort_values('stop_idx').groupby(['walk_id','ID']):
    g=g.reset_index(drop=True)
    for a in range(len(g)-1):
        if g.stop_idx[a+1]-g.stop_idx[a]==1: rows.append((w,g.r7[a],g.r7[a+1],g.T_corr[a]))
T=pd.DataFrame(rows,columns=['walk','ri','rj','Tc'])
walks=T.walk.unique(); wi={w:T.index[T.walk==w].to_numpy() for w in walks}; RNG=np.random.default_rng(0)
EDG=np.arange(22.5,33.01,1.5); MID=(EDG[:-1]+EDG[1:])/2
def curve(data,from_mask,to_event):
    out=np.full(len(MID),np.nan)
    for k,(lo,hi) in enumerate(zip(EDG[:-1],EDG[1:])):
        sub=data[(data.Tc>=lo)&(data.Tc<hi)&from_mask(data)]
        if len(sub)>=12: out[k]=to_event(sub).mean()
    return out
def band(from_mask,to_event):
    BS=[]
    for _ in range(800):
        bs=T.loc[np.concatenate([wi[w] for w in RNG.choice(walks,len(walks),replace=True)])]
        BS.append(curve(bs,from_mask,to_event))
    BS=np.array(BS); return np.nanpercentile(BS,16,0),np.nanpercentile(BS,84,0)
# U boundary
R=curve(T,lambda x:x.ri>=4,lambda s:(s.rj<4).astype(float)); Rlo,Rhi=band(lambda x:x.ri>=4,lambda s:(s.rj<4).astype(float))
O=curve(T,lambda x:x.ri<4,lambda s:(s.rj>=4).astype(float)); Olo,Ohi=band(lambda x:x.ri<4,lambda s:(s.rj>=4).astype(float))
# VU boundary onset
OV=curve(T,lambda x:x.ri<6,lambda s:(s.rj>=6).astype(float)); OVlo,OVhi=band(lambda x:x.ri<6,lambda s:(s.rj>=6).astype(float))

fig,(axA,axB)=plt.subplots(1,2,figsize=(17*CM,6.8*CM))
fig.subplots_adjust(wspace=0.27,left=0.07,right=0.985,bottom=0.16,top=0.9)
GREEN="#2C9E8F"; RED="#d62728"; PUR="#6a3d9a"
for ax in (axA,axB): ax.axvline(29,color="0.55",ls="--",lw=1.0,zorder=1); ax.axvspan(27,28,color="0.92",zorder=0)
m=~np.isnan(R)
axA.fill_between(MID[m],Rlo[m],Rhi[m],color=GREEN,alpha=0.16); axA.plot(MID[m],R[m],"o-",color=GREEN,ms=5,lw=1.9,label="recovery  U$\\to$not-U")
mo=~np.isnan(O); axA.fill_between(MID[mo],Olo[mo],Ohi[mo],color=RED,alpha=0.14); axA.plot(MID[mo],O[mo],"s-",color=RED,ms=4.5,lw=1.7,label="onset  not-U$\\to$U")
axA.text(27.5,0.04,"elbow",fontsize=7,color="0.45",ha="center"); axA.text(29.15,0.83,"$m{=}0$",fontsize=7.5,color="0.4")
axA.set_xlabel(r"corrected air temperature  $T_{\rm corr}$  ("+DEG+")"); axA.set_ylabel("transition probability")
axA.set_ylim(0,0.9); axA.set_xlim(22.5,33); axA.legend(frameon=False,loc="upper right",fontsize=8.5)
axA.set_title("discomfort boundary: recovery closes",fontsize=9)
mv=~np.isnan(OV); axB.fill_between(MID[mv],OVlo[mv],OVhi[mv],color=PUR,alpha=0.16); axB.plot(MID[mv],OV[mv],"o-",color=PUR,ms=5,lw=1.9)
axB.set_xlabel(r"corrected air temperature  $T_{\rm corr}$  ("+DEG+")"); axB.set_ylabel("onset into very-uncomfortable")
axB.set_xlim(22.5,33); axB.set_ylim(0,None); axB.set_title("extreme boundary: onset rises",fontsize=9)
save(fig,"F_recovery_onset")
print("recovery:",np.round(R,2)); print("onset U:",np.round(O,2)); print("onset VU:",np.round(OV,3))
