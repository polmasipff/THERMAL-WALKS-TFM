"""F_cap5_affinity -- breaking of detailed balance, resolved by forcing.
For a 3-state cycle C->N->U->C the single Schnakenberg affinity A=ln[(P_CN P_NU P_UC)/(P_NC P_UN P_CU)]
fully characterises detailed-balance breaking: A=0 is equilibrium, A!=0 a non-equilibrium steady
state. (a) A reverses sign cold(-)->hot(+) with walk-stratified bootstrap CIs; (b) the net cycle
current J=pi_C P_CN - pi_N P_NC does the same. The reversal -- recovery-sense circulation in the cold
regime, deterioration-sense in the hot -- is why the aggregate current nearly cancels."""
import os, numpy as np, matplotlib.pyplot as plt
from _common import set_style, save, CM
from _cap5_data import transitions, countmat, affinity
set_style()
T=transitions(); walks=T.walk.unique(); widx={w:T.index[T.walk==w].to_numpy() for w in walks}
regs=[0,1,2]; names=["cold","central","hot"]; cols=["#1f77b4","#7f7f7f","#d62728"]
A0={}; J0={}
for r in regs: A0[r],J0[r],_=affinity(countmat(T[T.reg==r]))
rng=np.random.default_rng(0); BA={r:[] for r in regs}; BJ={r:[] for r in regs}
for _ in range(600):
    ch=rng.choice(walks,len(walks),True); rr=T.loc[np.concatenate([widx[w] for w in ch])]
    for r in regs:
        a,j,_=affinity(countmat(rr[rr.reg==r]))
        if not np.isnan(a): BA[r].append(a); BJ[r].append(j)
fig,(axA,axJ)=plt.subplots(1,2,figsize=(14*CM,6*CM))
x=np.arange(3)
for ax,val,boot,ylab,ttl in [(axA,A0,BA,"cycle affinity $A$  (nats)","(a) detailed-balance breaking"),
                              (axJ,J0,BJ,"cycle current $J$","(b) net cycle current")]:
    ax.axhline(0,color="0.6",lw=0.7,ls="--")
    for i,r in enumerate(regs):
        b=np.array(boot[r]); lo,hi=np.percentile(b,[16,84]); lo2,hi2=np.percentile(b,[2.5,97.5])
        ax.plot([i,i],[lo2,hi2],color=cols[i],lw=1.0,alpha=0.5)
        ax.plot([i,i],[lo,hi],color=cols[i],lw=2.6)
        ax.plot(i,val[r],"o",color=cols[i],ms=6,zorder=5,mec="white",mew=0.6)
    ax.plot(x,[val[r] for r in regs],"-",color="0.5",lw=0.8,zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(names,fontsize=7)
    ax.set_xlabel("$T_{\\rm env}$ regime"); ax.set_ylabel(ylab); ax.set_title(ttl,fontsize=7.5)
    ax.set_xlim(-0.4,2.55)
# sense arrows on the affinity axis (once), and P(sign) below each point
axA.annotate("deterioration sense",xy=(2.28,0.9),xytext=(2.28,0.9),fontsize=4.6,color="#d62728",rotation=90,va="center",ha="left")
axA.annotate("recovery sense",xy=(2.28,-0.9),xytext=(2.28,-0.9),fontsize=4.6,color="#1f77b4",rotation=90,va="center",ha="left")
yb=axA.get_ylim()[0]
for i,r in enumerate(regs):
    b=np.array(BA[r]); frac=max(np.mean(b>0),np.mean(b<0))
    axA.text(i,yb+0.07*(axA.get_ylim()[1]-yb),"sign %.0f%%"%(100*frac),ha="center",fontsize=4.3,color="0.45")
fig.subplots_adjust(wspace=0.3)
save(fig,"F_cap5_affinity")
print("A:",{names[i]:round(A0[r],3) for i,r in enumerate(regs)})
print("J:",{names[i]:round(J0[r],4) for i,r in enumerate(regs)})
