"""F_dynamics -- transition dynamics of the C/N/U comfort state (combined Ch5). Three panels:
(A) aggregate stop-to-stop transition matrix P (row=from, col=to); (B) the kinetic mechanism --
recovery U->C and persistence U->U by T_env regime, with walk-stratified bootstrap 95% CI (the
robust signal: discomfort gets more persistent under heat); (C) Schnakenberg cycle affinity by
regime with bootstrap CI and zero line (detailed balance broken in the central/crossover regime;
cold leans negative; per-regime estimates are noisy). Paper-style. Data: markovian_analysis_baseline.csv."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__))
B=os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")
CUTS=(24.62,30.83); RNG=np.random.default_rng(0); NB=1200
d=pd.read_csv(B).dropna(subset=['ID','walk_id','stop_idx','comfort3','T_env']).copy()
d['stop_idx']=d.stop_idx.astype(int); St={'comfortable':0,'neutral':1,'uncomfortable':2}; d['s']=d.comfort3.map(St)
rows=[]
for (w,i),g in d.sort_values('stop_idx').groupby(['walk_id','ID']):
    g=g.reset_index(drop=True)
    for a in range(len(g)-1):
        if g.stop_idx[a+1]-g.stop_idx[a]==1: rows.append((w,int(g.s[a]),int(g.s[a+1]),g.T_env[a]))
T=pd.DataFrame(rows,columns=['walk','si','sj','Tenv'])
T['reg']=T.Tenv.map(lambda t:0 if t<CUTS[0] else (1 if t<CUTS[1] else 2))
def cm(sub):
    C=np.zeros((3,3)); np.add.at(C,(sub.si.values,sub.sj.values),1); return C
def st(sub):
    C=cm(sub); P=C/np.clip(C.sum(1,keepdims=True),1,None)
    num=P[0,1]*P[1,2]*P[2,0]; den=P[1,0]*P[2,1]*P[0,2]
    A=np.log(num/den) if num>0 and den>0 else np.nan
    return P,P[2,0],P[2,2],A
Pagg=st(T)[0]
walks=T.walk.unique(); wi={w:T.index[T.walk==w].to_numpy() for w in walks}
def boot(reg):
    uc,uu,af=[],[],[]
    for _ in range(NB):
        bs=T.loc[np.concatenate([wi[w] for w in RNG.choice(walks,len(walks),replace=True)])]
        bs=bs[bs.reg==reg]
        if len(bs)<30: continue
        _,a,b,A=st(bs); uc.append(a); uu.append(b)
        if not np.isnan(A): af.append(A)
    q=lambda x:(np.percentile(x,2.5),np.percentile(x,97.5))
    return (np.array(uc),np.array(uu),np.array(af))
REG=["cold","central","hot"]; pts={r:st(T[T.reg==r]) for r in range(3)}; bt={r:boot(r) for r in range(3)}

fig=plt.figure(figsize=(17.5*CM,5.8*CM))
gs=fig.add_gridspec(1,3,width_ratios=[1,1.15,1.0],wspace=0.42,left=0.06,right=0.985,bottom=0.17,top=0.88)
# A: matrix
axA=fig.add_subplot(gs[0]); im=axA.imshow(Pagg,cmap="Purples",vmin=0,vmax=0.6)
for i in range(3):
    for j in range(3): axA.text(j,i,"%.2f"%Pagg[i,j],ha="center",va="center",fontsize=8.5,color="white" if Pagg[i,j]>0.35 else "0.2")
axA.set_xticks(range(3)); axA.set_xticklabels(["C","N","U"]); axA.set_yticks(range(3)); axA.set_yticklabels(["C","N","U"])
axA.set_xlabel("to"); axA.set_ylabel("from"); axA.set_title("aggregate $P_{ij}$",fontsize=9)
# B: channels
axB=fig.add_subplot(gs[1]); x=np.arange(3)
for key,idx,col,lab,dx in [("rec",1,"#2C9E8F","recovery U$\\to$C",-0.07),("per",2,"#d62728","persistence U$\\to$U",0.07)]:
    pv=[pts[r][idx] for r in range(3)]
    lo=[np.percentile(bt[r][0 if idx==1 else 1],2.5) for r in range(3)]
    hi=[np.percentile(bt[r][0 if idx==1 else 1],97.5) for r in range(3)]
    axB.errorbar(x+dx,pv,yerr=[np.array(pv)-lo,np.array(hi)-pv],fmt="o-",color=col,ms=5,lw=1.6,capsize=2.5,label=lab)
axB.set_xticks(x); axB.set_xticklabels(REG); axB.set_ylabel("transition probability"); axB.set_ylim(0.1,0.75)
axB.legend(frameon=False,fontsize=8,loc="upper left"); axB.set_title("kinetic mechanism",fontsize=9)
# C: affinity
axC=fig.add_subplot(gs[2]); axC.axhline(0,color="0.5",lw=0.8,ls="--")
av=[pts[r][3] for r in range(3)]; alo=[np.percentile(bt[r][2],2.5) for r in range(3)]; ahi=[np.percentile(bt[r][2],97.5) for r in range(3)]
axC.errorbar(x,av,yerr=[np.array(av)-alo,np.array(ahi)-av],fmt="o",color="#6a3d9a",ms=6,lw=1.5,capsize=2.5)
axC.set_xticks(x); axC.set_xticklabels(REG); axC.set_ylabel(r"cycle affinity $\mathcal{A}$ (nats)"); axC.set_title("detailed-balance breaking",fontsize=9)
axC.set_xlim(-0.5,2.5)
save(fig,"F_dynamics")
print("P_agg diag:",np.round(np.diag(Pagg),3))
for r in range(3): print(f"  {REG[r]:8s} U->C={pts[r][1]:.2f} U->U={pts[r][2]:.2f} A={pts[r][3]:+.2f}")
