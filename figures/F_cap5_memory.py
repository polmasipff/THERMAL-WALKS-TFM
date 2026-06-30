"""F_cap5_memory -- the very-uncomfortable state is a memory-bearing trap.
On the {CN,U,VU} magnifier, conditioning on two consecutive VU votes raises the probability of a
third far above the first-order (current-state-only) prediction. Compared against a null in which
trajectories are simulated from the fitted first-order chain (same lengths, same starts), the excess
is well outside what memoryless dynamics produce: order-2 P(VU|VU,VU)=0.87 vs order-1 0.45, n=23,
null median 0.21 on the L1 statistic, p~0.002. The bulk of the space is first-order Markovian
(weighted L1=0.12); only the extreme keeps history."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import defaultdict
from _common import set_style, save, CM
set_style()
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")).dropna(subset=['ID','walk_id','stop_idx','comfort3']).copy()
d['stop_idx']=d.stop_idx.astype(int)
vu=(d['is_very_uncomfortable']==1) if 'is_very_uncomfortable' in d else (d['comfort7']=='Very uncomfortable')
d['s4']=np.where(vu,2,np.where(d.comfort3=='uncomfortable',1,0))
seqs=[]
for (w,i),g in d.sort_values('stop_idx').groupby(['walk_id','ID']):
    g=g.reset_index(drop=True); run=[g.s4[0]]
    for a in range(1,len(g)):
        if g.stop_idx[a]-g.stop_idx[a-1]==1: run.append(g.s4[a])
        else: seqs.append(run); run=[g.s4[a]]
    seqs.append(run)
C1=np.zeros((3,3))
for r in seqs:
    for a in range(len(r)-1): C1[r[a],r[a+1]]+=1
P1=C1/np.clip(C1.sum(1,keepdims=True),1,None)
C2=defaultdict(lambda: np.zeros(3))
for r in seqs:
    for a in range(2,len(r)): C2[(r[a-2],r[a-1])][r[a]]+=1
nVU=int(C2[(2,2)].sum()); p2=C2[(2,2)]/nVU
obs_p=p2[2]; ord1_p=P1[2,2]; L1obs=np.abs(p2-P1[2]).sum()
# null: simulate order-1
rng=np.random.default_rng(0); lens=[len(r) for r in seqs]; st=[r[0] for r in seqs]
nullP=[]; nullL1=[]
for _ in range(1500):
    cc=np.zeros(3); 
    for L,s0 in zip(lens,st):
        seq=[s0]
        for _ in range(L-1): seq.append(rng.choice(3,p=P1[seq[-1]]))
        for a in range(2,len(seq)):
            if seq[a-2]==2 and seq[a-1]==2: cc[seq[a]]+=1
    n=cc.sum()
    if n>=1: nullP.append(cc[2]/n); nullL1.append(np.abs(cc/n-P1[2]).sum())
nullP=np.array(nullP); nullL1=np.array(nullL1)
pval=(np.sum(nullL1>=L1obs)+1)/(len(nullL1)+1)

fig,(axB,axN)=plt.subplots(1,2,figsize=(14*CM,6*CM),gridspec_kw={"width_ratios":[1,1.25]})
# (a) bars
axB.bar([0,1],[ord1_p,obs_p],color=["0.6","#d62728"],width=0.6,alpha=0.9)
axB.set_xticks([0,1]); axB.set_xticklabels(["order-1\n$P(VU|VU)$","order-2\n$P(VU|VU,VU)$"],fontsize=6)
for x,v in [(0,ord1_p),(1,obs_p)]: axB.text(x,v+0.02,"%.2f"%v,ha="center",fontsize=7,fontweight="bold")
axB.set_ylabel("probability of remaining VU"); axB.set_ylim(0,1.0)
axB.set_title("(a) persistence of the extreme  (n=%d triplets)"%nVU,fontsize=7)
# (b) null distribution of P(VU|VU,VU)
axN.hist(nullP,bins=20,color="0.8",edgecolor="0.6",lw=0.4,density=True)
axN.axvline(ord1_p,color="0.4",ls="--",lw=1.0,label="order-1 expectation %.2f"%ord1_p)
axN.axvline(obs_p,color="#d62728",lw=1.8,label="observed %.2f"%obs_p)
axN.set_xlabel("$P(VU|VU,VU)$ under first-order null"); axN.set_ylabel("density")
axN.set_title("(b) vs memoryless null   ($p\\approx$%.3f)"%pval,fontsize=7)
axN.legend(fontsize=5,frameon=False,loc="upper center")
fig.subplots_adjust(wspace=0.34)
save(fig,"F_cap5_memory")
print("order1=%.3f order2=%.3f n=%d L1obs=%.3f nullmedL1=%.3f p=%.4f"%(ord1_p,obs_p,nVU,L1obs,np.median(nullL1),pval))
