"""FA (Appendix A) -- candidate 3/4-class partitions scored on the three retained criteria
(ML recoverability, classical separability, class balance); the temperature-separation
criterion is dropped (it privileges a weak local coordinate and was distorting the global score).
Bars: the three component scores. Right markers: P(rank-1) over 20000 random criterion weightings
(Dirichlet) -- comfort3_option2 leads; symmetric C/N/U is the best-balanced, competitive runner-up,
selected as the base state space for the spin-1 mapping (Ch.4), not as the empirical winner."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, save, CM
set_style()
HERE=os.path.dirname(os.path.abspath(__file__))
r=pd.read_csv(os.path.join(HERE,"..","data","candidate_groupings_rank_table.csv"))
r=r[["pretty_label","ml_score","stats_score","balance_score"]].copy()
# Dirichlet P(rank1)
rng=np.random.default_rng(0); W=rng.dirichlet([1,1,1],20000)
M=r[["ml_score","stats_score","balance_score"]].to_numpy()
S=W@M.T; top1=np.bincount((-S).argsort(1)[:,0],minlength=len(r))/len(W)
r["p_top1"]=top1
r["composite"]=M.mean(1)
r=r.sort_values("composite").reset_index(drop=True)
comp=["ml_score","stats_score","balance_score"]; clab=["ML","stats","balance"]
cols=["#1f77b4","#7f7f7f","#2ca02c"]
fig,ax=plt.subplots(figsize=(11*CM,6.2*CM))
y=np.arange(len(r)); h=0.24
for j,c in enumerate(comp):
    ax.barh(y+(j-1)*h, r[c], height=h, color=cols[j], alpha=0.85, label=clab[j])
ax.set_yticks(y); ax.set_yticklabels(r.pretty_label, fontsize=6.5)
ax.set_xlabel("normalised score (within candidate set)"); ax.set_xlim(0,1.18)
for yi,p in zip(y,r.p_top1):
    ax.text(1.02, yi, "P(top)=%.0f%%"%(100*p), va="center", fontsize=5.2, color="0.3")
ax.legend(fontsize=5.6,frameon=False,loc="lower right",ncol=3)
ax.set_title("candidate partitions on the three retained criteria (temp dropped)",fontsize=7)
save(fig,"FA_partition_scorecard")
print(r[["pretty_label","ml_score","stats_score","balance_score","p_top1"]].round(3).to_string(index=False))
