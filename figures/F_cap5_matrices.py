"""F_cap5_matrices -- stop-to-stop C/N/U transition matrices, aggregate and by T_env regime.
Cells show P(j|i) (count beneath). The recovery channel U->C weakens and the uncomfortable
persistence U->U strengthens from cold to hot -- the kinetic signature of trapping."""
import os, numpy as np, matplotlib.pyplot as plt
from _common import set_style, save, CM
from _cap5_data import transitions, countmat
set_style()
T=transitions(); lab=['C','N','U']
panels=[("aggregate",T)]+[(n,T[T.reg==r]) for r,n in zip([0,1,2],["cold ($T_{\\rm env}<24.6$)","central","hot ($>29.8$)"])]
fig,axes=plt.subplots(1,4,figsize=(16*CM,5*CM))
for ax,(name,sub) in zip(axes,panels):
    C=countmat(sub); P=C/np.clip(C.sum(1,keepdims=True),1,None)
    im=ax.imshow(P,cmap="Blues",vmin=0,vmax=0.62)
    for i in range(3):
        for j in range(3):
            ax.text(j,i-0.12,"%.2f"%P[i,j],ha="center",va="center",fontsize=6,
                    color="white" if P[i,j]>0.4 else "0.2",fontweight="bold")
            ax.text(j,i+0.22,"%d"%C[i,j],ha="center",va="center",fontsize=4.2,
                    color="white" if P[i,j]>0.4 else "0.45")
    # highlight U->C (recovery) and U->U (persistence)
    for (i,j),c in [((2,0),"#2ca02c"),((2,2),"#d62728")]:
        ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor=c,lw=1.4))
    ax.set_xticks(range(3)); ax.set_xticklabels(lab,fontsize=6.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(lab,fontsize=6.5)
    ax.set_title("%s\n(n=%d)"%(name,len(sub)),fontsize=6.3)
    ax.set_xlabel("to",fontsize=5.5)
    if ax is axes[0]: ax.set_ylabel("from",fontsize=5.5)
fig.text(0.5,-0.02,"green = recovery U$\\to$C (weakens with heat)   red = persistence U$\\to$U (strengthens)",ha="center",fontsize=5,color="0.4")
fig.subplots_adjust(wspace=0.32)
save(fig,"F_cap5_matrices")
print("done")
