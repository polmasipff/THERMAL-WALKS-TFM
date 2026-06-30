"""F_field_additivity -- the comfort field is a single additive scalar coordinate.
Left: T_env weights recovered INDEPENDENTLY as ratios of per-vote logistic coefficients
(gamma_x/gamma_Tcorr), with walk-stratified bootstrap 95% CI, vs the grid-calibrated values.
Right: walk-grouped CV AUC -- the additive index (M_env, 1 predictor) captures all the signal;
freeing the weights or adding nonlinear/cross terms does not help. Numbers from
analysis_field_additivity.py and analysis_field_cv.py. Paper-style."""
import os, numpy as np, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
DEG=r"$^{\circ}$C"; PURPLE="#6a3d9a"; RED="#d62728"

# recovered weights: (label, k_hat, lo, hi, calibrated)
W=[("$k_{\\rm mix}$",  2.83,-0.19, 6.09, 3.0),
   ("$k_{\\rm sun}$", 10.48, 7.80,15.71, 9.5),
   ("$k_{\\rm wind}$",17.14,10.51,26.22,16.0)]
# CV: (model, AUC, nparams)
CV=[("$T_{\\rm corr}$\nonly",0.622,1),("additive\nindex $T_{\\rm env}$",0.725,1),
    ("free\nlinear",0.717,4),("+ nonlinear\n/ cross",0.716,8)]

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17*CM,7.2*CM),gridspec_kw={"width_ratios":[1.15,1]})
fig.subplots_adjust(wspace=0.32,left=0.10,right=0.985,bottom=0.17,top=0.90)

# Left: recovered vs calibrated weights
y=np.arange(len(W))[::-1]
for yi,(lab,k,lo,hi,cal) in zip(y,W):
    ax1.plot([lo,hi],[yi,yi],color=PURPLE,lw=2.2,solid_capstyle="round",zorder=2)
    ax1.scatter([k],[yi],color=PURPLE,s=42,zorder=4,edgecolor="white",linewidth=0.6,
                label="recovered (logistic ratio)" if yi==y[0] else None)
    ax1.scatter([cal],[yi],color=RED,marker="D",s=34,zorder=5,
                label="grid-calibrated" if yi==y[0] else None)
ax1.set_yticks(y); ax1.set_yticklabels([w[0] for w in W])
ax1.set_xlabel(r"weight  ("+DEG+" per unit factor)")
ax1.set_xlim(-2,28)
ax1.legend(frameon=False,loc="lower right",fontsize=8.5)
ax1.set_title("weights recovered independently",fontsize=9.5)

# Right: CV AUC
x=np.arange(len(CV))
cols=["0.6",PURPLE,"#9c7fc0","#c2b0d8"]
ax2.bar(x,[c[1] for c in CV],color=cols,width=0.66,zorder=3,edgecolor="white")
ax2.axhline(0.725,color=PURPLE,ls=":",lw=1.0,zorder=2)
for xi,(lab,a,npar) in zip(x,CV):
    ax2.text(xi,a+0.004,"%.3f"%a,ha="center",fontsize=8,color="0.2")
    ax2.text(xi,0.512,"%d par"%npar,ha="center",fontsize=7,color="0.4")
ax2.set_xticks(x); ax2.set_xticklabels([c[0] for c in CV],fontsize=8)
ax2.set_ylabel("walk-grouped CV AUC"); ax2.set_ylim(0.50,0.76)
ax2.set_title("one additive index suffices",fontsize=9.5)
save(fig,"F_field_additivity")
print("done")
