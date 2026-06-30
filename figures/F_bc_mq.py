"""F_bc_mq -- Blume-Capel order parameter m(T) and quadrupolar density q(T). Paper-style:
empirical binned values (the honest data) with walk-stratified bootstrap bands; the logistic fit
is used ONLY to locate the m=0 crossing (thin dashed line), never to smooth q. No title/grid;
legend without frame; large axis labels. Spin-1: S=-1 comfortable, 0 neutral, +1 uncomfortable."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__))
b=pd.read_csv(os.path.join(HERE,"_bc_binned.csv"))
T0=29.07; WIN=(27.1,30.1); DEG=r"$^{\circ}$C"
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11.5*CM,10.5*CM),sharex=True,gridspec_kw={"height_ratios":[1.25,1],"hspace":0.12})
for ax in (ax1,ax2): ax.axvspan(*WIN,color="0.92",zorder=0)
ax1.axhline(0,color="0.55",lw=0.8,ls="--",zorder=1)
ax1.fill_between(b.Tc,b.m_lo,b.m_hi,color="#d62728",alpha=0.18,zorder=2)
ax1.plot(b.Tc,b.m,"o-",color="#d62728",ms=5,lw=1.8,zorder=4)
ax1.axvline(T0,color="0.4",ls=":",lw=1.0,zorder=3)
ax1.text(T0+0.2,-0.30,"$m=0$ at %.0f%s"%(T0,DEG),fontsize=8.5,color="0.3")
ax1.set_ylabel(r"$m = P(U)-P(C)$")
ax1.set_ylim(-0.62,0.62)
qbar=np.average(b.q,weights=1/((b.q_hi-b.q_lo)**2+1e-6))
ax2.fill_between(b.Tc,b.q_lo,b.q_hi,color="#555555",alpha=0.20,zorder=2)
ax2.plot(b.Tc,b.q,"o-",color="0.2",ms=5,lw=1.8,zorder=4)
ax2.axhline(qbar,color="0.45",ls="--",lw=1.0,zorder=3)
ax2.text(24.4,qbar+0.018,r"$q\approx%.2f$"%qbar,fontsize=8.5,color="0.3")
ax2.set_ylabel(r"$q = P(U)+P(C)$"); ax2.set_ylim(0.55,0.95)
ax2.set_xlabel(r"corrected air temperature  $T_{\rm corr}$  ("+DEG+")"); ax2.set_xlim(24,33)
save(fig,"F_bc_mq")
print("qbar=%.3f"%qbar)
