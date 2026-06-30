"""F_bc_fields -- effective Blume-Capel fields from the empirical compositions. Paper-style.
beta*h(T) = 1/2 ln[P(U)/P(C)]  (external field; air temperature sets it, sign change = crossover).
beta*D(T) = -1/2 ln[P(U)P(C)/P(N)^2]  (crystal field; cost of leaving neutral). Only beta*h and
beta*D are identifiable from a single composition; beta is the populational heterogeneity (not separable)."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__))
b=pd.read_csv(os.path.join(HERE,"_bc_binned.csv")); DEG=r"$^{\circ}$C"; WIN=(27.1,30.1)
# linear fit to beta*h to read the zero crossing
m,c=np.polyfit(b.Tc,b.bh,1); T0=-c/m
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11.5*CM,10.5*CM),sharex=True,gridspec_kw={"hspace":0.12})
ax1.axvspan(*WIN,color="0.92",zorder=0); ax2.axvspan(*WIN,color="0.92",zorder=0)
ax1.axhline(0,color="0.55",lw=0.8,ls="--",zorder=1)
ax1.fill_between(b.Tc,b.bh_lo,b.bh_hi,color="#6a3d9a",alpha=0.16,zorder=2)
xx=np.linspace(b.Tc.min(),b.Tc.max(),50)
ax1.plot(xx,m*xx+c,color="#6a3d9a",lw=1.2,ls=(0,(4,2)),zorder=3,label="linear fit")
ax1.plot(b.Tc,b.bh,"o",color="#6a3d9a",ms=5,zorder=4,label="binned estimate")
ax1.axvline(T0,color="0.4",ls=":",lw=1.0,zorder=3)
ax1.text(T0+0.2,b.bh.min()*0.8,r"$\beta h=0$ at %.1f%s"%(T0,DEG),fontsize=8.5,color="0.3")
ax1.set_ylabel(r"$\beta h = \frac{1}{2}\ln\frac{P(U)}{P(C)}$  (field)")
ax1.legend(frameon=False,loc="upper left")
# crystal field
ax2.fill_between(b.Tc,b.bd_lo,b.bd_hi,color="#2c7fb8",alpha=0.16,zorder=2)
ax2.plot(b.Tc,b.bd,"o-",color="#2c7fb8",ms=5,lw=1.6,zorder=4)
ax2.axhline(np.nanmean(b.bd),color="0.45",ls="--",lw=1.0,zorder=3)
ax2.set_ylabel(r"$\beta D = -\frac{1}{2}\ln\frac{P(U)P(C)}{P(N)^2}$")
ax2.set_xlabel(r"corrected air temperature  $T_{\rm corr}$  ("+DEG+")"); ax2.set_xlim(24,33)
save(fig,"F_bc_fields")
print("beta*h zero at %.2f, mean beta*D=%.3f"%(T0,np.nanmean(b.bd)))
