"""F2b -- separability scale on standardised coordinates, with walk-stratified bootstrap CIs.
eps^2 of the 3-state separation vs window width Delta (in each coordinate's SD units). Solid line =
point estimate (median over windows); shaded band = walk-stratified bootstrap 16-84%; grey dashed =
permutation 95% null. The separation only becomes robustly non-null (bootstrap CI clear of the null)
around Delta ~ 1 SD, and even then eps^2 stays small in absolute terms (a few x10^-2): comfort is a
noisy response that no coordinate resolves strongly at the local scale. What the figure shows is the
COMPARISON -- T_env's local separability is ~3-4x that of T_corr and HDX at every scale.
Source: _f2b_data.csv (built by the bootstrap computation)."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, save, CM, COL
set_style()
HERE=os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(HERE,"_f2b_data.csv"))
spec=[("T_corr",COL["T_corr"],r"$T_{\rm corr}$"),("HDX_corr",COL["HDX"],"HDX"),("T_env",COL["T_env"],r"$T_{\rm env}$")]
FLOOR=1.0
fig,ax=plt.subplots(figsize=(9.2*CM,6.8*CM))
ax.axvspan(0,FLOOR,color="0.94",zorder=0)
for col,c,disp in spec:
    s=df[df.coord==col].sort_values("delta")
    ax.fill_between(s.delta,s.lo,s.hi,color=c,alpha=0.16,zorder=1)
    ax.plot(s.delta,s.obs,"-o",color=c,ms=2.6,lw=1.3,label=disp,zorder=3)
# single pooled null band (max over coords ~ same near 0)
s0=df.groupby("delta").null95.max()
ax.plot(s0.index,s0.values,"--",color="0.55",lw=0.8,zorder=2,label="perm. null (95%)")
ax.axvline(FLOOR,color="0.5",ls=":",lw=0.8)
ax.text(FLOOR+0.04,0.066,"separation robust\nabove null only\nfor $\\Delta\\gtrsim1$ SD",fontsize=4.8,color="0.4",va="top")
ax.set_xlabel(r"window width  $\Delta$  (SD units of each coordinate)")
ax.set_ylabel(r"separability  $\varepsilon^2$  (3-state KW)")
ax.set_xlim(0,3.05); ax.set_ylim(-0.002,0.075)
ax.set_title("separability scale, standardised coordinates",fontsize=7.5)
ax.legend(fontsize=5.4,frameon=False,loc="upper left")
save(fig,"F2b_scale_std")
print("done")
