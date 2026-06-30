"""F3 -- within-walk thermal contrast a coordinate delivers, in standardised units.
Per walk, the range (max-min between stops) of each coordinate, divided by that coordinate's global SD,
so it is comparable to the discrimination floor of Fig 3.3 (which is also in SD). The floor (~0.5 SD,
where eps^2 first exceeds the permutation null) is the minimum contrast for the comfort states to
separate at all. T_corr and especially HDX deliver contrast only around the floor -- HDX least of all,
despite its wide global spread, because almost all of its variance is BETWEEN walks (a regime variable).
Only T_env (median 1.5 SD) reaches well into the separating regime. (median °C ranges annotated.)"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from _common import set_style, save, CM, COL
set_style()
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")).dropna(subset=['T_corr','HDX_corr','T_env','walk'])
coords=[("T_corr","T_corr",COL["T_corr"],r"$T_{\rm corr}$"),
        ("HDX_corr","HDX",COL["HDX"],"HDX"),
        ("T_env","T_env",COL["T_env"],r"$T_{\rm env}$")]
FLOOR=1.0  # SD, from Fig 3.3 (eps^2 robustly above null only for Delta>~1 SD)
fig,ax=plt.subplots(figsize=(8.8*CM,6.6*CM))
rng=np.random.default_rng(0)
data=[]
for i,(c,nm,col,disp) in enumerate(coords):
    sd=d[c].std()
    r=d.groupby('walk')[c].apply(lambda s:s.max()-s.min())
    rsd=(r/sd).values; med_c=r.median()
    data.append(rsd)
    bp=ax.boxplot(rsd,positions=[i],widths=0.55,patch_artist=True,showfliers=False,zorder=2)
    for b in bp["boxes"]: b.set(facecolor=col,alpha=0.25,edgecolor=col,lw=1.0)
    for k in ("whiskers","caps","medians"):
        for ln in bp[k]: ln.set(color=col,lw=1.0)
    ax.scatter(i+rng.uniform(-0.13,0.13,len(rsd)),rsd,s=7,color=col,alpha=0.55,zorder=3,edgecolor="none")
    ax.text(i,-0.18,"med %.1f°C\n=%.2f SD"%(med_c,np.median(rsd)),ha="center",fontsize=5.2,color=col)
ax.axhspan(0,FLOOR,color="0.88",zorder=0)
ax.axhline(FLOOR,color="0.5",ls="--",lw=0.9)
ax.set_xticks(range(3)); ax.set_xticklabels([c[3] for c in coords])
ax.set_ylabel("within-walk range  (SD units of each coordinate)")
ax.set_xlim(-0.5,2.5); ax.set_ylim(-0.35,3.2)
ax.set_title("within-walk thermal contrast vs the discrimination floor",fontsize=7.3)
save(fig,"F3_intrawalk_vs_threshold")
print("median range/SD:",{c[1]:round(np.median(v),2) for c,v in zip(coords,data)})
