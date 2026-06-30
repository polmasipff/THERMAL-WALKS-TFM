"""F_simplex_trajectory -- alternative bridge figure: the C/N/U composition as a TRAJECTORY in the
2-simplex (the triangle of all possible (P(C),P(N),P(U))), one point per temperature bin, coloured
by T_corr. As temperature rises the population walks from the comfort vertex toward the discomfort
vertex, staying near the lower (low-neutral... ) edge -> directly visualises m sweeping while q
(distance from the N vertex) stays roughly constant. Paper-style."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()
HERE=os.path.dirname(os.path.abspath(__file__)); DEG=r"$^{\circ}$C"
D=os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")
d=pd.read_csv(D).dropna(subset=["state3","T_corr"]).copy()
# vertices: C bottom-left, U bottom-right, N top
VC=np.array([0,0]); VU=np.array([1,0]); VN=np.array([0.5,np.sqrt(3)/2])
def tern(pC,pN,pU): return pC*VC+pN*VN+pU*VU
EDG=np.arange(22,34.01,1.0); MID=(EDG[:-1]+EDG[1:])/2
g=pd.cut(d.T_corr,EDG,right=False); pts=[]; Ts=[]
for iv,sub in d.groupby(g,observed=True):
    if len(sub)<12: continue
    pC=(sub.state3=="comfortable").mean(); pN=(sub.state3=="neutral").mean(); pU=(sub.state3=="uncomfortable").mean()
    pts.append(tern(pC,pN,pU)); Ts.append((iv.left+iv.right)/2)
pts=np.array(pts); Ts=np.array(Ts)
fig,ax=plt.subplots(figsize=(9.5*CM,8.8*CM))
# triangle frame
tri=np.array([VC,VU,VN,VC]); ax.plot(tri[:,0],tri[:,1],color="0.6",lw=1.0)
for v,lab,dx,dy,col in [(VC,"comfort (C)",-0.02,-0.06,"#1f77b4"),(VU,"discomfort (U)",0.02,-0.06,"#d62728"),(VN,"neutral (N)",0,0.04,"#7f7f7f")]:
    ax.text(v[0],v[1]+dy,lab,ha="center",va="top" if dy<0 else "bottom",fontsize=8.5,color=col,fontweight="bold")
# light gridlines (constant q = constant P(N))
for qv in [0.2,0.4,0.6,0.8]:
    a=np.array([tern((1-qv)* (1-t),qv,(1-qv)*t) for t in [0,1]])
    ax.plot(a[:,0],a[:,1],color="0.85",lw=0.6,zorder=0)
# trajectory colored by T
segs=np.concatenate([pts[:-1,None],pts[1:,None]],axis=1)
lc=LineCollection(segs,cmap="inferno",lw=2.4,zorder=3); lc.set_array(Ts[:-1]); ax.add_collection(lc)
sc=ax.scatter(pts[:,0],pts[:,1],c=Ts,cmap="inferno",s=34,zorder=4,edgecolor="white",linewidth=0.5)
cb=fig.colorbar(sc,ax=ax,fraction=0.046,pad=0.02); cb.set_label(r"$T_{\rm corr}$  ("+DEG+")",fontsize=8)
ax.text(0.5,-0.12,"each point = comfort composition at one temperature",transform=ax.transAxes,ha="center",fontsize=7,color="0.4")
ax.set_xlim(-0.08,1.08); ax.set_ylim(-0.16,0.95); ax.set_aspect("equal"); ax.axis("off")
save(fig,"F_simplex_trajectory")
print("done; q range:",round(1-pts[:,1].max()/(np.sqrt(3)/2),2),"to",round(1-pts[:,1].min()/(np.sqrt(3)/2),2))
