"""F_ml_importance -- which conditions drive a discomfort vote. Walk-grouped permutation importance
(mean AUC drop when a feature is shuffled in the held-out fold) of an L2 logistic model predicting
P(uncomfortable) from raw environment+context (T_env excluded). Bars coloured by family
(regime / local thermal / urban form / personal). Sign = direction of the univariate rank association
with discomfort. Message: local sun(+) and wind(-) dominate, the walk's temperature regime next;
WITHIN-walk air temperature anomaly is nearly useless -- temperature acts between walks, not within.
Source: markovian_analysis_baseline.csv ; walk-grouped 5-fold CV."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings; warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
from _common import set_style, save, CM
set_style()
HERE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")).dropna(subset=["comfort3","walk_id"]).copy()
FAM={"temp_rel_walk":"local","HDX_rel_walk":"local","sun_s":"local","wind_s":"local",
 "temp_mean_walk":"regime","HDX_mean_walk":"regime",
 "SVF_1.5m":"urban","NDVI_1.5m":"urban","IVAC":"urban","climate_shelter":"urban",
 "distance_fountain":"urban","distance_drinking_fountain":"urban","distance_green_zone":"urban","surface_type":"urban","space_category":"urban",
 "gender":"personal","age":"personal","participants_clothing":"personal","spent_time":"personal","stop_idx":"personal"}
NICE={"sun_s":"sun exposure","wind_s":"wind","temp_mean_walk":"temperature (walk regime)","SVF_1.5m":"sky-view factor",
 "age":"age","HDX_rel_walk":"humidex (local)","gender":"gender","temp_rel_walk":"temperature (local)","NDVI_1.5m":"greenness (NDVI)",
 "IVAC":"vulnerability index","participants_clothing":"clothing","HDX_mean_walk":"humidex (regime)","stop_idx":"stop position",
 "distance_green_zone":"dist. green zone","distance_fountain":"dist. fountain","distance_drinking_fountain":"dist. drinking fountain",
 "climate_shelter":"dist. climate shelter","surface_type":"surface type","space_category":"space category","spent_time":"time outdoors"}
COLF={"local":"#d62728","regime":"#ff7f0e","urban":"#2ca02c","personal":"#1f77b4"}
num=[f for f in FAM if d[f].dtype!=object]; cat=[f for f in FAM if d[f].dtype==object]
feats=num+cat
X=d[feats].reset_index(drop=True); yb=(d.comfort3.to_numpy()=="uncomfortable").astype(int); grp=d.walk_id.to_numpy()
pre=ColumnTransformer([("n",Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler())]),num),
                       ("c",Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]),cat)])
mdl=Pipeline([("p",pre),("m",LogisticRegression(max_iter=3000,class_weight="balanced"))])
gkf=GroupKFold(5); rng=np.random.default_rng(0); imp={f:[] for f in feats}; baseA=[]
for tr,te in gkf.split(X,yb,grp):
    m=mdl.fit(X.iloc[tr],yb[tr]); Xte=X.iloc[te].reset_index(drop=True); yte=yb[te]
    base=roc_auc_score(yte,m.predict_proba(Xte)[:,1]); baseA.append(base)
    for f in feats:
        dd=[]
        for _ in range(6):
            Xp=Xte.copy(); Xp[f]=rng.permutation(Xp[f].values); dd.append(base-roc_auc_score(yte,m.predict_proba(Xp)[:,1]))
        imp[f].append(np.mean(dd))
impm={f:np.mean(v) for f,v in imp.items()}
sgn={f:(spearmanr(d[f],yb).correlation if f in num else np.nan) for f in feats}
ordf=sorted(feats,key=lambda f:impm[f],reverse=True)[:12][::-1]
fig,ax=plt.subplots(figsize=(10*CM,7.4*CM))
yy=np.arange(len(ordf))
for i,f in enumerate(ordf):
    ax.barh(i,impm[f],color=COLF[FAM[f]],alpha=0.9,height=0.7)
    s=sgn[f]
    if not np.isnan(s): ax.text(impm[f]+0.0012,i,"↑" if s>0 else "↓",va="center",fontsize=7,color="0.3")
ax.set_yticks(yy); ax.set_yticklabels([NICE[f] for f in ordf],fontsize=6.3)
ax.set_xlabel("permutation importance  (drop in discomfort AUC)")
ax.set_title("what drives a discomfort vote  (AUC$_0$=%.2f, walk-grouped)"%np.mean(baseA),fontsize=7.5)
handles=[Line2D([0],[0],marker="s",color="w",markerfacecolor=COLF[k],ms=6,label=l) for k,l in
         [("local","local thermal (sun, wind, local T/HDX)"),("regime","walk temperature regime"),("urban","urban form"),("personal","personal / position")]]
ax.legend(handles=handles,fontsize=4.8,frameon=False,loc="lower right")
save(fig,"F_ml_importance")
gtot={g:sum(impm[f] for f in feats if FAM[f]==g) for g in COLF}
print("grouped:",{g:round(v,3) for g,v in gtot.items()})
print("top:",[(NICE[f],round(impm[f],3),round(sgn[f],2) if not np.isnan(sgn[f]) else None) for f in ordf[::-1][:6]])
