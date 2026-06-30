"""F_heterogeneity_sidebyside -- comfort<->discomfort threshold T(m=0) by stratum, measured in
T_corr (left) and T_env (right). Logistic [U vs C] ~ coord, threshold -b0/b1, walk-stratified
bootstrap. Three blocks: environmental (sun/wind), personal (gender/age), place (city).
Side-by-side, shared x-axis scale, all CIs fully visible."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _common import set_style, set_paper_style, save, CM
set_style(); set_paper_style()

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
DEG = r"$^{\circ}$C"
B = os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")
RNG = np.random.default_rng(2); NB = 2000

d = pd.read_csv(B).copy()
d["city"] = d["city"].replace({"L'Hospitalet de Llobreat":"L'Hospitalet de Llobregat"})
d = d.dropna(subset=["comfort3","T_corr","T_env","walk_id"]).copy()
d = d[d.comfort3.isin(["comfortable","uncomfortable"])].copy()
d["y"] = (d.comfort3=="uncomfortable").astype(float)
YOUNG = {"Less than 10","10-12","13-15"}; ELD = {"65-74","75-84"}
d["agebin"] = d.age.map(lambda a:"le15" if a in YOUNG else ("ge16" if pd.notna(a) and a not in ELD else np.nan))
CITYAB = {"L'Hospitalet de Llobregat":"L'Hospitalet","Montcada i Reixac":"Montcada",
          "Barri Sant Pere / La Ribera - Barcelona":"Sant Pere","Barri Congrés / Els Indians  - Barcelona":"Congrés",
          "Sant Vicenç dels Horts":"Sant Vicenç"}

def fit(sub,C):
    if len(sub)<40 or sub.y.nunique()<2: return np.nan
    x=sub[C].to_numpy(float); y=sub.y.to_numpy(float); X=np.column_stack([np.ones_like(x),x]); b=np.zeros(2)
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(X@b,-30,30))); W=np.maximum(p*(1-p),1e-9)
        try: s=np.linalg.solve((X*W[:,None]).T@X+1e-8*np.eye(2),X.T@(y-p))
        except: return np.nan
        b+=s
        if np.max(np.abs(s))<1e-9: break
    return -b[0]/b[1] if b[1]>0 else np.nan

# ------------------------------------------------------------
# Bootstrap: retorna totes les mostres i després les resumeix
# ------------------------------------------------------------

def boot_samples(sub, C):
    wl = sub.walk_id.unique()

    if len(wl) == 0 or len(sub) == 0:
        return np.array([])

    wi = {
        w: sub.index[sub.walk_id == w].to_numpy()
        for w in wl
    }

    o = []

    for _ in range(NB):
        sampled_walks = RNG.choice(wl, len(wl), replace=True)

        idx = np.concatenate([
            wi[w] for w in sampled_walks
        ])

        t = fit(sub.loc[idx], C)

        if np.isfinite(t):
            o.append(t)

    return np.array(o)


def summarize_boot(o):
    o = np.asarray(o)
    o = o[np.isfinite(o)]

    if len(o) == 0:
        return (np.nan, np.nan, np.nan)

    return (
        np.median(o),
        np.percentile(o, 10),
        np.percentile(o, 90)
    )


def pct_lower(a, b, n_mc=100000, seed=123):
    """
    Percentatge de comparacions bootstrap on a < b.
    Per exemple: Women < Men.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    rng = np.random.default_rng(seed)

    aa = rng.choice(a, size=n_mc, replace=True)
    bb = rng.choice(b, size=n_mc, replace=True)

    return 100 * np.mean(aa < bb)


# ------------------------------------------------------------
# Definició dels subgrups
# ------------------------------------------------------------

# sun_s / wind_s no existeixen en aquest csv -> els derivem
SUNMAP = {"In full shade": 0.0, "In a mixture of sun and shadow": 0.5, "In full sun": 1.0}
d["sun_s"]  = d["sun"].map(SUNMAP)
d["wind_s"] = d["wind_sc"]          # mateixa escala 0–1 (0, 0.25, 0.5, 0.75, 1)


g = d[d.gender.isin(["Man", "Woman"])]

items = [
    ("All votes", d, "ref"),

    ("ENV", "header", None),
    ("Shade", d[d.sun_s == 0], "env"),
    ("Sun", d[d.sun_s >= 0.5], "env"),
    ("Calm", d[d.wind_s <= 0.25], "env"),
    ("Windy", d[d.wind_s >= 0.5], "env"),

    ("PERSONAL", "header", None),
    ("Women", g[g.gender == "Woman"], "gender"),
    ("Men", g[g.gender == "Man"], "gender"),
    (r"Age $\leq$15", d[d.agebin == "le15"], "age"),
    (r"Age $\geq$16", d[d.agebin == "ge16"], "age"),

    ("PLACE", "header", None)
]

for full, ab in sorted(CITYAB.items(), key=lambda kv: -len(d[d.city == kv[0]])):
    sub = d[d.city == full]
    frag = " *" if sub.walk_id.nunique() < 4 else ""
    items.append((ab + frag, sub, "place"))


# ------------------------------------------------------------
# Càlcul de la taula principal + guardat de mostres bootstrap
# ------------------------------------------------------------

rows = []
boots = {}

for lab, sub, kind in items:

    if isinstance(sub, str):
        rows.append((lab, kind, "header"))
        continue

    boot_Tcorr = boot_samples(sub, "T_corr")
    boot_Tenv  = boot_samples(sub, "T_env")

    boots[(lab, "T_corr")] = boot_Tcorr
    boots[(lab, "T_env")]  = boot_Tenv

    rows.append((
        lab,
        {
            "T_corr": summarize_boot(boot_Tcorr),
            "T_env": summarize_boot(boot_Tenv),
            "n": len(sub)
        },
        kind
    ))


# ------------------------------------------------------------
# Comparacions entre subgrups
# ------------------------------------------------------------

for C in ["T_corr", "T_env"]:

    pct_women_lower = pct_lower(
        boots[("Women", C)],
        boots[("Men", C)]
    )

    pct_young_lower = pct_lower(
        boots[(r"Age $\leq$15", C)],
        boots[(r"Age $\geq$16", C)]
    )

    print(f"{C}")
    print(f"Women < Men: {pct_women_lower:.0f}%")
    print(f"Age <= 15 < Age >= 16: {pct_young_lower:.0f}%")
    print()
# ── LÍMIT X COMPARTIT: rang global dels CI de les dues coordenades ────────────
# ── LÍMITS FIXOS (mateixa amplada 10°C → mateixa escala visual per unitat) ───
XLIMS={"T_corr":(25,35),"T_env":(22,33)}

# ── PALETA ────────────────────────────────────────────────────────────────────
COL = {
    "ref": "#918686",
    "env": "#E07B39",
    "gender": "#9CE5E8",
    "place": "#2C7AB5",
    "age": "#CC639D",   # purple/magenta, color-blind friendly
}
COL_DARK="#152F61"
HEAD={"ENV":"environmental (sun / wind)","PERSONAL":"personal (gender / age)","PLACE":"place (city)"}
n=len(rows); y=np.arange(n)[::-1]

# ── FIGURA: costat a costat, gran, molt separats ─────────────────────────────
fig,(axL,axR)=plt.subplots(1,2,figsize=(34*CM,14*CM),sharey=True)
fig.subplots_adjust(wspace=0.28,left=0.14,right=0.985,bottom=0.11,top=0.88)

for ax,C,xlab in [(axL,"T_corr",r"$T_{\mathrm{corr}}$  (°C)"),
                  (axR,"T_env", r"$T_{\mathrm{env}}$  (°C)")]:
    gmed=fit(d,C); x0,x1=XLIMS[C]
    ax.axvline(gmed,color="0.55",ls="--",lw=0.9,zorder=0)
    ax.text(gmed,len(rows)+0.3,f"{gmed:.1f}°C",fontsize=12,ha="center",
            va="bottom",color="0.55",fontweight="bold")

    for yi,(lab,val,kind) in zip(y,rows):
        if kind=="header":
            continue
        m,lo,hi=val[C]
        if np.isnan(m): continue
        lo2=max(lo,x0+0.2); hi2=min(hi,x1-0.2)
        ax.plot([lo2,hi2],[yi,yi],color=COL[kind],lw=2.4,
                solid_capstyle="round",zorder=2)
        # tapetes on el CI no sobresurt, fletxa on sobresurt
        if lo>=x0: ax.plot([lo,lo],[yi-0.20,yi+0.20],color=COL[kind],lw=1.6,zorder=2)
        else:       ax.plot([x0+0.2],[yi],marker="<",color=COL[kind],ms=5,zorder=3)
        if hi<=x1:  ax.plot([hi,hi],[yi-0.20,yi+0.20],color=COL[kind],lw=1.6,zorder=2)
        else:        ax.plot([x1-0.2],[yi],marker=">",color=COL[kind],ms=5,zorder=3)
        ax.scatter([m],[yi],color=COL[kind],s=38,zorder=4,
                   edgecolor="white",linewidth=0.6)

    ax.set_xlim(*XLIMS[C])
    ax.set_xlabel(xlab,fontsize=14,color=COL_DARK)
    ax.tick_params(axis="x",labelsize=12,colors=COL_DARK)
    ax.tick_params(axis="y",length=0)
    ax.grid(axis="x",alpha=0.18,zorder=0)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_edgecolor(COL_DARK)
    ax.spines["bottom"].set_linewidth(0.8)

axL.set_yticks([yi for yi,(l,v,k) in zip(y,rows) if k!="header"])
axL.set_yticklabels([l for (l,v,k) in rows if k!="header"],fontsize=11,color=COL_DARK)
axL.set_ylabel("Subgroup",fontsize=14,color=COL_DARK)

# ── LLEGENDA: a dalt, 2 files (ncol=3), sense caixa ─────────────────────────
leg=[Line2D([0],[0],color=COL["ref"],lw=2.4,marker="o",ms=7,
            markeredgecolor="white",markeredgewidth=0.6,label="All votes"),
     Line2D([0],[0],color=COL["env"],lw=2.4,marker="o",ms=7,
            markeredgecolor="white",markeredgewidth=0.6,label="Sun / wind"),
     Line2D([0],[0],color=COL["gender"],lw=2.4,marker="o",ms=7,
            markeredgecolor="white",markeredgewidth=0.6,label="Gender"),
    Line2D([0],[0],color=COL["age"],lw=2.4,marker="o",ms=7,
            markeredgecolor="white",markeredgewidth=0.6,label="Age"),
     Line2D([0],[0],color=COL["place"],lw=2.4,marker="o",ms=7,
            markeredgecolor="white",markeredgewidth=0.6,label="Place"),
     Line2D([0],[0],color="0.55",lw=0.9,ls="--",label="Pooled threshold")]
fig.legend(handles=leg,fontsize=11,loc="upper center",ncol=6,
           frameon=False,bbox_to_anchor=(0.50,1.00))

save(fig,"F_heterogeneity_sidebyside")
plt.show()
print("env spread T_corr=%.1f  T_env=%.1f"%(
    max(r[1]["T_corr"][0] for r in rows if r[2]=="env")-min(r[1]["T_corr"][0] for r in rows if r[2]=="env"),
    max(r[1]["T_env"][0]  for r in rows if r[2]=="env")-min(r[1]["T_env"][0]  for r in rows if r[2]=="env")))