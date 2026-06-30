"""make_occupation_csv.py
Builds one tidy CSV with the C/N/U occupation probabilities binned along BOTH thermal
coordinates (T_corr and T_env), with the Blume-Capel order parameters and effective fields.

Output: occupation_CNU_Tcorr_Tenv.csv   (long format, one row per coordinate x bin)
Columns: coordinate, bin_lo, bin_hi, T_center, n, P_C, P_N, P_U,
         P_C_lo, P_C_hi, P_N_lo, P_N_hi, P_U_lo, P_U_hi   (Wilson 95% CIs)
         m=P_U-P_C, q=P_U+P_C, beta_h=0.5 ln(P_U/P_C), beta_D=-0.5 ln(P_U P_C / P_N^2)

NOTE: the Blume-Capel chapter uses the T_corr block (bare field, non-circular). The T_env
block is provided for comparison only; T_env is calibrated against comfort (see build_T_env.py).
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.join(HERE,"..","data","votes_with_Tenv_comfort.csv")
MIN_N=10
LAB={"comfortable":"C","neutral":"N","uncomfortable":"U"}

def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (c-h,c+h)

d=pd.read_csv(D).dropna(subset=["state3","T_corr","T_env"]).copy()
d["cat"]=d.state3.map(LAB)

rows=[]
for coord in ["T_corr","T_env"]:
    x=d[coord]
    lo,hi=np.floor(x.min()),np.ceil(x.max())
    edges=np.arange(lo,hi+0.001,1.0)
    g=pd.cut(x,edges,right=False)
    for iv,sub in d.groupby(g,observed=True):
        n=len(sub)
        if n<MIN_N: continue
        nC=(sub.cat=="C").sum(); nN=(sub.cat=="N").sum(); nU=(sub.cat=="U").sum()
        pC,pN,pU=nC/n,nN/n,nU/n
        cClo,cChi=wilson(nC,n); cNlo,cNhi=wilson(nN,n); cUlo,cUhi=wilson(nU,n)
        m=pU-pC; q=pU+pC
        bh=0.5*np.log(pU/pC) if pC>0 and pU>0 else np.nan
        bd=-0.5*np.log(pU*pC/pN**2) if pC>0 and pU>0 and pN>0 else np.nan
        rows.append(dict(coordinate=coord,bin_lo=iv.left,bin_hi=iv.right,
            T_center=round((iv.left+iv.right)/2,2),n=n,
            P_C=round(pC,4),P_N=round(pN,4),P_U=round(pU,4),
            P_C_lo=round(cClo,4),P_C_hi=round(cChi,4),
            P_N_lo=round(cNlo,4),P_N_hi=round(cNhi,4),
            P_U_lo=round(cUlo,4),P_U_hi=round(cUhi,4),
            m=round(m,4),q=round(q,4),
            beta_h=round(bh,4) if not np.isnan(bh) else np.nan,
            beta_D=round(bd,4) if not np.isnan(bd) else np.nan))
out=pd.DataFrame(rows)
p=os.path.join(HERE,"occupation_CNU_Tcorr_Tenv.csv")
out.to_csv(p,index=False)
print("wrote",p,"->",len(out),"rows")
for coord in ["T_corr","T_env"]:
    s=out[out.coordinate==coord]
    print(f"\n=== {coord} ===  ({s.n.sum()} votes, {len(s)} bins)")
    print(s[["T_center","n","P_C","P_N","P_U","m","q","beta_h","beta_D"]].to_string(index=False))
