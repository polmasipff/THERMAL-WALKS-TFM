"""Shared builder for Chapter 5 figures: stop-to-stop C/N/U transitions + T_env regime."""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__))
BASE=os.path.join(HERE,"..","data","markovian_analysis_baseline.csv")
CUTS=(24.6,29.8)
def transitions():
    d=pd.read_csv(BASE).dropna(subset=['ID','walk_id','stop_idx','comfort3','T_env']).copy()
    d['stop_idx']=d.stop_idx.astype(int); St={'comfortable':0,'neutral':1,'uncomfortable':2}; d['s']=d.comfort3.map(St)
    rows=[]
    for (w,i),g in d.sort_values('stop_idx').groupby(['walk_id','ID']):
        g=g.reset_index(drop=True)
        for a in range(len(g)-1):
            if g.stop_idx[a+1]-g.stop_idx[a]==1: rows.append((w,g.s[a],g.s[a+1],g.T_env[a]))
    T=pd.DataFrame(rows,columns=['walk','si','sj','Tenv'])
    T['reg']=T.Tenv.map(lambda t:0 if t<CUTS[0] else (1 if t<CUTS[1] else 2))
    return T
def countmat(sub):
    C=np.zeros((3,3))
    for si,sj in zip(sub.si,sub.sj): C[int(si),int(sj)]+=1
    return C
def affinity(C):
    P=C/np.clip(C.sum(1,keepdims=True),1,None)
    num=P[0,1]*P[1,2]*P[2,0]; den=P[1,0]*P[2,1]*P[0,2]
    if num<=0 or den<=0: return np.nan,np.nan,P
    wv,vv=np.linalg.eig(P.T); pi=np.real(vv[:,np.argmin(abs(wv-1))]); pi=pi/pi.sum()
    return np.log(num/den), pi[0]*P[0,1]-pi[1]*P[1,0], P
