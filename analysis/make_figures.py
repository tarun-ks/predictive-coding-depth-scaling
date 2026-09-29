"""Generate all figures as vector PDFs, one image per file.

Every figure is a data visualisation computed directly from the collected runs.
"""
from __future__ import annotations
import json, glob, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path("../paper_nn/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
CPC, CALM, CTH = "#0072B2", "#D55E00", "#666666"

def rows(p):
    return [json.loads(l) for f in glob.glob(p) for l in open(f) if l.strip()]

def ttarget(R, m, depths, frac=0.90, seeds=range(5)):
    bp={(r['depth'],r['seed']):r['test_acc'] for r in R if r['method']=='bp'}
    out={}
    for L in depths:
        v=[]
        for s in seeds:
            rr=sorted([r for r in R if r['method']==m and r['depth']==L and r['seed']==s],
                      key=lambda r:r['budget'])
            if (L,s) not in bp or not rr: continue
            h=next((r['budget'] for r in rr if r['test_acc']>=frac*bp[(L,s)]), None)
            if h: v.append(h)
        if v: out[L]=v
    return out

# ---------------- Figure 1: T_target vs depth, both grids ----------------
R=rows("results/sweep/*.jsonl"); O=rows("results/offset/*.jsonl")
fig,ax=plt.subplots(figsize=(4.6,3.6))
for m,c,lab in (("pc",CPC,"PC"),("pcalm",CALM,"PC-ALM")):
    d=ttarget(R,m,(4,8,16,32,64,128))
    L=np.array(sorted(d)); y=np.array([np.mean(d[k]) for k in L])
    ax.plot(L,y,"o-",color=c,label=f"{lab} (geometric grid)",ms=5)
    do=ttarget(O,m,(8,16,32,64))
    if do:
        Lo=np.array(sorted(do)); yo=np.array([np.mean(do[k]) for k in Lo])
        ax.plot(Lo,yo,"s--",color=c,alpha=.55,ms=4,label=f"{lab} (offset grid)")
Lr=np.array([4,128])
ax.plot(Lr,3*(Lr/4)**2,":",color=CTH,lw=1.2)
ax.plot(Lr,3*(Lr/4)**1,"-.",color=CTH,lw=1.2)
ax.plot(Lr,(Lr-2),color="k",lw=1.0,alpha=.8)
ax.text(120,120,r"$\Omega(L)$ floor",fontsize=7,ha="right")
ax.text(120,3*(128/4)**2*1.15,r"$L^2$",fontsize=7,color=CTH,ha="right")
ax.text(120,3*(128/4)*1.15,r"$L^1$",fontsize=7,color=CTH,ha="right")
ax.set_xscale("log",base=2); ax.set_yscale("log",base=2)
ax.set_xlabel("Depth $L$"); ax.set_ylabel(r"$T_{\mathrm{target}}$ (90% of depth-matched BP)")
ax.legend(fontsize=7,frameon=False,loc="upper left")
fig.savefig(OUT/"Figure_1.pdf"); plt.close(fig); print("Figure_1.pdf")

# ---------------- Figure 2: kappa vs depth, muP vs standard ----------------
def kap(files):
    d={}
    for f in files:
        try: rr=list(csv.DictReader(open(f)))
        except Exception: continue
        for r in rr:
            dep=(r.get('depth') or '').strip()
            k=(r.get('kappa') or '').strip()
            if not dep.isdigit() or k in ('','nan'): continue
            d.setdefault(int(dep),[]).append(float(k))
    return d
mup=kap(["results/kappa/standard_w32.csv","results/kappa/mup_256.csv"])
sp =kap(["results/kappa/sp_w32.csv","results/kappa/sp_64.csv"])
fig,ax=plt.subplots(figsize=(4.6,3.6))
for d,c,lab,mk in ((mup,CPC,r"$\mu$P parameterization","o"),(sp,CALM,"standard parameterization","s")):
    L=np.array(sorted(d)); y=np.array([np.mean(d[k]) for k in L])
    if d is sp:
        # L=64 is an indicative estimate only (indicative only): the shifted-operator recovery
        # loses digits in proportion to kappa and the implied error there exceeds unity.
        # Draw it hollow so the figure cannot be read as four resolved points plus one.
        res=L<64; ind=L>=64
        ax.plot(L[res],y[res],mk+"-",color=c,label=lab,ms=5)
        ax.plot(L,y,"-",color=c,lw=1,alpha=.5)
        ax.plot(L[ind],y[ind],mk,mfc="none",mec=c,ms=6,mew=1.2,
                label="standard, indicative only")
    else:
        ax.plot(L,y,mk+"-",color=c,label=lab,ms=5)
L=np.array(sorted(mup)); y=np.array([np.mean(mup[k]) for k in L])
r=stats.linregress(np.log(L),np.log(y))
ax.plot(L,np.exp(r.intercept)*L**2.0,":",color=CTH,lw=1.2,label=r"$\kappa\propto L^2$")
ax.set_xscale("log",base=2); ax.set_yscale("log")
ax.set_xlabel("Depth $L$"); ax.set_ylabel(r"condition number $\kappa(H_z)$")
ax.legend(fontsize=7,frameon=False,loc="upper left")
ax.annotate("not resolved\nbeyond $L$=64",xy=(64,1.2e8),xytext=(10,3e8),fontsize=7,
            color=CALM,arrowprops=dict(arrowstyle="->",color=CALM,lw=.8))
fig.savefig(OUT/"Figure_2.pdf"); plt.close(fig); print("Figure_2.pdf")

# ---------------- Figure 3: accuracy vs budget, step-size sweep ----------------
T2=rows("results/task2/*.jsonl")
fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),sharey=True)
for ax,L in zip(axes,(64,128)):
    for sc,c in zip((1.0,0.5,0.25,0.1),(CPC,"#56B4E9",CALM,"#E69F00")):
        pts={}
        for r in T2:
            if r['depth']==L and r['lr_scale']==sc:
                pts.setdefault(r['mult_of_knee'],[]).append(r['test_acc'])
        if not pts: continue
        x=np.array(sorted(pts)); y=np.array([100*np.mean(pts[k]) for k in x])
        ax.plot(x,y,"o-",color=c,ms=4,label=rf"$\eta_h\times${sc}")
    ax.set_xscale("log",base=2); ax.set_xlabel(r"budget $T$ ($\times$ own knee)")
    ax.set_title(f"$L$={L}",fontsize=9)
axes[0].set_ylabel("test accuracy (%)"); axes[0].legend(fontsize=7,frameon=False,loc="lower right")
fig.savefig(OUT/"Figure_3.pdf"); plt.close(fig); print("Figure_3.pdf")

# ---------------- Figure 4: per-layer rotation heatmap ----------------
fig,axes=plt.subplots(1,3,figsize=(7.4,2.7))
for ax,L in zip(axes,(16,64,128)):
    fs=sorted(glob.glob(f"results/item1/L{L}_s*.npz"))
    if not fs: continue
    zs=[np.load(f) for f in fs]
    Ts=zs[0]["Ts"]; cl=np.mean([z["per_layer_cos"] for z in zs],axis=0)
    im=ax.imshow(cl.T,aspect="auto",origin="lower",cmap="viridis",vmin=.55,vmax=1.0,
                 extent=[0,len(Ts)-1,0,cl.shape[1]])
    xt=[i for i,t in enumerate(Ts) if t in (1,16,256,2763)]
    ax.set_xticks(xt); ax.set_xticklabels([str(Ts[i]) for i in xt],fontsize=7)
    ax.set_xlabel("budget $T$"); ax.set_title(f"$L$={L}",fontsize=9)
    if ax is axes[0]: ax.set_ylabel("layer index (0 = input side)")
    ax.grid(False)
fig.colorbar(im,ax=axes,shrink=.85,label=r"$\cos(g_\ell,b_\ell)$")
fig.savefig(OUT/"Figure_4.pdf", dpi=600)  # imshow is raster; 600 dpi meets the
                                          # journal threshold for combination art
plt.close(fig); print("Figure_4.pdf")

# ---------------- Figure 5: residual trajectories ----------------
fig,ax=plt.subplots(figsize=(4.6,3.6))
for L,c in ((16,"#56B4E9"),(64,CPC),(256,"#003f5c")):
    fs=sorted(glob.glob(f"results/resid/L{L}_s*_pc.npz"))
    if fs:
        y=np.median([np.load(f)["trained"] for f in fs],axis=0)
        ax.plot(np.arange(1,len(y)+1),y,color=c,label=f"PC, $L$={L}")
for L,c in ((16,"#F0A868"),(64,CALM),(256,"#8c2d04")):
    fs=sorted(glob.glob(f"results/resid/L{L}_s*_pcalm.npz"))
    if fs:
        y=np.median([np.load(f)["trained"] for f in fs],axis=0)
        ax.plot(np.arange(1,len(y)+1),y,"--",color=c,label=f"PC-ALM, $L$={L}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("inner iteration $T$"); ax.set_ylabel("max relative constraint residual")
ax.legend(fontsize=6.5,frameon=False,ncol=2)
ax.set_title("PC's residual grows from zero; it is the error signal",fontsize=8)
fig.savefig(OUT/"Figure_5.pdf"); plt.close(fig); print("Figure_5.pdf")
