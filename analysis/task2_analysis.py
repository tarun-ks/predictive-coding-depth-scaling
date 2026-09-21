"""TASK 2: is the post-knee accuracy decay windup, or a step-size artefact?

Pre-registered reading (PREREGISTRATION.md, recorded before completion):
  decay persists at all step-size scales -> dual windup
  decay weakens/vanishes as the step shrinks -> step-size interaction
Each condition is compared at the SAME multiple of its OWN knee, since the knee
moves as ~1/lr_scale; comparing at fixed T would confound the two effects.
"""
from __future__ import annotations
import json, glob, statistics as S

rows=[json.loads(l) for f in glob.glob("results/task2/*.jsonl") for l in open(f) if l.strip()]
print("Peak accuracy, its location, and accuracy at 4x the knee. 3 seeds.")
print("'drop' = peak - acc@4x knee, in percentage points. A step-size artefact")
print("would show drop shrinking toward 0 as lr_scale shrinks.\n")
print(f"  {'L':>4s} {'lr_scale':>9s} {'n':>3s} {'peak%':>8s} {'argmax(xknee)':>14s} "
      f"{'acc@4x%':>9s} {'drop(pp)':>9s}")
summary={}
for L in (64,128):
    for sc in (1.0,0.5,0.25,0.1):
        rs=[r for r in rows if r['depth']==L and r['lr_scale']==sc]
        if not rs: continue
        seeds=sorted({r['seed'] for r in rs})
        peaks,args,at4,drops=[],[],[],[]
        for s in seeds:
            rr=sorted([r for r in rs if r['seed']==s], key=lambda r:r['budget'])
            if len(rr)<4: continue
            best=max(rr,key=lambda r:r['test_acc'])
            four=[r for r in rr if abs(r['mult_of_knee']-4.0)<0.01]
            if not four: continue
            peaks.append(best['test_acc']); args.append(best['mult_of_knee'])
            at4.append(four[0]['test_acc']); drops.append(best['test_acc']-four[0]['test_acc'])
        if not drops:
            print(f"  {L:4d} {sc:9.2f} {len(seeds):3d}   -- incomplete --")
            continue
        summary[(L,sc)]=100*S.mean(drops)
        sd=100*S.stdev(drops) if len(drops)>1 else 0.0
        print(f"  {L:4d} {sc:9.2f} {len(drops):3d} {100*S.mean(peaks):8.2f} "
              f"{S.mean(args):14.2f} {100*S.mean(at4):9.2f} {100*S.mean(drops):6.2f}+-{sd:.2f}")
    print()
print("VERDICT per depth (does the drop shrink as the step shrinks?):")
for L in (64,128):
    ks=[(sc,summary[(L,sc)]) for sc in (1.0,0.5,0.25,0.1) if (L,sc) in summary]
    if len(ks)<2: 
        print(f"  L={L}: insufficient scales"); continue
    txt="  ".join(f"{sc}:{d:+.2f}pp" for sc,d in ks)
    big,small=ks[0][1],ks[-1][1]
    trend = ("DECAY PERSISTS (windup)" if small>=0.5*big and small>0.5
             else "DECAY VANISHES (step-size artefact)" if small<0.5 
             else "DECAY WEAKENS (partial step-size interaction)")
    print(f"  L={L}: {txt}   -> {trend}")
