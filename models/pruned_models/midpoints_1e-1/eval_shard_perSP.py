#!/usr/bin/env python3
"""Eval a SHARD of converged lambda=1e-1 models, PER STATEPOINT, on the reduced
CN test set. Usage: eval_shard_perSP.py <shard_idx> <n_shards>
Writes results/perSP_shard_<idx>.csv  (long format).
"""
import os, sys, ctypes, glob, re, csv, time
import numpy as np
import multiprocessing as mp
from collections import defaultdict

# machine-specific paths are read from the environment (set before running):
#   CHIMES_CALC_LIB  path to libchimescalc_dl.so (ChIMES calculator shared lib)
#   CN_TEST_NPZ      reduced CN test set (reduced_test_10perSP.npz)
#   WORK             directory holding the pruned_midpoints_1e-1/ run dirs
LIB = os.environ.get("CHIMES_CALC_LIB", "libchimescalc_dl.so")
REDNPZ = os.environ.get("CN_TEST_NPZ", "reduced_test_10perSP.npz")
BASE = os.path.join(os.environ.get("WORK", "."), "pruned_midpoints_1e-1")
RESDIR = os.path.join(BASE, "results")
KC = 1.0/23.0605
HB = 27.211386245988/0.529177210903
NAME = re.compile(r"^a(\d{3})_pct(\d{3})_rep(\d{2})$")

_frames=None; _lib=None
def _init(params):
    global _frames, _lib
    try: os.sched_setaffinity(0, range(os.cpu_count() or 1))
    except Exception: pass
    _frames = list(np.load(REDNPZ, allow_pickle=True)["frames"])
    dn=os.open(os.devnull,os.O_WRONLY); sv=os.dup(1); os.dup2(dn,1)
    _lib=ctypes.CDLL(LIB); _lib.set_chimes_serial(ctypes.c_bool(False))
    _lib.init_chimes_serial(ctypes.c_char_p(params.encode()), ctypes.byref(ctypes.c_int(0)))
    os.dup2(sv,1); os.close(sv); os.close(dn)

def _work(i):
    fr=_frames[i]; nat=fr["natoms"]; co=fr["coords"]; ce=fr["cell"]
    enc=[e.encode() for e in fr["elements"]]
    ix=(ctypes.c_double*nat)(*co[:,0]); iy=(ctypes.c_double*nat)(*co[:,1]); iz=(ctypes.c_double*nat)(*co[:,2])
    ityp=(ctypes.c_char_p*nat)(*enc)
    a=(ctypes.c_double*3)(*ce[0]); b=(ctypes.c_double*3)(*ce[1]); c=(ctypes.c_double*3)(*ce[2])
    ie=ctypes.c_double(0.0)
    fx=(ctypes.c_double*nat)(*(0.0,)*nat); fy=(ctypes.c_double*nat)(*(0.0,)*nat); fz=(ctypes.c_double*nat)(*(0.0,)*nat)
    st=(ctypes.c_double*9)(*(0.0,)*9)
    _lib.calculate_chimes(ctypes.c_int(nat), ix,iy,iz, ityp, a,b,c, ctypes.byref(ie), fx,fy,fz, st)
    pf=(np.column_stack([list(fx),list(fy),list(fz)]).astype(float)*KC).ravel()
    rf=(np.asarray(fr["forces"], float)*HB).ravel()
    return str(fr["statepoint"]), pf, rf

def converged_models():
    out=[]
    for d in sorted(glob.glob(os.path.join(BASE,"a*_pct*_rep*"))):
        m=NAME.match(os.path.basename(d)); p=os.path.join(d,"params.txt")
        if m and os.path.isfile(p):
            with open(p,"rb") as f:
                if b"ENDFILE" in f.read(): out.append((os.path.basename(d),p,m))
    return out

def main():
    shard=int(sys.argv[1]); nsh=int(sys.argv[2])
    os.makedirs(RESDIR, exist_ok=True)
    n=len(np.load(REDNPZ, allow_pickle=True)["frames"])
    models=[x for k,x in enumerate(converged_models()) if k % nsh == shard]
    print(f"[shard {shard}/{nsh}] {len(models)} models", flush=True)
    rows=[]
    for name,params,m in models:
        t=time.time()
        with mp.Pool(min(48,n), initializer=_init, initargs=(params,)) as pool:
            res=pool.map(_work, range(n))
        bySP=defaultdict(lambda:[[],[]])
        allP=[]; allR=[]
        for sp,pf,rf in res:
            bySP[sp][0].append(pf); bySP[sp][1].append(rf); allP.append(pf); allR.append(rf)
        P=np.concatenate(allP); R=np.concatenate(allR)
        over=float(np.sqrt(((P-R)**2).mean()))
        for sp in sorted(bySP):
            p=np.concatenate(bySP[sp][0]); r=np.concatenate(bySP[sp][1])
            rows.append({"model":name,"alpha":int(m.group(1)),"pct":int(m.group(2)),
                         "rep":int(m.group(3)),"statepoint":sp,
                         "force_rmse_eVA":round(float(np.sqrt(((p-r)**2).mean())),4)})
        rows.append({"model":name,"alpha":int(m.group(1)),"pct":int(m.group(2)),
                     "rep":int(m.group(3)),"statepoint":"OVERALL","force_rmse_eVA":round(over,4)})
        print(f"[shard {shard}] {name} overall={over:.3f} ({time.time()-t:.0f}s)", flush=True)
    out=os.path.join(RESDIR,f"perSP_shard_{shard}.csv")
    with open(out,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["model","alpha","pct","rep","statepoint","force_rmse_eVA"])
        w.writeheader(); w.writerows(rows)
    print(f"[shard {shard}] wrote {out}", flush=True)

if __name__=="__main__":
    main()
