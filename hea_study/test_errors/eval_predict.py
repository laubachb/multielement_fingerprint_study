#!/usr/bin/env python3
"""
Predict ChIMES forces+energy on the mixed holdout frames for one model, caching
EACH FRAME to disk as it is computed so an interrupted run resumes trivially.

Layout:
    results_pred/<model>/f<frame_id>.npz   one file per frame (forces, energy, natoms)
    results_pred/<model>/DONE              written once all frames are present
On restart, frames whose f<id>.npz already exists are skipped.

Usage:  python3 eval_predict.py <model_name>       # e.g. a000_pct025_rep00, pct100
"""
import os, sys, ctypes
import numpy as np

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(WORK_DIR, "..", "pruned_models", "runs_holdout20_mixed_clean")
FRAMES_NPZ = os.path.join(WORK_DIR, "test_frames_mixed.npz")
OUT_DIR = os.path.join(WORK_DIR, "results_pred")
LIBCHIMES = os.environ.get("CHIMES_CALC_LIB", "libchimescalc_dl.so")


def load_params(lib, param_path, rank=0):
    lib.set_chimes_serial(ctypes.c_bool(False))
    lib.init_chimes_serial(ctypes.c_char_p(param_path.encode()), ctypes.byref(ctypes.c_int(rank)))


def calc_frame(lib, fr):
    n = fr["natoms"]; xyz = fr["coords"]; cell = fr["cell"]
    typ = (ctypes.c_char_p * n)(*[e.encode() for e in fr["elements"]])
    x = (ctypes.c_double * n)(*xyz[:, 0]); y = (ctypes.c_double * n)(*xyz[:, 1]); z = (ctypes.c_double * n)(*xyz[:, 2])
    a = (ctypes.c_double * 3)(*cell[0]); b = (ctypes.c_double * 3)(*cell[1]); c = (ctypes.c_double * 3)(*cell[2])
    e = ctypes.c_double(0.0)
    fx = (ctypes.c_double * n)(*(0.0,) * n); fy = (ctypes.c_double * n)(*(0.0,) * n); fz = (ctypes.c_double * n)(*(0.0,) * n)
    stress = (ctypes.c_double * 9)(*(0.0,) * 9)
    lib.calculate_chimes(ctypes.c_int(n), x, y, z, typ, a, b, c,
                         ctypes.byref(e), fx, fy, fz, stress)
    return e.value, np.column_stack([list(fx), list(fy), list(fz)])


def main():
    model = sys.argv[1]
    param_path = os.path.join(RUNS_DIR, model, "params.txt")
    if not os.path.isfile(param_path):
        print(f"ERROR: no params for {model}", flush=True); sys.exit(1)

    mdir = os.path.join(OUT_DIR, model)
    os.makedirs(mdir, exist_ok=True)
    done_marker = os.path.join(mdir, "DONE")
    if os.path.exists(done_marker):
        print(f"{model}: DONE, skip", flush=True); return

    frames = list(np.load(FRAMES_NPZ, allow_pickle=True)["frames"])

    # figure out what still needs doing (per-frame resume)
    todo = [fr for fr in frames if not os.path.exists(os.path.join(mdir, f"f{fr['frame_id']}.npz"))]
    if not todo:
        open(done_marker, "w").close()
        print(f"{model}: all {len(frames)} frames already cached", flush=True); return

    lib = ctypes.CDLL(LIBCHIMES)
    load_params(lib, param_path)
    print(f"{model}: {len(frames)-len(todo)}/{len(frames)} cached, computing {len(todo)} …", flush=True)

    for fr in todo:
        e, f = calc_frame(lib, fr)
        fp = os.path.join(mdir, f"f{fr['frame_id']}.npz")
        # tmp must already end in .npz (np.savez_compressed appends .npz otherwise);
        # leading-dot prefix keeps it out of the f*.npz resume glob.
        tmp = os.path.join(mdir, f".tmp_f{fr['frame_id']}.npz")
        np.savez_compressed(tmp, forces=f.astype(np.float32), energy=np.float64(e),
                            natoms=np.int32(fr["natoms"]), frame_id=np.int32(fr["frame_id"]))
        os.replace(tmp, fp)  # atomic: a killed run never leaves a half-written frame

    # all present -> mark done
    if all(os.path.exists(os.path.join(mdir, f"f{fr['frame_id']}.npz")) for fr in frames):
        open(done_marker, "w").close()
        print(f"{model}: DONE ({len(frames)} frames)", flush=True)
    else:
        print(f"{model}: partial", flush=True)


if __name__ == "__main__":
    main()
