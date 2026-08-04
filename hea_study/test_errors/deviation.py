#!/usr/bin/env python3
"""
Compute each pruned model's DEVIATION from the pct100 (full-mixed-train) model on
the 25 mixed holdout frames, treating pct100 as ground truth. Aggregates by
(alpha, retention) over replicates. Reads cached predictions from results_pred/.

  force deviation  = RMSD of force components (M vs pct100), reported in eV/A
  energy deviation = per-atom energy diff, offset-corrected per model
                     (ChIMES total energy has a model-dependent constant), in eV/atom

Output: deviation_summary.csv  (+ per-model deviation_permodel.csv)
Usage:  python3 deviation.py
"""
import os, re, glob, csv
import numpy as np

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_DIR = os.path.join(WORK_DIR, "results_pred")
KCAL_TO_EV = 1.0 / 23.0605
NAME_RE = re.compile(r"^a(\d{3})_pct(\d{3})_rep(\d{2})$")


def load_pred(model):
    """Return {frame_id: (forces natoms x3, energy, natoms)} for a model, or None."""
    mdir = os.path.join(PRED_DIR, model)
    if not os.path.exists(os.path.join(mdir, "DONE")):
        return None
    out = {}
    for fp in glob.glob(os.path.join(mdir, "f*.npz")):
        d = np.load(fp)
        out[int(d["frame_id"])] = (d["forces"].astype(np.float64), float(d["energy"]), int(d["natoms"]))
    return out


def deviation(model_pred, ref_pred):
    """Force RMSD (all components) and offset-corrected per-atom energy RMSD, in eV units."""
    fids = sorted(set(model_pred) & set(ref_pred))
    fsq = []          # squared force-component diffs (kcal/mol/A)
    e_pa_diff = []    # per-atom energy diff per frame (kcal/mol/atom)
    for fid in fids:
        fM, eM, nat = model_pred[fid]
        fR, eR, _ = ref_pred[fid]
        fsq.append(((fM - fR) ** 2).ravel())
        e_pa_diff.append((eM - eR) / nat)
    fsq = np.concatenate(fsq)
    f_rmsd = np.sqrt(fsq.mean()) * KCAL_TO_EV
    e_pa_diff = np.array(e_pa_diff)
    e_shift = e_pa_diff - e_pa_diff.mean()           # remove model-dependent energy offset
    e_rmsd = np.sqrt((e_shift ** 2).mean()) * KCAL_TO_EV
    return f_rmsd, e_rmsd, len(fids)


def main():
    ref = load_pred("pct100")
    if ref is None:
        print("pct100 predictions not ready yet (reference model not built/predicted). "
              "Re-run once pct100 has a DONE marker in results_pred/.")
        return

    permodel = []
    for mdir in sorted(glob.glob(os.path.join(PRED_DIR, "a*_pct*_rep*"))):
        model = os.path.basename(mdir)
        m = NAME_RE.match(model)
        if not m:
            continue
        pred = load_pred(model)
        if pred is None:
            continue
        alpha = int(m.group(1)) / 100.0
        ret = int(m.group(2))
        rep = int(m.group(3))
        f_dev, e_dev, nf = deviation(pred, ref)
        permodel.append(dict(model=model, alpha=alpha, retention=ret, rep=rep,
                             f_dev_eVA=f_dev, e_dev_eV_atom=e_dev, n_frames=nf))

    if not permodel:
        print("no pruned-model predictions found yet."); return

    with open(os.path.join(WORK_DIR, "deviation_permodel.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(permodel[0].keys())); w.writeheader(); w.writerows(permodel)

    # aggregate by (alpha, retention) over reps
    agg = {}
    for r in permodel:
        agg.setdefault((r["alpha"], r["retention"]), []).append(r)
    rows = []
    for (alpha, ret), items in sorted(agg.items()):
        fd = np.array([x["f_dev_eVA"] for x in items])
        ed = np.array([x["e_dev_eV_atom"] for x in items])
        rows.append(dict(alpha=alpha, retention=ret, n_reps=len(items),
                         f_dev_mean=fd.mean(), f_dev_std=fd.std(ddof=1) if len(fd) > 1 else 0.0,
                         e_dev_mean=ed.mean(), e_dev_std=ed.std(ddof=1) if len(ed) > 1 else 0.0))
    with open(os.path.join(WORK_DIR, "deviation_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"Wrote deviation for {len(permodel)} models "
          f"({len({r['model'][:r['model'].rindex('_rep')] for r in permodel})} alpha/retention cells).")
    print(f"{'alpha':>6}{'ret%':>6}{'reps':>6}{'F_dev(eV/A)':>14}{'E_dev(eV/at)':>14}")
    for r in rows:
        print(f"{r['alpha']:6.2f}{r['retention']:6d}{r['n_reps']:6d}"
              f"{r['f_dev_mean']:11.4f}+-{r['f_dev_std']:.3f}{r['e_dev_mean']:11.5f}+-{r['e_dev_std']:.4f}")


if __name__ == "__main__":
    main()
