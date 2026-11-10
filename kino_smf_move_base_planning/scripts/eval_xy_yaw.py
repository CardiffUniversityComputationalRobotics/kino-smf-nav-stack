"""Head-to-head evaluation: combined XYYawCNN vs the original separate models.

The sampler does not care about F1. It calls checkXYInference() to decide whether to keep
a sampled configuration, so what matters operationally is:
   RECALL   - how much of the true near-optimal region survives the filter
   RETAINED - how much of the grid the filter keeps (uniform sampling keeps 100%)
A useful model has high recall at a small retained fraction. F1 punishes the model for
the fact that the near-optimal region is one random SST draw among many, so it understates
the value; both are reported.

The original models were trained with an UNSEEDED random_split, so some of this test set
was very likely in their training data. That bias favours them, not the combined model.

Run:  python3 eval_xy_yaw.py
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_model import XYYawCNN  # noqa: E402
import training_script_xy_yaw as T  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OLD_XY = os.path.expanduser("~/trained_model/model_xy_0.0001_final.pth")
OLD_YAW = os.path.expanduser("~/trained_model/model_yaw_0.1_final_artifact_mse.pth")
THRS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def load_test():
    d = T.preprocess()
    n = len(d["inp"])
    perm = np.random.default_rng(T.SEED).permutation(n)
    te = perm[int(0.8 * n) + int(0.1 * n):]
    return d, te


@torch.no_grad()
def collect(predict, d, te, bs=32):
    """predict(img, rs, so, go) -> (xy_prob [B,1,H,W], yaw_angle_rad [B,1,H,W] or None)"""
    ds = T.GridDataset(d, te)
    tp = {t: 0.0 for t in THRS}; fp = dict(tp); fn = dict(tp); kept = dict(tp)
    total = 0.0
    ang_sum = ang_n = 0.0
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        img = torch.stack([x[0] for x in items]).to(DEV)
        xy = torch.stack([x[1] for x in items]).to(DEV)
        yaw = torch.stack([x[2] for x in items]).to(DEV)
        ym = torch.stack([x[3] for x in items]).to(DEV).float()
        rs = torch.stack([x[4] for x in items]).to(DEV)
        so = torch.stack([x[5] for x in items]).to(DEV)
        go = torch.stack([x[6] for x in items]).to(DEV)
        p, ang = predict(img, rs, so, go)
        for t in THRS:
            b = (p > t).float()
            tp[t] += float((b * xy).sum()); fp[t] += float((b * (1 - xy)).sum())
            fn[t] += float(((1 - b) * xy).sum()); kept[t] += float(b.sum())
        total += float(xy.numel())
        if ang is not None:
            ang_sum += T.circular_error(ang, yaw, ym) * float(ym.sum())
            ang_n += float(ym.sum())
    rows = []
    for t in THRS:
        prec = tp[t] / max(tp[t] + fp[t], 1e-9)
        rec = tp[t] / max(tp[t] + fn[t], 1e-9)
        rows.append(dict(thr=t, prec=prec, rec=rec,
                         f1=2 * prec * rec / max(prec + rec, 1e-9),
                         iou=tp[t] / max(tp[t] + fp[t] + fn[t], 1e-9),
                         kept=kept[t] / max(total, 1e-9)))
    return rows, (ang_sum / ang_n if ang_n else None)


def show(name, rows, circ):
    print(f"\n--- {name} ---")
    print("  thr    P       R       F1      IoU     grid kept   speedup vs uniform")
    for r in rows:
        star = "  <- deployment" if abs(r["thr"] - 0.2) < 1e-9 else ""
        sp = 1.0 / max(r["kept"], 1e-9)
        print(f"  {r['thr']:.1f}  {r['prec']:.4f}  {r['rec']:.4f}  {r['f1']:.4f}  "
              f"{r['iou']:.4f}  {100*r['kept']:8.2f}%  {sp:12.1f}x{star}")
    if circ is not None:
        print(f"  orientation circular error: {circ:.4f} rad ({math.degrees(circ):.1f} deg)")


if __name__ == "__main__":
    d, te = load_test()
    print(f"test samples: {len(te)}  (seeded split, disjoint from train/val)")
    print(f"true positive rate in the targets: {100*d['xy'][te].mean():.3f}% of cells")

    # ---- combined model
    m = XYYawCNN(pretrained=False).to(DEV).eval()
    m.load_state_dict(torch.load(os.path.expanduser("~/07_08_dataset/trained/xy_yaw_best.pth"),
                                 map_location=DEV))

    def pred_new(img, rs, so, go):
        logits, c, s = m.raw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
        return torch.sigmoid(logits), torch.remainder(torch.atan2(s, c), T.TWO_PI)

    rows_new, circ_new = collect(pred_new, d, te)
    show("COMBINED XYYawCNN (this run)", rows_new, circ_new)

    # ---- originals
    try:
        old_xy = torch.jit.load(OLD_XY, map_location=DEV).eval()
        old_yaw = torch.jit.load(OLD_YAW, map_location=DEV).eval()

        def pred_old(img, rs, so, go):
            p = old_xy(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
            y = old_yaw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
            # the old yaw model emits the +1-offset value directly
            return p, torch.remainder(y - 1.0, T.TWO_PI)

        rows_old, circ_old = collect(pred_old, d, te)
        show("ORIGINAL separate models (may have TRAINED on part of this split)",
             rows_old, circ_old)
    except Exception as e:
        print(f"\n--- ORIGINAL models: could not evaluate ({type(e).__name__}: {str(e)[:150]}) ---")
        rows_old = None

    # ---- trivial baselines, to show what the numbers mean
    pr = float(d["xy"][te].mean())
    print(f"\n--- baselines ---")
    print(f"  keep everything (uniform sampler): R 1.0000  grid kept 100.00%  P {pr:.4f}")
    print(f"  random {100*pr:.2f}% of cells:      R {pr:.4f}  grid kept {100*pr:.2f}%  P {pr:.4f}")

    if rows_old:
        a = [r for r in rows_new if abs(r["thr"] - 0.2) < 1e-9][0]
        b = [r for r in rows_old if abs(r["thr"] - 0.2) < 1e-9][0]
        print(f"\n=== at the deployment threshold 0.2 ===")
        print(f"  combined : R {a['rec']:.4f}  keeps {100*a['kept']:.2f}%  F1 {a['f1']:.4f}"
              f"  circ {circ_new:.4f} rad")
        print(f"  original : R {b['rec']:.4f}  keeps {100*b['kept']:.2f}%  F1 {b['f1']:.4f}"
              f"  circ {circ_old:.4f} rad")
