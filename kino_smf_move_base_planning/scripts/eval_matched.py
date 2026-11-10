"""Compare models at MATCHED retention -- the only fair way to rank them.

Recall and retained-fraction trade off against each other, and different models are
calibrated differently, so comparing recall at a shared probability threshold compares
calibration as much as quality. Instead: for each model, pick the threshold that makes it
keep the same fraction of the grid, then compare recall there.

Run:  python3 eval_matched.py
"""
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
BUDGETS = (0.01, 0.02, 0.0271, 0.05)


@torch.no_grad()
def gather(predict, d, idx, bs=32):
    ds = T.GridDataset(d, idx)
    P, Y, A, G, M = [], [], [], [], []
    for i in range(0, len(ds), bs):
        items = [ds[j] for j in range(i, min(i + bs, len(ds)))]
        img = torch.stack([x[0] for x in items]).to(DEV)
        rs = torch.stack([x[4] for x in items]).to(DEV)
        so = torch.stack([x[5] for x in items]).to(DEV)
        go = torch.stack([x[6] for x in items]).to(DEV)
        p, ang = predict(img, rs, so, go)
        P.append(p.cpu()); A.append(ang.cpu())
        Y.append(torch.stack([x[1] for x in items]))
        G.append(torch.stack([x[2] for x in items]))
        M.append(torch.stack([x[3] for x in items]))
    return (torch.cat(P), torch.cat(Y), torch.cat(A), torch.cat(G), torch.cat(M).float())


def at_budget(P, Y, budget):
    """Threshold so exactly `budget` of all cells are kept, then measure recall."""
    thr = torch.quantile(P.flatten().float()[torch.randperm(P.numel())[:2_000_000]],
                         1.0 - budget)
    b = (P > thr).float()
    tp = float((b * Y).sum()); fn = float(((1 - b) * Y).sum()); fp = float((b * (1 - Y)).sum())
    return dict(thr=float(thr), rec=tp / max(tp + fn, 1e-9),
                prec=tp / max(tp + fp, 1e-9), kept=float(b.mean()))


if __name__ == "__main__":
    d = T.preprocess()
    n = len(d["inp"])
    perm = np.random.default_rng(T.SEED).permutation(n)
    te = perm[int(0.8 * n) + int(0.1 * n):]
    print(f"test samples {len(te)}   true positive rate {100*d['xy'][te].mean():.3f}%\n")

    models = {}
    for tag, ckpt in (("combined (no aug)", "xy_yaw_best_long.pth"),
                      ("combined (aug)", "xy_yaw_best_aug.pth"),
                      ("combined pos-head only", "xy_yaw_best_posonly.pth"),
                      ("combined SGD recipe", "xy_yaw_best_sgd.pth"),
                      ("combined simple+aug", "xy_yaw_best_simple.pth"),
                      ("combined distilled", "xy_yaw_best_kd.pth"),
                      ("combined 600ep final", "xy_yaw_best_final.pth")):
        pth = Path(os.path.expanduser("~/07_08_dataset/trained")) / ckpt
        if not pth.exists():
            continue
        head = "simple" if any(k in ckpt for k in ("simple", "kd", "final")) else "deep"
        m = XYYawCNN(pretrained=False, head=head).to(DEV).eval()
        m.load_state_dict(torch.load(pth, map_location=DEV))

        def mk(m):
            def f(img, rs, so, go):
                lg, c, s = m.raw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
                return torch.sigmoid(lg), torch.remainder(torch.atan2(s, c), T.TWO_PI)
            return f
        models[tag] = mk(m)

    old_xy = torch.jit.load(OLD_XY, map_location=DEV).eval()
    old_yaw = torch.jit.load(OLD_YAW, map_location=DEV).eval()

    def f_old(img, rs, so, go):
        p = old_xy(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
        y = old_yaw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
        return p, torch.remainder(y - 1.0, T.TWO_PI)
    models["original (2 nets)"] = f_old

    print(f"{'model':20s} " + "  ".join(f"R@{100*b:.2f}%kept" for b in BUDGETS) + "   circ_err")
    for name, fn in models.items():
        P, Y, A, G, M = gather(fn, d, te)
        cells = []
        for b in BUDGETS:
            cells.append(f"{at_budget(P, Y, b)['rec']:11.4f}")
        circ = T.circular_error(A, G, M)
        print(f"{name:20s} " + "  ".join(cells) + f"   {circ:.4f}")
