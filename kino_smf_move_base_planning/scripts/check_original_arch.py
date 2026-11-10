"""Discriminating test: train the ORIGINAL XYCNN inside THIS pipeline.

Every lever on XYYawCNN has been ruled out, so the remaining possibilities are (a) my
model differs from the original in some way that matters, or (b) my data pipeline differs
from the one the original was trained on. Training the original architecture here splits
them:

  their recipe (SGD 0.1, plain BCELoss, no augmentation) is the closest reproduction of
  how model_xy_0.0001_final.pth was actually produced. If it reaches ~0.93 the pipeline is
  faithful and the fault is in XYYawCNN; if it lands ~0.70 the preprocessing differs and
  the model was never the problem.

Run:  python3 check_original_arch.py [--recipe theirs|mine] [--epochs N]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_model import XYCNN  # noqa: E402
from model_primitives import FullyConnectedNN  # noqa: E402
import training_script_xy_yaw as T  # noqa: E402
import eval_matched as E  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build():
    m = XYCNN()
    # the shared default is input_dim=10 (what the expansion models use); these scripts
    # pass 3 attributes, which is what it must have been when the original was trained
    m.fcn = FullyConnectedNN(input_dim=3)
    return m.to(DEV)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    P, Y = [], []
    for img, xy, yaw, ym, rs, so, go in loader:
        p = model(img.to(DEV), rs.to(DEV).squeeze(1), so.to(DEV).squeeze(1), go.to(DEV).squeeze(1))
        P.append(p.cpu()); Y.append(xy)
    return torch.cat(P), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="theirs", choices=["theirs", "mine"])
    ap.add_argument("--epochs", type=int, default=250)
    args = ap.parse_args()

    torch.manual_seed(T.SEED); np.random.seed(T.SEED)
    d = T.preprocess()
    n = len(d["inp"])
    perm = np.random.default_rng(T.SEED).permutation(n)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    tr, va, te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

    aug = args.recipe == "mine"
    train_loader = DataLoader(T.GridDataset(d, tr, augment=aug), batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(T.GridDataset(d, va), batch_size=32, num_workers=4)
    test_loader = DataLoader(T.GridDataset(d, te), batch_size=32, num_workers=4)

    model = build()
    if args.recipe == "theirs":
        opt = torch.optim.SGD(model.parameters(), lr=0.1)          # exactly as before
        pw = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        pos = float(d["xy"][tr].sum()); neg = float(d["xy"][tr].size - pos)
        pw = neg / pos
    print(f"recipe={args.recipe}  augment={aug}  pos_weight={pw}  epochs={args.epochs}")

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nb = 0
        t0 = time.time()
        for img, xy, yaw, ym, rs, so, go in train_loader:
            img, xy = img.to(DEV), xy.to(DEV)
            rs, so, go = rs.to(DEV), so.to(DEV), go.to(DEV)
            p = model(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1)).clamp(1e-6, 1 - 1e-6)
            if pw is None:
                loss = F.binary_cross_entropy(p, xy)
            else:
                w = torch.where(xy > 0.5, torch.full_like(xy, pw), torch.ones_like(xy))
                loss = F.binary_cross_entropy(p, xy, weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        P, Y = evaluate(model, val_loader)
        r = E.at_budget(P, Y, 0.0271)["rec"]
        flag = ""
        if r > best:
            best, bad = r, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  *best"
        else:
            bad += 1
        if ep % 10 == 0 or flag:
            print(f"ep {ep+1:3d}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
                  f"val R@2.71%kept {r:.4f}  {time.time()-t0:.0f}s{flag}")
        if bad >= 60:
            print("early stop"); break

    model.load_state_dict(best_state)
    P, Y = evaluate(model, test_loader)
    print(f"\n=== ORIGINAL XYCNN trained in THIS pipeline ({args.recipe} recipe) ===")
    for b in (0.01, 0.02, 0.0271, 0.05):
        s = E.at_budget(P, Y, b)
        print(f"  R@{100*b:5.2f}%kept  {s['rec']:.4f}")
    print("\n  reference -- shipped model_xy_0.0001_final.pth: R@2.71%kept 0.9263")
    print("  reference -- my XYYawCNN simple+aug:            R@2.71%kept 0.6979")
    torch.save(best_state, os.path.expanduser("~/07_08_dataset/trained/xycnn_check_%s.pth") % args.recipe)


if __name__ == "__main__":
    main()
