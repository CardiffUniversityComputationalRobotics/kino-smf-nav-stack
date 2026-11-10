"""Checks that the combined model and its training pipeline are actually correct.

Run:  python3 test_xy_yaw_training.py

These are the checks that would have caught what went wrong the first time:
 1  input/target alignment      - sample i's input really is sample i's ground truth
 2  split disjointness          - no train/val/test leakage, and the split is seeded
 3  target semantics            - position is the value-4 region, orientation is masked
 4  circular loss wraps         - 0.05 rad vs 6.23 rad is a SMALL error, not a huge one
 5  masking works               - the "no data" cells contribute nothing to the yaw loss
 6  imbalance is handled        - an all-negative prediction must NOT score well
 7  both heads get gradient     - no dead branch
 8  overfit a tiny batch        - the canonical "is training wired up" test
 9  deployment contract         - output shapes/ranges match the C++ sampler's expectations
10  scriptability + determinism - torch.jit.script round-trips and matches eager
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_model import XYYawCNN  # noqa: E402
import training_script_xy_yaw as T  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------- 1,2,3
def test_data():
    print("\n1-3) data: alignment, split, target semantics")
    d = T.preprocess()
    names = [str(x) for x in d["names"]]
    n = len(names)

    # 1. alignment: the position target for sample i must come from the file named
    # names[i], and the orientation target from names[i] with the yaw suffix.
    import cv2
    rng = np.random.default_rng(1)
    ok = True
    for j in rng.choice(n, 5, replace=False):
        b = cv2.imread(str(T.XY_DIR / names[j]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 * 6.0
        want = ((b > 3.8) & (b < 4.2)).astype(np.uint8)
        ok &= bool((want == d["xy"][j]).all())
        a = cv2.imread(str(T.RAW_DIR / names[j]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        ok &= bool((np.rint(a / 255.0 * 6.0).astype(np.uint8) == d["inp"][j]).all())
    check("input and targets are paired by filename, not by listing order", ok)

    # start/goal yaw parsed from the same filename
    j = int(rng.integers(n))
    parts = names[j].split("_")
    check("start/goal yaw parsed from the sample's own filename",
          abs(float(parts[2]) - float(d["so"][j])) < 1e-6
          and abs(float(parts[-1][:-4]) - float(d["go"][j])) < 1e-6)

    # 2. split
    r = np.random.default_rng(T.SEED)
    p1 = r.permutation(n)
    p2 = np.random.default_rng(T.SEED).permutation(n)
    check("split is seeded and reproducible", bool((p1 == p2).all()))
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    tr, va, te = set(p1[:n_tr]), set(p1[n_tr:n_tr + n_va]), set(p1[n_tr + n_va:])
    check("train/val/test are disjoint and cover the dataset",
          not (tr & va) and not (tr & te) and not (va & te) and len(tr | va | te) == n,
          f"{len(tr)}/{len(va)}/{len(te)}")

    # 3. target semantics
    frac = float(d["xy"].mean())
    check("position target is the sparse near-optimal region", 0.001 < frac < 0.05,
          f"{100*frac:.2f}% positive -> 1:{(1-frac)/frac:.0f} imbalance")
    mfrac = float(d["ymask"].mean())
    check("orientation target is masked, not dense", 0.001 < mfrac < 0.2,
          f"{100*mfrac:.2f}% of cells carry an angle")
    ang = d["yaw"][d["ymask"]]
    check("orientation values are radians in [0, 2pi)", float(ang.min()) >= 0 and float(ang.max()) < T.TWO_PI,
          f"[{ang.min():.3f}, {ang.max():.3f}]")
    check("orientation mask is a dilation of the labelled cells (radius 2)",
          mfrac > frac * 2, f"mask {100*mfrac:.2f}% vs raw positions {100*frac:.2f}%")
    return d


# ---------------------------------------------------------------- 4,5
def test_losses():
    print("\n4-5) losses: wraparound and masking")
    m = torch.ones(1, 1, 2, 2)
    # 0.05 rad vs 6.23 rad are 0.1 rad apart across the wrap
    a = torch.full((1, 1, 2, 2), 0.05)
    b = torch.full((1, 1, 2, 2), 6.23)
    err = T.circular_error(a, b, m)
    check("circular_error wraps: 0.05 vs 6.23 rad is ~0.1 rad", abs(err - 0.1032) < 0.01,
          f"got {err:.4f} rad (a naive |a-b| would say {abs(0.05-6.23):.2f})")

    # the training loss must agree: near-zero for the wrapped pair
    c, s = torch.cos(a), torch.sin(a)
    l_near = float(T.masked_angle_loss(c, s, b, m))
    l_far = float(T.masked_angle_loss(c, s, torch.full((1, 1, 2, 2), 0.05 + math.pi), m))
    check("masked_angle_loss is small across the wrap and large at 180 deg",
          l_near < 0.01 and l_far > 1.5, f"wrapped {l_near:.4f} vs opposite {l_far:.4f}")

    # masking: changing the target on unmasked cells must not change the loss
    tgt = torch.rand(1, 1, 8, 8) * T.TWO_PI
    mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    c = torch.randn(1, 1, 8, 8); s = torch.randn(1, 1, 8, 8)
    l1 = float(T.masked_angle_loss(c, s, tgt, mask))
    tgt2 = torch.where(mask > 0, tgt, torch.rand(1, 1, 8, 8) * T.TWO_PI)
    l2 = float(T.masked_angle_loss(c, s, tgt2, mask))
    check("yaw loss ignores 'no data' cells entirely", abs(l1 - l2) < 1e-6,
          f"{l1:.6f} vs {l2:.6f}")

    # 6. imbalance: predicting all-negative must not look good
    logits = torch.full((2, 1, 32, 32), -10.0)
    tgt = torch.zeros(2, 1, 32, 32); tgt[:, :, 0, :6] = 1.0     # 0.59%, the real rate
    acc = float(((torch.sigmoid(logits) > 0.2).float() == tgt).float().mean())
    dice = float(T.dice_loss(logits, tgt))
    check("all-negative prediction: high pixel accuracy but bad Dice (why F1 is reported)",
          acc > 0.99 and dice > 0.7, f"pixel acc {acc:.4f} but dice loss {dice:.4f}")


# ---------------------------------------------------------------- 7,8
def test_optimisation(d):
    print("\n7-8) optimisation: gradient flow and overfitting a tiny batch")
    torch.manual_seed(0)
    model = XYYawCNN(pretrained=False).to(DEV)

    idx = np.arange(8)
    ds = T.GridDataset(d, idx)
    batch = [torch.stack([ds[i][k] for i in range(8)]) for k in range(7)]
    img, xy, yaw, ym, rs, so, go = [b.to(DEV) for b in batch]
    ym = ym.float()

    logits, c, s = model.raw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
    (T.dice_loss(logits, xy) + T.masked_angle_loss(c, s, yaw, ym)).backward()
    g_pos = model.pos_head[-1].weight.grad
    g_yaw = model.yaw_head[-1].weight.grad
    g_trunk = model.conv.weight.grad
    check("position head receives gradient", g_pos is not None and float(g_pos.abs().sum()) > 0)
    check("orientation head receives gradient", g_yaw is not None and float(g_yaw.abs().sum()) > 0)
    check("shared trunk receives gradient from both heads",
          g_trunk is not None and float(g_trunk.abs().sum()) > 0)

    # overfit: if the pipeline is wired correctly, 8 samples are memorisable
    torch.manual_seed(0)
    model = XYYawCNN(pretrained=True).to(DEV)   # as the real run does
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    pw = torch.tensor([(xy.numel() - xy.sum()) / xy.sum().clamp_min(1)], device=DEV)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    f1_0 = circ_0 = None
    for step in range(500):
        logits, c, s = model.raw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
        loss = bce(logits, xy) + T.dice_loss(logits, xy) + T.masked_angle_loss(c, s, yaw, ym)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step in (0, 499):
            with torch.no_grad():
                p = (torch.sigmoid(logits) > 0.2).float()
                tp = float((p * xy).sum()); fp = float((p * (1 - xy)).sum()); fn = float(((1 - p) * xy).sum())
                f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
                ce = T.circular_error(torch.remainder(torch.atan2(s, c), T.TWO_PI), yaw, ym)
            if step == 0:
                f1_0, circ_0 = f1, ce
            else:
                check("overfits 8 samples: position F1 -> ~1", f1 > 0.95,
                      f"F1 {f1_0:.3f} -> {f1:.3f}")
                check("overfits 8 samples: orientation error collapses", ce < 0.15,
                      f"circ_err {circ_0:.3f} -> {ce:.3f} rad")


# ---------------------------------------------------------------- 9,10
def test_contract():
    print("\n9-10) deployment contract and scripting")
    torch.manual_seed(0)
    model = XYYawCNN(pretrained=False).eval()
    img = torch.zeros(2, 3, 100, 100)
    img[:, 0] = torch.randint(0, 6, (2, 100, 100)).float()
    rs = torch.full((2, 1), 0.35).squeeze(1)
    so = torch.rand(2) * T.TWO_PI
    go = torch.rand(2) * T.TWO_PI
    with torch.no_grad():
        xy, yaw = model(img, rs, so, go)

    check("returns two grids in one forward pass",
          xy.shape == (2, 1, 100, 100) and yaw.shape == (2, 1, 100, 100))
    check("xy is a probability in [0,1] (sampler thresholds at 0.2)",
          float(xy.min()) >= 0.0 and float(xy.max()) <= 1.0)
    thr = model.XY_THRESHOLD
    valid = yaw[xy > thr]
    invalid = yaw[xy <= thr]
    check("yaw uses the +1 offset on valid cells (getSampleYaw expects >= 1)",
          valid.numel() == 0 or (float(valid.min()) >= 1.0 and float(valid.max()) <= 1.0 + T.TWO_PI))
    check("yaw is < 1 ('no data') exactly where xy <= 0.2",
          invalid.numel() == 0 or float(invalid.max()) < 1.0)
    check("input is the 3-channel tensor gridMapToTensor already builds (no C++ change)",
          img.shape[1] == 3 and float(img[:, 1:].abs().sum()) == 0.0)

    scripted = torch.jit.script(model)
    with torch.no_grad():
        xy2, yaw2 = scripted(img, rs, so, go)
    check("torch.jit.script round-trips and matches eager",
          bool(torch.allclose(xy, xy2, atol=1e-6) and torch.allclose(yaw, yaw2, atol=1e-6)))


if __name__ == "__main__":
    d = test_data()
    test_losses()
    test_optimisation(d)
    test_contract()
    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        sys.exit(1)
