"""Train the combined position+orientation model (XYYawCNN) on the SST dataset.

What this fixes relative to training_script_xy_artifacts.py / training_script_yaw.py:

1. ONE model instead of two. Their trunks were identical, so half the encoder compute
   was duplicated at every planning cycle.
2. Class imbalance. Only ~0.6% of cells are near-optimal positions (1:162). Plain
   nn.BCELoss() drives the model to predict "empty" everywhere -- 99.4% pixel accuracy
   and useless recall. Here: BCEWithLogits with a data-derived pos_weight, plus soft
   Dice, and the metrics reported are F1/IoU/recall, never pixel accuracy.
3. The orientation loss was UNMASKED. 99.2% of orientation cells are the "no data"
   sentinel, so CircularLossMod spent ~99% of its gradient teaching the net to emit the
   sentinel. Here the circular loss is masked to cells that actually carry an angle.
4. Angle wraparound. Regressing the angle directly puts a discontinuity in the middle of
   the target range; the model predicts (cos, sin) and atan2 recovers it.
5. Optimiser. SGD(lr=0.1), no momentum, no schedule, 500-2000 epochs, no validation and
   no best-checkpoint selection. Here: AdamW + warmup + cosine, AMP, early stopping on a
   held-out validation split, and the best checkpoint is the one that gets saved.
6. Reproducibility. random_split() was unseeded, so the "test" set differed every run.
   Split is seeded and the three subsets are disjoint by construction.
7. Preprocessing was per-epoch Python double loops over 100x100 images. Done once here
   and cached to an .npz.

Usage:  python3 training_script_xy_yaw.py --dataset DIR [--epochs N] [--quick]
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_model import XYYawCNN  # noqa: E402

# relative to the working directory; main() overrides them from --dataset/--cache/--out-dir
ROOT = Path("dataset")
RAW_DIR = ROOT / "test"                 # model input grids
XY_DIR = ROOT / "train" / "path"        # position ground truth
YAW_DIR = ROOT / "train" / "orientation"  # orientation ground truth
CACHE = Path("dataset_cache.npz")
OUT_DIR = Path("trained")

SEED = 0
TWO_PI = 2.0 * math.pi
DEPLOY_THRESHOLD = 0.2                  # checkXYInference() in the sampler
KEEP_BUDGET = 0.03                      # max fraction of the grid the filter may keep


# =====================================================================
# PREPROCESSING  (reproduces utils.get_pdf_images / utils.get_yaw_image)
# =====================================================================
def build_yaw_target(v):
    """utils.get_yaw_image, vectorised over the ~76 labelled cells instead of 10000.

    Cells with 6.8 < v < 7.2 are the "no data" sentinel. Every other cell carries an
    angle, which is written into its 5x5 neighbourhood (radius 2) in row-major order,
    last write wins, and then restored exactly at the original cell.
    """
    valid = v < 6.8
    rr, cc = np.nonzero(valid)
    ang = np.zeros_like(v)
    mask = np.zeros(v.shape, dtype=bool)
    H, W = v.shape
    for r, c in zip(rr, cc):                       # row-major, matches the original
        r0, r1 = max(0, r - 2), min(H, r + 3)
        c0, c1 = max(0, c - 2), min(W, c + 3)
        ang[r0:r1, c0:c1] = v[r, c]
        mask[r0:r1, c0:c1] = True
    ang[rr, cc] = v[rr, cc]                        # exact values at the true cells
    return ang.astype(np.float32), mask


def preprocess():
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        print(f"loaded cache {CACHE} ({len(d['inp'])} samples)")
        return d

    names = sorted(os.listdir(RAW_DIR))
    names = [n for n in names if n.endswith(".png")]
    inp = np.zeros((len(names), 100, 100), np.uint8)
    xy = np.zeros((len(names), 100, 100), np.uint8)
    yaw = np.zeros((len(names), 100, 100), np.float32)
    ymask = np.zeros((len(names), 100, 100), bool)
    so = np.zeros(len(names), np.float32)
    go = np.zeros(len(names), np.float32)

    t0 = time.time()
    for i, n in enumerate(names):
        # input grid: classes 0..5
        a = cv2.imread(str(RAW_DIR / n), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        inp[i] = np.rint(a / 255.0 * 6.0).astype(np.uint8)

        # position target: value 4 marks the near-optimal solution region
        b = cv2.imread(str(XY_DIR / n), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0 * 6.0
        xy[i] = ((b > 3.8) & (b < 4.2)).astype(np.uint8)

        # orientation target
        c = cv2.imread(str(YAW_DIR / (n[:-4] + "yaw.png")), cv2.IMREAD_GRAYSCALE)
        c = c.astype(np.float32) / 255.0 * 8.0
        yaw[i], ymask[i] = build_yaw_target(c)

        # start/goal yaw come from the filename, exactly as utils.get_images_and_params
        parts = n.split("_")
        so[i] = float(parts[2])
        go[i] = float(parts[-1][:-4])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(names)}  ({time.time()-t0:.0f}s)")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, inp=inp, xy=xy, yaw=yaw, ymask=ymask,
                        so=so, go=go, names=np.array(names))
    print(f"cached {len(names)} samples to {CACHE} ({time.time()-t0:.0f}s)")
    return np.load(CACHE, allow_pickle=True)


class GridDataset(Dataset):
    def __init__(self, d, idx, dilate=0, augment=False):
        # 8-fold dihedral augmentation. These are grids, so 90-degree rotations and a flip
        # are exact symmetries of the problem -- but only if every angle rotates with the
        # grid. The stored angles use x=row, y=col (verified against the path tangents),
        # so np.rot90(k) is a +k*pi/2 rotation in (x,y) and a row-flip maps theta to
        # pi-theta. Start/goal orientations and the orientation target must move too.
        self.augment = augment
        # dilate>0 thickens the POSITION target during training only. The stored target is
        # the 1px curve of one SST draw, but the quantity the sampler needs is the REGION
        # where near-optimal paths exist -- re-running SST on the same scene yields a
        # different curve. Training on the bare curve makes the model sharpen onto one
        # random draw (recall collapses as it fits harder); training on a corridor teaches
        # the region. Evaluation always uses the untouched ground truth.
        self.dilate = dilate
        self.inp, self.xy, self.yaw, self.ymask = d["inp"], d["xy"], d["yaw"], d["ymask"]
        self.so, self.go = d["so"], d["go"]
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        inp_t, xy_t = self.inp[j], self.xy[j]
        yaw_t, ym_t = self.yaw[j], self.ymask[j]
        so_t, go_t = float(self.so[j]), float(self.go[j])
        if self.augment:
            k = int(np.random.randint(4))
            flip = bool(np.random.randint(2))
            if k:
                inp_t = np.rot90(inp_t, k); xy_t = np.rot90(xy_t, k)
                yaw_t = np.rot90(yaw_t, k); ym_t = np.rot90(ym_t, k)
                delta = k * (math.pi / 2.0)
                yaw_t = np.where(ym_t, (yaw_t + delta) % (2 * math.pi), 0.0).astype(np.float32)
                so_t = (so_t + delta) % (2 * math.pi)
                go_t = (go_t + delta) % (2 * math.pi)
            if flip:
                inp_t = inp_t[::-1]; xy_t = xy_t[::-1]
                yaw_t = yaw_t[::-1]; ym_t = ym_t[::-1]
                yaw_t = np.where(ym_t, (math.pi - yaw_t) % (2 * math.pi), 0.0).astype(np.float32)
                so_t = (math.pi - so_t) % (2 * math.pi)
                go_t = (math.pi - go_t) % (2 * math.pi)
            inp_t = np.ascontiguousarray(inp_t); xy_t = np.ascontiguousarray(xy_t)
            yaw_t = np.ascontiguousarray(yaw_t); ym_t = np.ascontiguousarray(ym_t)
        if self.dilate:
            k = 2 * self.dilate + 1
            xy_t = torch.nn.functional.max_pool2d(
                torch.from_numpy(xy_t.astype(np.float32))[None, None], k, 1, self.dilate
            )[0, 0].numpy()
        # channel 0 carries the classes, channels 1-2 stay zero: exactly the tensor
        # gridMapToTensor() hands the model in C++
        img = torch.zeros(3, 100, 100)
        img[0] = torch.from_numpy(inp_t.astype(np.float32))
        return (img,
                torch.from_numpy(xy_t.astype(np.float32)).unsqueeze(0),
                torch.from_numpy(yaw_t).unsqueeze(0),
                torch.from_numpy(ym_t).unsqueeze(0),
                torch.tensor([0.35], dtype=torch.float32),
                torch.tensor([so_t], dtype=torch.float32),
                torch.tensor([go_t], dtype=torch.float32))


# =====================================================================
# LOSSES
# =====================================================================
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1.0 - num / den).mean()


def masked_angle_loss(cos_t, sin_t, target_angle, mask):
    """1 - cos(pred - gt) on labelled cells only. Smooth, wrap-free, minimised at 0.

    Plus a weak unit-norm pull so (cos, sin) stays on the circle and atan2 is well
    conditioned.
    """
    n = mask.sum().clamp_min(1.0)
    norm = torch.sqrt(cos_t * cos_t + sin_t * sin_t + 1e-8)
    c, s = cos_t / norm, sin_t / norm
    cg, sg = torch.cos(target_angle), torch.sin(target_angle)
    ang = ((1.0 - (c * cg + s * sg)) * mask).sum() / n
    unit = (((norm - 1.0) ** 2) * mask).sum() / n
    return ang + 0.1 * unit


def circular_error(pred_angle, target_angle, mask):
    """The paper's metric (eq. 2-3), in radians, over labelled cells."""
    d = torch.remainder(torch.abs(pred_angle - target_angle), TWO_PI)
    d = torch.where(d > math.pi, TWO_PI - d, d)
    n = mask.sum().clamp_min(1.0)
    return float((d * mask).sum() / n)


# =====================================================================
# METRICS
# =====================================================================
@torch.no_grad()
def evaluate(model, loader, device, thresholds=(0.2, 0.3, 0.5)):
    model.eval()
    tp = {t: 0.0 for t in thresholds}
    fp = {t: 0.0 for t in thresholds}
    fn = {t: 0.0 for t in thresholds}
    kept = {t: 0.0 for t in thresholds}
    total = 0.0
    ang_sum, ang_n = 0.0, 0.0
    probs, tgts = [], []
    for img, xy, yaw, ym, rs, so, go in loader:
        img, xy, yaw, ym = img.to(device), xy.to(device), yaw.to(device), ym.to(device)
        rs, so, go = rs.to(device), so.to(device), go.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, c, s = model.raw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
        logits, c, s = logits.float(), c.float(), s.float()
        p = torch.sigmoid(logits)
        probs.append(p.detach().cpu()); tgts.append(xy.detach().cpu())
        for t in thresholds:
            b = (p > t).float()
            tp[t] += float((b * xy).sum())
            fp[t] += float((b * (1 - xy)).sum())
            fn[t] += float(((1 - b) * xy).sum())
            kept[t] += float(b.sum())
        total += float(xy.numel())
        m = ym.float()
        pred_ang = torch.remainder(torch.atan2(s, c), TWO_PI)
        ang_sum += circular_error(pred_ang, yaw, m) * float(m.sum())
        ang_n += float(m.sum())

    out = {}
    for t in thresholds:
        prec = tp[t] / max(tp[t] + fp[t], 1e-9)
        rec = tp[t] / max(tp[t] + fn[t], 1e-9)
        out[f"f1@{t}"] = 2 * prec * rec / max(prec + rec, 1e-9)
        out[f"prec@{t}"] = prec
        out[f"rec@{t}"] = rec
        out[f"iou@{t}"] = tp[t] / max(tp[t] + fp[t] + fn[t], 1e-9)
        out[f"kept@{t}"] = kept[t] / max(total, 1e-9)
    out["circ_err"] = ang_sum / max(ang_n, 1e-9)
    # recall when the threshold is set so the model keeps exactly KEEP_BUDGET of the grid
    allp = torch.cat(probs).flatten()
    thr = torch.quantile(allp[torch.randperm(allp.numel())[:2_000_000]], 1.0 - KEEP_BUDGET)
    tgt = torch.cat(tgts).flatten()
    keep = allp > thr
    out["rec@budget"] = float((keep.float() * tgt).sum() / tgt.sum().clamp_min(1))
    return out


# =====================================================================
# TRAIN
# =====================================================================
def main():
    global RAW_DIR, XY_DIR, YAW_DIR, CACHE, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT,
                    help="dataset root holding test/, train/path/ and train/orientation/")
    ap.add_argument("--cache", type=Path, default=CACHE,
                    help="preprocessed dataset, built from --dataset on the first run")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where the best checkpoint and the TorchScript model are saved")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    ap.add_argument("--dice", type=int, default=1, help="add soft Dice to the position loss")
    ap.add_argument("--posweight", type=int, default=1, help="weight positives by neg/pos in BCE")
    ap.add_argument("--tag", type=str, default="", help="suffix for the checkpoint name")
    ap.add_argument("--distill", type=float, default=0.0,
                    help="weight on matching the ORIGINAL two models' outputs. They already "
                         "generalise well (zero train/test gap); distilling transfers that "
                         "into one network with two heads instead of two full trunks.")
    ap.add_argument("--head", type=str, default="simple", choices=["simple", "deep"],
                    help="decoder capacity: 'simple' is the original one-conv head")
    ap.add_argument("--opt", type=str, default="adamw", choices=["adamw", "sgd"],
                    help="sgd reproduces the original recipe (lr 0.1, no momentum), whose "
                         "large steps act as strong implicit regularisation")
    ap.add_argument("--heads", type=str, default="both", choices=["both", "pos", "yaw"],
                    help="train both heads (shared trunk) or a single head, to test whether "
                         "sharing a trunk between the classification and regression tasks costs accuracy")
    ap.add_argument("--augment", type=int, default=1,
                    help="8-fold dihedral augmentation on the training split")
    ap.add_argument("--dilate", type=int, default=0,
                    help="thicken the position target by this radius during TRAINING only")
    args = ap.parse_args()
    RAW_DIR = args.dataset / "test"
    XY_DIR = args.dataset / "train" / "path"
    YAW_DIR = args.dataset / "train" / "orientation"
    CACHE, OUT_DIR = args.cache, args.out_dir

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d = preprocess()
    n = len(d["inp"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_tr, n_va = int(0.8 * n), int(0.1 * n)
    tr_idx, va_idx, te_idx = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
    if args.quick:
        tr_idx, va_idx, te_idx = tr_idx[:128], va_idx[:64], te_idx[:64]
        args.epochs = min(args.epochs, 3)
    print(f"samples {n}  train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")

    mk = lambda idx, sh: DataLoader(GridDataset(d, idx, args.dilate if sh else 0,
                                                augment=sh and bool(args.augment)),
                                    batch_size=args.batch, shuffle=sh, num_workers=4,
                                    pin_memory=True, drop_last=sh)
    train_loader, val_loader, test_loader = mk(tr_idx, True), mk(va_idx, False), mk(te_idx, False)

    tgt_tr = d["xy"][tr_idx]
    if args.dilate:
        k = 2 * args.dilate + 1
        tgt_tr = torch.nn.functional.max_pool2d(
            torch.from_numpy(tgt_tr.astype(np.float32))[:, None], k, 1, args.dilate
        )[:, 0].numpy()
    pos = float(tgt_tr.sum())
    neg = float(tgt_tr.size - pos)
    pos_weight = torch.tensor([neg / pos], device=device) if args.posweight else None
    print(f"positive cells {100*pos/(pos+neg):.3f}%  ->  pos_weight "
          f"{float(pos_weight) if pos_weight is not None else 'off'}  dice {bool(args.dice)}")

    model = XYYawCNN(pretrained=True, head=args.head).to(device)

    teach_xy = teach_yaw = None
    if args.distill > 0:
        teach_xy = torch.jit.load(
            os.path.expanduser("~/trained_model/model_xy_0.0001_final.pth"), map_location=device).eval()
        teach_yaw = torch.jit.load(
            os.path.expanduser("~/trained_model/model_yaw_0.1_final_artifact_mse.pth"),
            map_location=device).eval()
        for p_ in list(teach_xy.parameters()) + list(teach_yaw.parameters()):
            p_.requires_grad_(False)
        print(f"distilling from the original two models, weight {args.distill}")
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.opt == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(train_loader)) * args.epochs
    warm = max(1, int(0.03 * steps))
    if args.opt == "sgd":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1.0)   # flat, as before
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: k / warm if k < warm else
            0.5 * (1 + math.cos(math.pi * (k - warm) / max(1, steps - warm))))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best, best_ep, bad = -1e9, -1, 0
    for ep in range(args.epochs):
        model.train()
        tot = tot_p = tot_y = 0.0
        t0 = time.time()
        for img, xy, yaw, ym, rs, so, go in train_loader:
            img, xy, yaw, ym = img.to(device), xy.to(device), yaw.to(device), ym.to(device)
            rs, so, go = rs.to(device), so.to(device), go.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, c, s = model.raw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1))
            logits, c, s = logits.float(), c.float(), s.float()
            l_pos = bce(logits, xy) + (dice_loss(logits, xy) if args.dice else 0.0)
            l_yaw = masked_angle_loss(c, s, yaw, ym.float())
            if args.distill > 0:
                with torch.no_grad():
                    t_xy = teach_xy(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1)).float()
                    t_yaw = teach_yaw(img, rs.squeeze(1), so.squeeze(1), go.squeeze(1)).float()
                    t_ang = torch.remainder(t_yaw - 1.0, TWO_PI)
                    t_valid = (t_yaw >= 1.0).float()
                # match the teacher's probability map, and its angles where it has any
                l_pos = l_pos + args.distill * F.binary_cross_entropy_with_logits(logits, t_xy)
                l_yaw = l_yaw + args.distill * masked_angle_loss(c, s, t_ang, t_valid)
            if args.heads == "pos":
                loss = l_pos
            elif args.heads == "yaw":
                loss = l_yaw
            else:
                loss = l_pos + l_yaw
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss); tot_p += float(l_pos); tot_y += float(l_yaw)

        nb = max(1, len(train_loader))
        m = evaluate(model, val_loader, device)
        # The sampler's job is to keep the near-optimal region while shrinking the space
        # it samples from, so select on RECALL subject to a retention budget rather than
        # on F1. F1 penalises breadth that costs the sampler almost nothing, and picking
        # on it lands the model at a precision-leaning operating point the planner does
        # not want.
        # Recall and retention move in OPPOSITE directions over training, so neither is a
        # valid selection signal on its own: early epochs win on recall (while keeping
        # most of the grid) and late epochs win on retention (having thrown recall away).
        # Score at a matched retention budget instead, which is comparable across epochs.
        score = m["rec@budget"]
        flag = ""
        if score > best:
            best, best_ep, bad = score, ep, 0
            torch.save(model.state_dict(), OUT_DIR / f"xy_yaw_best{args.tag}.pth")
            flag = "  *best"
        else:
            bad += 1
        print(f"ep {ep+1:3d}/{args.epochs}  loss {tot/nb:.4f} (pos {tot_p/nb:.4f} "
              f"yaw {tot_y/nb:.4f})  val R@0.2 {m['rec@0.2']:.4f} kept {100*m['kept@0.2']:.2f}% "
              f"R@{100*KEEP_BUDGET:.1f}%kept {score:.4f}  "
              f"circ_err {m['circ_err']:.4f} rad  {time.time()-t0:.0f}s{flag}")
        if bad >= args.patience:
            print(f"early stop: no val improvement for {args.patience} epochs")
            break

    model.load_state_dict(torch.load(OUT_DIR / f"xy_yaw_best{args.tag}.pth", map_location=device))
    print(f"\nbest epoch {best_ep+1} (val recall@0.2 {best:.4f} within a {100*KEEP_BUDGET:.0f}% retention budget)")
    print("\n================ TEST (held-out) ================")
    m = evaluate(model, test_loader, device, thresholds=(0.1, 0.2, 0.3, 0.5, 0.7))
    for t in (0.1, 0.2, 0.3, 0.5, 0.7):
        star = "   <- deployment threshold" if t == DEPLOY_THRESHOLD else ""
        print(f"  thr {t:.1f}  F1 {m[f'f1@{t}']:.4f}  IoU {m[f'iou@{t}']:.4f}  "
              f"P {m[f'prec@{t}']:.4f}  R {m[f'rec@{t}']:.4f}  "
              f"kept {100*m[f'kept@{t}']:.2f}%{star}")
    print(f"  orientation mean circular error: {m['circ_err']:.4f} rad "
          f"({math.degrees(m['circ_err']):.1f} deg) on labelled cells")

    model.eval().cpu()
    scripted = torch.jit.script(model)
    out = OUT_DIR / f"model_xy_yaw{args.tag}.pt"
    scripted.save(str(out))
    print(f"\nsaved scripted model -> {out}")


if __name__ == "__main__":
    main()
