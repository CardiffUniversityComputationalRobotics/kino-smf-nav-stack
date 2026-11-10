"""Acceptance test for a TensorRT export of the combined model.

Per-cell max drift is the WRONG test: the yaw grid wraps at 0/2pi (so a raw |a-b| reports
~6.28 for two identical directions), and a single cell sitting exactly on the 0.2 gate can
flip on any rounding. What matters is whether the planner's own numbers move -- recall at
matched retention, and mean circular error on labelled cells.

Run:  python3 validate_trt_xy_yaw.py
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_tensorrt  # noqa: F401  (registers the TRT engine class for torch.jit.load)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_model import XYYawCNN  # noqa: E402
import training_script_xy_yaw as T  # noqa: E402
import eval_matched as E  # noqa: E402

TRAINED = Path(os.path.expanduser("~/07_08_dataset/trained"))
VARIANTS = {
    "eager fp32": None,
    "trt fp16": TRAINED / "model_xy_yaw_trt_fp16.ts",
    "trt tf32": TRAINED / "model_xy_yaw_trt_tf32.ts",
}


def main():
    d = T.preprocess()
    n = len(d["inp"])
    perm = np.random.default_rng(T.SEED).permutation(n)
    te = perm[int(0.8 * n) + int(0.1 * n):]
    ds = T.GridDataset(d, te)
    # d is a lazy NpzFile: d["yaw"] decompresses the whole 173MB array on EVERY access,
    # so pull it out once instead of indexing it inside the comprehension
    yaw_all = d["yaw"]
    gt_yaw = torch.from_numpy(yaw_all[te])[:, None]

    eager = XYYawCNN(pretrained=False, head="simple").to(E.DEV).eval()
    eager.load_state_dict(torch.load(TRAINED / "xy_yaw_best_final.pth", map_location=E.DEV))

    print(f"{'variant':12s} {'R@2.71%kept':>12s} {'circ_err':>10s} {'yaw cover':>10s}"
          f"   (test n={len(te)})")
    for name, path in VARIANTS.items():
        if path is not None and not path.exists():
            print(f"{name:12s} {'-- missing --':>12s}")
            continue
        m = eager if path is None else torch.jit.load(str(path)).cuda().eval()
        P, G, A, M, labelled = [], [], [], [], []
        with torch.no_grad():
            for i in range(len(ds)):
                img, xy_t, yaw_t, ym, rs, so, go = ds[i]
                xy, yaw = m(img[None].to(E.DEV), rs[None].to(E.DEV),
                            so[None].to(E.DEV), go[None].to(E.DEV))
                yc = yaw.float().cpu()
                P.append(xy.float().cpu()); G.append(xy_t[None])
                A.append(torch.remainder(yc - 1.0, 2 * math.pi))
                # score only where the model actually SUPPLIES an orientation. Gated
                # cells hold 0, which decodes to a meaningless 5.28 rad, so including
                # them measures the gate, not the orientation head.
                M.append(ym[None].float() * (yc >= 1.0).float())
                labelled.append(ym[None].float())
        P, G, A, M = torch.cat(P), torch.cat(G), torch.cat(A), torch.cat(M)
        r = E.at_budget(P, G, 0.0271)["rec"]
        cov = float(M.sum() / torch.cat(labelled).sum())
        print(f"{name:12s} {r:12.4f} {T.circular_error(A, gt_yaw, M):10.4f} "
              f"{100*cov:9.1f}%")


if __name__ == "__main__":
    main()
