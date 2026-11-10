"""Export the combined XYYawCNN as a batch-1 TensorRT engine, saved as TorchScript so
the C++ planner loads it with torch::jit::load unchanged.

The local planner calls this model once per planning cycle on a single grid map
(inputsToTensor() builds [1,3,100,100] plus three [1,1] scalars), so the engine is built
static at batch 1 rather than dynamic.

Verified against eager on real dataset samples, not random noise: the model correctly
emits near-zero probabilities on an unstructured grid, so random inputs would compare two
all-zero maps and prove nothing.

Run:  python3 optimize_xy_yaw.py
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_tensorrt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_model import XYYawCNN  # noqa: E402
import training_script_xy_yaw as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="trained/xy_yaw_best.pth")
ap.add_argument("--out", default="trained/model_xy_yaw_trt_fp16.ts")
ap.add_argument("--dataset-cache", type=Path, default=Path("dataset_cache.npz"),
                help="dataset_cache.npz from training_script_xy_yaw.py, to verify on held-out samples")
ap.add_argument("--head", default="simple")
ap.add_argument("--precision", default="fp16", choices=["fp32_strict", "tf32", "fp16"],
                help="fp32_strict disables tensor cores entirely and is SLOWER than eager "
                     "on this ResNet50 trunk; tf32/fp16 are what make TRT worth using here")
args = ap.parse_args()

model = XYYawCNN(pretrained=False, head=args.head)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
model.eval().cuda()

example = (
    torch.zeros(1, 3, 100, 100).cuda(),
    torch.zeros(1, 1).cuda(),
    torch.zeros(1, 1).cuda(),
    torch.zeros(1, 1).cuda(),
)

exported = torch.export.export(model, example)

trt_inputs = [
    torch_tensorrt.Input(shape=(1, 3, 100, 100), dtype=torch.float32),
    torch_tensorrt.Input(shape=(1, 1), dtype=torch.float32),
    torch_tensorrt.Input(shape=(1, 1), dtype=torch.float32),
    torch_tensorrt.Input(shape=(1, 1), dtype=torch.float32),
]

kw = dict(inputs=trt_inputs, min_block_size=1)
if args.precision == "fp32_strict":
    kw.update(use_explicit_typing=True, disable_tf32=True)
elif args.precision == "tf32":
    kw.update(use_explicit_typing=False, enabled_precisions={torch.float32})
else:
    kw.update(use_explicit_typing=False, enabled_precisions={torch.float16})
trt_gm = torch_tensorrt.dynamo.compile(exported, **kw)

# If the graph fragments, most of the ResNet50 runs in PyTorch and TRT buys nothing.
n_engines = sum(1 for n in trt_gm.graph.nodes
                if n.op == "call_module" and "engine" in str(n.target).lower()
                or (n.op == "call_module" and str(n.target).startswith("_run_on_acc")))
n_fallback = sum(1 for n in trt_gm.graph.nodes
                 if n.op == "call_module" and str(n.target).startswith("_run_on_gpu"))
print(f"TRT engines: {n_engines}   PyTorch fallback blocks: {n_fallback}")

torch_tensorrt.save(trt_gm, args.out, output_format="torchscript", inputs=list(example))
print("saved:", args.out)

# ---------------------------------------------------------------- verify on REAL data
reloaded = torch.jit.load(args.out).cuda().eval()
T.CACHE = args.dataset_cache
d = T.preprocess()
n = len(d["inp"])
perm = np.random.default_rng(T.SEED).permutation(n)
te = perm[int(0.8 * n) + int(0.1 * n):]
ds = T.GridDataset(d, te)

worst_xy = worst_yaw = 0.0
worst_kept = 0.0
with torch.no_grad():
    for i in range(60):
        img, xy_t, yaw_t, ym, rs, so, go = ds[i]
        ins = (img[None].cuda(), rs[None].cuda(), so[None].cuda(), go[None].cuda())
        a_xy, a_yaw = reloaded(*ins)
        b_xy, b_yaw = model(*ins)
        worst_xy = max(worst_xy, float((a_xy - b_xy).abs().max()))
        # compare yaw only where BOTH consider the cell valid; the gate is a hard
        # threshold on xy, so a cell sitting right at 0.2 can flip side on tiny drift
        m = ((a_yaw >= 1.0) & (b_yaw >= 1.0)).float()
        if float(m.sum()):
            dd = torch.remainder((a_yaw - b_yaw).abs(), 2 * math.pi)
            dd = torch.minimum(dd, 2 * math.pi - dd) * m   # circular, 0 == 2pi
            worst_yaw = max(worst_yaw, float(dd.max()))
        # how often does the deployment gate disagree at all
        ga, gb = (a_xy > 0.2), (b_xy > 0.2)
        worst_kept = max(worst_kept, float((ga ^ gb).float().mean()))

print(f"verified on 60 real held-out samples:")
print(f"  max |dxy|  vs eager: {worst_xy:.3e}")
print(f"  max |dyaw| vs eager (both-valid cells): {worst_yaw:.3e}")
print(f"  worst gate disagreement at the 0.2 threshold: {100*worst_kept:.4f}% of cells")

img, *_ = ds[0]
ins = (img[None].cuda(), torch.tensor([[0.35]]).cuda(),
       torch.tensor([[1.2]]).cuda(), torch.tensor([[2.7]]).cuda())
with torch.no_grad():
    for fn, name in ((model, "eager  "), (reloaded, "tensorrt")):
        for _ in range(20):
            fn(*ins)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(200):
            fn(*ins)
        torch.cuda.synchronize()
        print(f"  {name} latency @batch1: {(time.perf_counter()-t0)/200*1e3:.3f} ms")
