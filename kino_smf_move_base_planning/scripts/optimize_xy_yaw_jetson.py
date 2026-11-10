"""Jetson (aarch64 / Orin) build of the combined XYYawCNN TensorRT engine: static batch 1,
saved as TorchScript so the C++ planner loads it with torch::jit::load unchanged.

Same export as optimize_xy_yaw.py, rebuilt on the Jetson itself: a TensorRT engine only
runs on the GPU model, CPU architecture and TensorRT version that built it, so the x86
(RTX 4090, TensorRT 10.14) engine crashes the planner on load here. Differences from the
x86 script:
  - FP16 goes through Torch-TensorRT's rule-based autocast. TensorRT 11 only builds
    strongly typed networks, so Torch-TensorRT 2.14 dropped enabled_precisions /
    use_explicit_typing and silently ignores them (optimize_xy_yaw.py would build an FP32
    engine here). The calibration samples let autocast keep nodes whose activations leave
    the FP16-safe range in FP32, and the outputs are cast back to FP32 so the planner
    reads the same dtype as before.
  - The held-out dataset only exists on the training machine. With --dataset-cache the
    check runs on real samples exactly like optimize_xy_yaw.py; without it, on synthetic
    local grids encoded the way getLocalInputs() builds them in C++.

Run:  python3 optimize_xy_yaw_jetson.py
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_tensorrt
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn_model import XYYawCNN  # noqa: E402

MODELS = Path.home() / "kino_smf_models"

ap = argparse.ArgumentParser(description="Jetson batch-1 TensorRT export of the combined XYYawCNN.")
ap.add_argument("--ckpt", type=Path, default=MODELS / "xy_yaw_best_final.pth")
ap.add_argument("--out", type=Path, default=MODELS / "model_xy_yaw_trt_fp16.ts")
ap.add_argument("--head", default="simple")
ap.add_argument("--precision", default="fp16", choices=["fp32_strict", "tf32", "fp16"],
                help="fp32_strict disables tensor cores entirely; tf32/fp16 are what make TRT "
                     "worth using on this ResNet50 trunk")
ap.add_argument("--dataset-cache", type=Path, default=None,
                help="dataset_cache.npz from training_script_xy_yaw.py: calibrate and verify on "
                     "real samples instead of synthetic grids")
ap.add_argument("--samples", type=int, default=60, help="Samples used for the verification.")
args = ap.parse_args()


def synthetic_samples(n, seed):
    """Local grids as getLocalInputs() builds them: 10 m x 10 m at 0.1 m around the robot,
    0 free, 1 obstacle, 5 agent, 2 start at the centre, 3 goal; robot radius 0.35 and
    start/goal yaw in [0, 2pi). Channels 1-2 stay zero, as in gridMapToTensor()."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        g = np.zeros((100, 100), np.float32)
        for _ in range(rng.integers(2, 6)):  # wall segments
            r, c = rng.integers(0, 100, 2)
            length, thick = rng.integers(15, 70), rng.integers(1, 3)
            if rng.random() < 0.5:
                g[r:r + thick, c:c + length] = 1.0
            else:
                g[r:r + length, c:c + thick] = 1.0
        for _ in range(rng.integers(0, 6)):  # furniture
            r, c = rng.integers(0, 95, 2)
            h, w = rng.integers(2, 10, 2)
            g[r:r + h, c:c + w] = 1.0
        g[44:57, 44:57] = 0.0  # the robot stands in free space
        for _ in range(rng.integers(0, 6)):  # an agent marks a single cell
            r, c = rng.integers(0, 100, 2)
            if g[r, c] == 0.0:
                g[r, c] = 5.0
        g[50, 50] = 2.0
        while True:  # goal 1.5-4.5 m away, on a free cell
            r, c = rng.integers(3, 97, 2)
            if g[r, c] == 0.0 and 15 <= math.hypot(r - 50, c - 50) <= 45:
                g[r, c] = 3.0
                break
        img = torch.zeros(3, 100, 100)
        img[0] = torch.from_numpy(g)
        samples.append((img,
                        torch.tensor([0.35], dtype=torch.float32),
                        torch.tensor([rng.uniform(0.0, 2 * math.pi)], dtype=torch.float32),
                        torch.tensor([rng.uniform(0.0, 2 * math.pi)], dtype=torch.float32)))
    return samples


def real_samples(n, split):
    """Held-out ("test") or training ("train") samples, split as in optimize_xy_yaw.py."""
    import training_script_xy_yaw as T

    T.CACHE = args.dataset_cache
    d = T.preprocess()
    total = len(d["inp"])
    perm = np.random.default_rng(T.SEED).permutation(total)
    idx = perm[int(0.8 * total) + int(0.1 * total):] if split == "test" else perm[:int(0.8 * total)]
    ds = T.GridDataset(d, idx)
    return [(img, rs, so, go) for img, _, _, _, rs, so, go in (ds[i] for i in range(min(n, len(ds))))]


if args.dataset_cache:
    calib_samples, check_samples = real_samples(16, "train"), real_samples(args.samples, "test")
    check_name = "real held-out samples"
else:
    calib_samples, check_samples = synthetic_samples(16, seed=1), synthetic_samples(args.samples, seed=2)
    check_name = "synthetic local grids"

class FP32Outputs(torch.nn.Module):
    """Autocast leaves the last nodes (sigmoid, where) in FP16; the planner expects FP32."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, robot_size, start_orientation, goal_orientation):
        xy, yaw = self.model(image, robot_size, start_orientation, goal_orientation)
        return xy.float(), yaw.float()


model = XYYawCNN(pretrained=False, head=args.head)
model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
model.eval().cuda()

example = (
    torch.zeros(1, 3, 100, 100).cuda(),
    torch.zeros(1, 1).cuda(),
    torch.zeros(1, 1).cuda(),
    torch.zeros(1, 1).cuda(),
)

exported = torch.export.export(FP32Outputs(model) if args.precision == "fp16" else model, example)

trt_inputs = [
    torch_tensorrt.Input(shape=(1, 3, 100, 100), dtype=torch.float32),
    torch_tensorrt.Input(shape=(1, 1), dtype=torch.float32),
    torch_tensorrt.Input(shape=(1, 1), dtype=torch.float32),
    torch_tensorrt.Input(shape=(1, 1), dtype=torch.float32),
]

kw = dict(inputs=trt_inputs, min_block_size=1)
if args.precision == "fp32_strict":
    kw.update(disable_tf32=True)
elif args.precision == "fp16":
    # each calibration batch is the 4 model inputs at batch 1, as the planner calls it
    calib_loader = DataLoader([tuple(t.cuda() for t in s) for s in calib_samples], batch_size=1)
    kw.update(enable_autocast=True,
              autocast_low_precision_type=torch.float16,
              # Torch-TensorRT 2.14 autocast records the explicit .float() casts (the class
              # one-hot in XYYawCNN._trunk, the output casts above) as FP16 although they
              # still produce FP32, so it skips the casts the next nodes need and the build
              # fails at the first conv after them. Keeping aten.to.dtype in FP32 makes the
              # recorded dtype true again, and real casts get inserted around those nodes.
              # The rule matches qualified-name strings, not op objects.
              autocast_excluded_ops={"torch.ops.aten.to.dtype"},
              autocast_calibration_dataloader=calib_loader)
# tf32: the defaults already build FP32 with TF32 tensor cores
trt_gm = torch_tensorrt.dynamo.compile(exported, **kw)

# If the graph fragments, most of the ResNet50 runs in PyTorch and TRT buys nothing.
n_engines = sum(1 for n in trt_gm.graph.nodes
                if n.op == "call_module" and str(n.target).startswith("_run_on_acc"))
n_fallback = sum(1 for n in trt_gm.graph.nodes
                 if n.op == "call_module" and str(n.target).startswith("_run_on_gpu"))
print(f"TRT engines: {n_engines}   PyTorch fallback blocks: {n_fallback}")

torch_tensorrt.save(trt_gm, str(args.out), output_format="torchscript", inputs=list(example))
print("saved:", args.out)

# ---------------------------------------------------------------- verify against eager
reloaded = torch.jit.load(str(args.out)).cuda().eval()

worst_xy = worst_yaw = 0.0
worst_kept = 0.0
kept_cells = []
with torch.no_grad():
    for img, rs, so, go in check_samples:
        ins = (img[None].cuda(), rs[None].cuda(), so[None].cuda(), go[None].cuda())
        a_xy, a_yaw = reloaded(*ins)
        b_xy, b_yaw = model(*ins)
        worst_xy = max(worst_xy, float((a_xy - b_xy).abs().max()))
        # compare yaw only where BOTH consider the cell valid; the gate is a hard
        # threshold on xy, so a cell sitting right at 0.2 can flip side on tiny drift
        m = ((a_yaw >= 1.0) & (b_yaw >= 1.0)).float()
        if float(m.sum()):
            dd = torch.remainder((a_yaw - b_yaw).abs(), 2 * math.pi)
            dd = torch.minimum(dd, 2 * math.pi - dd) * m  # circular, 0 == 2pi
            worst_yaw = max(worst_yaw, float(dd.max()))
        # how often does the deployment gate disagree at all
        ga, gb = (a_xy > 0.2), (b_xy > 0.2)
        worst_kept = max(worst_kept, float((ga ^ gb).float().mean()))
        kept_cells.append(int(gb.sum()))

print(f"verified on {len(check_samples)} {check_name}:")
print(f"  output dtypes: xy {a_xy.dtype}, yaw {a_yaw.dtype} (planner expects torch.float32)")
# near-empty eager maps would make the comparison meaningless, so show they are not
print(f"  eager keeps > 0.2 on {sum(k > 0 for k in kept_cells)}/{len(kept_cells)} samples "
      f"(median {int(np.median(kept_cells))} cells of 10000)")
print(f"  max |dxy|  vs eager: {worst_xy:.3e}")
print(f"  max |dyaw| vs eager (both-valid cells): {worst_yaw:.3e}")
print(f"  worst gate disagreement at the 0.2 threshold: {100*worst_kept:.4f}% of cells")

img, rs, so, go = check_samples[0]
ins = (img[None].cuda(), rs[None].cuda(), so[None].cuda(), go[None].cuda())
with torch.no_grad():
    for fn, name in ((model, "eager   "), (reloaded, "tensorrt")):
        for _ in range(20):
            fn(*ins)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            fn(*ins)
        torch.cuda.synchronize()
        print(f"  {name} latency @batch1: {(time.perf_counter()-t0)/200*1e3:.3f} ms")
