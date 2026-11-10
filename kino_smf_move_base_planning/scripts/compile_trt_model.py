#!/usr/bin/env python3
"""Compile the xy/yaw state-sampler TorchScript models with Torch-TensorRT.

The local planner's inference block spends ~100 ms per model per cycle in plain
TorchScript; the planner already links libtorchtrt_runtime, so a compiled model
loads with the existing torch::jit::load — no C++ changes, just swap the yaml
paths (position_model_path / orientation_model_path) to the new .ts files.

Usage:
    python3 compile_trt_model.py model_xy_0.0001_final.pth model_xy_trt.ts
    python3 compile_trt_model.py model_yaw_..._mse.pth model_yaw_trt.ts --fp32 --tol=0.15

--tol sets the max-abs-deviation gate (default 0.05, sized for the XY model's [0,1]
probabilities). The yaw model encodes angle+1 (range ~7) and TRT's resize/pooling
converters differ slightly at boundaries even in fp32, so ~0.08 absolute (~4.5 deg
on a sampling hint) is expected and harmless there.

Inputs match getLocalInputs()/inputsToTensor(): the map tensor is 3-channel
(gridMapToTensor stacks the map with two zero channels for the ResNet stem),
10 m at 0.1 m resolution -> [1,3,100,100], plus three [1,1] scalars
(robot radius, start yaw, goal yaw). Default precision fp16.

The .pth files are torch.jit.script()-ed modules with weights baked in
(training_script_xy_artifacts.py saves model_scripted), so no separate
state_dict loading is needed.
"""
import sys

import torch
import torch_tensorrt


def main():
    fp16 = "--fp32" not in sys.argv
    tol = 0.05
    for arg in sys.argv[1:]:
        if arg.startswith("--tol="):
            tol = float(arg.split("=", 1)[1])
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(paths) != 2:
        sys.exit(__doc__)
    src, dst = paths

    model = torch.jit.load(src).eval().cuda()

    example = [
        torch.rand(1, 3, 100, 100).cuda(),
        torch.tensor([[0.35]]).cuda(),
        torch.tensor([[1.0]]).cuda(),
        torch.tensor([[2.0]]).cuda(),
    ]

    # ir="torchscript": matches the module format the planner's libtorchtrt_runtime
    # loads via torch::jit::load, and avoids the (segfaulting) dynamo default path
    trt_model = torch_tensorrt.compile(
        model,
        ir="torchscript",
        inputs=[torch_tensorrt.Input(shape=list(t.shape)) for t in example],
        enabled_precisions={torch.half, torch.float} if fp16 else {torch.float},
        truncate_long_and_double=True,
    )

    # parity check: the sampler thresholds xy at 0.2, so small fp16 drift is fine,
    # but a large deviation means the compilation went wrong
    with torch.no_grad():
        ref = model(*example)
        out = trt_model(*example)
    diff = (ref - out).abs()
    print(f"abs diff vs original: max {diff.max().item():.4f} | mean {diff.mean().item():.5f}")
    assert diff.max().item() < tol, (
        f"deviation exceeds --tol={tol}; if already on --fp32 the gap is converter "
        f"semantics (resize/pooling), not precision - raise --tol if the consumer tolerates it")

    torch.jit.save(trt_model, dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    main()
