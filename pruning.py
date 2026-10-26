"""YOLOv8 C2f conversion and dependency-aware channel pruning.

Adapted from VainF/Torch-Pruning's MIT-licensed YOLOv8 example (Hyunseok Kim).
Source: https://github.com/VainF/Torch-Pruning/blob/v1.6.1/examples/yolov8/yolov8_pruning.py
"""
import bootstrap
import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics.nn.modules import C2f, Conv, Detect


class C2fSplit(nn.Module):
    """Equivalent split convolutions allow DepGraph to track channel dependencies."""

    def __init__(self, original):
        super().__init__()
        hidden = original.c
        self.cv0 = Conv(original.cv1.conv.in_channels, hidden, 1, 1)
        self.cv1 = Conv(original.cv1.conv.in_channels, hidden, 1, 1)
        self.cv2 = original.cv2
        self.m = original.m
        self.to(device=original.cv1.conv.weight.device, dtype=original.cv1.conv.weight.dtype)
        state = original.cv1.state_dict()
        for branch, section in [(self.cv0, slice(0, hidden)), (self.cv1, slice(hidden, None))]:
            own = branch.state_dict()
            for key in own:
                own[key] = state[key] if own[key].ndim == 0 else state[key][section].clone()
            branch.load_state_dict(own)
            branch.bn.eps = original.cv1.bn.eps
            branch.bn.momentum = original.cv1.bn.momentum
        for key in ("i", "f", "type", "np"):
            if hasattr(original, key):
                setattr(self, key, getattr(original, key))
        self.train(original.training)

    def forward(self, x):
        outputs = [self.cv0(x), self.cv1(x)]
        outputs.extend(block(outputs[-1]) for block in self.m)
        return self.cv2(torch.cat(outputs, 1))


def convert_c2f(model):
    count = 0
    for name, module in list(model.named_children()):
        if isinstance(module, C2f):
            setattr(model, name, C2fSplit(module))
            count += 1
        else:
            count += convert_c2f(module)
    return count


def convert_verified(model, size=320):
    model.eval().float()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(7321)
    image = torch.rand((1, 3, size, size), generator=generator, device=device)
    with torch.inference_mode():
        before = model(image)[0].detach().clone()
    count = convert_c2f(model)
    with torch.inference_mode():
        after = model(image)[0]
    error = float((before - after).abs().max())
    equivalent = bool(torch.allclose(before, after, rtol=1e-5, atol=2e-4))
    if not equivalent:
        raise RuntimeError(f"C2f conversion changed predictions: max error={error}")
    return {"converted_blocks": count, "maximum_absolute_output_error": error,
            "allclose": equivalent, "rtol": 1e-5, "atol": 2e-4}


def complexity(model, size):
    model.eval().float()
    sample = torch.zeros((1, 3, size, size), device=next(model.parameters()).device)
    macs, params = tp.utils.count_ops_and_params(model, sample)
    return {"parameters": int(params), "macs_per_image": float(macs),
            "counter": "torch_pruning.utils.count_ops_and_params; MAC-style estimate",
            "input": [1, 3, size, size]}


def prune_model(model, ratio, size=320):
    model.eval().float()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    before = complexity(model, size)
    sample = torch.zeros((1, 3, size, size), device=next(model.parameters()).device)
    ignored = [module for module in model.modules() if isinstance(module, Detect)]
    # Detect caches anchors created during inference-only equivalence checks.
    # Rebuild them in normal autograd mode before DepGraph traces dependencies.
    for head in ignored:
        head.shape = None
        head.anchors = head.anchors.clone()
        head.strides = head.strides.clone()
    with torch.enable_grad():
        pruner = tp.pruner.MagnitudePruner(
            model, sample, importance=tp.importance.GroupMagnitudeImportance(p=2),
            pruning_ratio=ratio, global_pruning=True, iterative_steps=1,
            ignored_layers=ignored, round_to=8,
        )
        pruner.step()
    after = complexity(model, size)
    with torch.inference_mode():
        output = model(sample)[0]
        if not torch.isfinite(output).all():
            raise RuntimeError("Pruned predictions contain non-finite values")
    if after["parameters"] >= before["parameters"]:
        raise RuntimeError("The pruning operation did not reduce parameter count")
    return {"requested_channel_ratio": ratio, "before": before, "after": after,
            "actual_parameter_reduction": 1 - after["parameters"] / before["parameters"],
            "actual_mac_reduction": 1 - after["macs_per_image"] / before["macs_per_image"],
            "protected_module": "Detect", "round_to": 8}
