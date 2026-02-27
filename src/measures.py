from torch import nn
import torch

from utils_quantization import FakeQuantParametrization


@torch.no_grad()
def sparsity(model: nn.Module, zero_eps: float = 0.0) -> float:
    vals = []
    for mod in model.modules():
        if parametrize.is_parametrized(mod, "weight"):
            plist = mod.parametrizations["weight"]
            if any(isinstance(p, FakeQuantParametrization) for p in plist):
                Wq = mod.weight  # goes through the FakeQuantParametrization
                if zero_eps > 0.0:
                    nz = (Wq.abs() > zero_eps).sum().item()
                else:
                    nz = (Wq != 0).sum().item()
                vals.append(1 - nz / float(Wq.numel()))
    return vals

@torch.no_grad()
def deadzones_logits(model) -> float:
    ds = []
    for m in model.modules():
        if isinstance(m, FakeQuantParametrization):
            d = m.quantizer.get_deadzone_logit()
            ds.append(d.item() if torch.is_tensor(d) else float(d))
    return ds

@torch.no_grad()
def bitwidths_logits(model) -> float:
    ds = []
    for m in model.modules():
        if isinstance(m, FakeQuantParametrization):
            d = m.quantizer.get_bitwidth_logit()
            ds.append(d.item() if torch.is_tensor(d) else float(d))
    return ds

@torch.no_grad()
def bitwidths(model) -> float:
    ds = []
    for m in model.modules():
        if isinstance(m, FakeQuantParametrization):
            d = m.quantizer.get_bitwidth()
            ds.append(d.item() if torch.is_tensor(d) else float(d))
    return ds
