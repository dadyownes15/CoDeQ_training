import torch
import torch.nn as nn
import math


def group_lasso(model, lambda_linear=1e-4, lambda_conv=1e-4):
    """Group L2/L1 structured sparsity regularization.

    Computes L2 norm per column (input feature dimension), then sums (L1).
    Pushes entire input features/channels to zero for structured pruning.

    Works on both nn.Linear and nn.Conv2d layers.
    """
    reg = torch.tensor(0.0, device=next(model.parameters()).device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            W = m.weight  # [out_features, in_features]
            out_features = m.weight.shape[0]
            reg = reg + lambda_linear * (torch.norm(W, dim=0).sum())*math.sqrt(out_features)
        elif isinstance(m, nn.Conv2d):
            W = m.weight  # [out_c, in_c, kH, kW]
            W_flat = W.view(W.size(0), -1)  # [out_c, in_c*kH*kW]
            reg = reg + lambda_conv * torch.norm(W_flat, dim=0).sum()
    return reg


LOSS_REGISTRY = {
    "group_lasso": group_lasso,
}
