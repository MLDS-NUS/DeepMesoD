import inspect

import torch

from .factor import FiniteRangeFactor


def load_factor(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    allowed = inspect.signature(FiniteRangeFactor).parameters
    config = {
        key: value for key, value in payload["model_config"].items() if key in allowed
    }
    factor = FiniteRangeFactor(**config).to(device)
    factor.load_state_dict(payload["model_state_dict"])
    factor.eval()
    for parameter in factor.parameters():
        parameter.requires_grad_(False)
    return factor, payload
