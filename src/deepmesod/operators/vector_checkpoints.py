"""Loaders for the vector factor checkpoints emitted by pipeline_r1."""

import inspect

import torch

from .vector_factor import VectorFiniteRangeFactor


def load_vector_factor(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = {k: v for k, v in payload["model_config"].items()
              if k in inspect.signature(VectorFiniteRangeFactor).parameters}
    factor = VectorFiniteRangeFactor(**config).to(device)
    factor.load_state_dict(payload["model_state_dict"])
    factor.eval()
    for parameter in factor.parameters():
        parameter.requires_grad_(False)
    return factor, payload
