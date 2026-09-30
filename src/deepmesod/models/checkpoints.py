"""Checkpoint loaders for the legacy-compatible vector models."""

import torch

from .vector import VectorPsiJModel, VectorWeakPriorDrift


def _state(payload):
    return payload.get("model_state_dict", payload)


def load_vector_drift(path, device="cpu", **config):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = _state(payload)
    if not config:
        config = {"patch_size": 13, "kernel_size": state["kernel.kernel.conv.weight"].shape[-1],
                  "mlp_hidden": state["mlp.0.weight"].shape[0], "mlp_layers": 3,
                  "n_components": state["mlp.6.weight"].shape[0]}
    model = VectorWeakPriorDrift(**config).to(device)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def load_vector_thermodynamics(path, device="cpu", jm_mode=None, **config):
    payload = torch.load(path, map_location=device, weights_only=False)
    state = _state(payload)
    kernel_size = state["kernel.kernel.conv.weight"].shape[-1]
    hidden = []
    index = 0
    while f"psi_mlp.mlp.{index}.weight" in state:
        hidden.append(state[f"psi_mlp.mlp.{index}.weight"].shape[0])
        index += 2
    model_config = {"patch_size": kernel_size, "kernel_size": kernel_size,
                    "mlp_hidden": tuple(hidden[:-1]),
                    "jm_mode": jm_mode or payload.get("jm_mode", "raw")}
    model_config.update(config)
    model = VectorPsiJModel(**model_config).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload
