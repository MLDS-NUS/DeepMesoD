import torch

from deepmesod.classes import register_loss


@register_loss("forward_rate_mse")
def forward_rate_mse(prediction, target, weight=None):
    error = prediction - target
    if weight is None:
        return error.square().mean()
    return (weight * error.square()).mean()


@register_loss("cg_force_target")
def cg_force_target(prediction, target, weight=None):
    return forward_rate_mse(prediction, target, weight)


@register_loss("direct_forward_rate")
def direct_forward_rate(prediction, target, weight=None):
    return forward_rate_mse(prediction, target, weight)


def state_weights(field, bins, maximum):
    index = ((field + 1) * 0.5 * bins).floor().clamp(0, bins - 1).long()
    counts = torch.bincount(index.reshape(-1), minlength=bins).double()
    weights = counts.sum() / (bins * counts.clamp_min(1))
    weights = weights.clamp_max(maximum).float()
    normalizer = (weights.double() * counts).sum() / counts.sum()
    return weights / normalizer.float(), index
