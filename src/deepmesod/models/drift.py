from itertools import pairwise

import torch
from torch import nn


def activation(name):
    return {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
    }[name]


class MLP(nn.Module):
    def __init__(self, input_size, hidden, output_size, activation_name):
        super().__init__()
        layer = activation(activation_name)
        sizes = [input_size, *hidden, output_size]
        modules = []
        for left, right in pairwise(sizes[:-1]):
            modules.extend((nn.Linear(left, right), layer()))
        modules.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*modules)

    def forward(self, values):
        return self.net(values)


class InteractionKernel(nn.Module):
    constraints = frozenset(
        {
            "none",
            "sum_one",
            "centrosymmetric",
            "centrosymmetric_sum_one",
        }
    )

    def __init__(self, size, *, constraint="none", initialization="uniform"):
        super().__init__()
        if constraint not in self.constraints:
            raise ValueError(f"unknown interaction constraint '{constraint}'")
        self.size = size
        self.constraint = constraint
        self.raw = nn.Parameter(torch.empty(1, 1, size, size))
        if initialization == "uniform":
            nn.init.constant_(self.raw, 1 / (size * size))
        elif initialization == "normal":
            nn.init.normal_(self.raw, std=0.01)
        else:
            raise ValueError(f"unknown initialization '{initialization}'")

    @property
    def weight(self):
        value = self.raw
        if self.constraint in {"centrosymmetric", "centrosymmetric_sum_one"}:
            value = 0.5 * (value + torch.flip(value, dims=(-2, -1)))
        if self.constraint in {"sum_one", "centrosymmetric_sum_one"}:
            value = (
                value - value.mean(dim=(-2, -1), keepdim=True) + 1 / (self.size * self.size)
            )
        return value


class UnitSumKernel(InteractionKernel):
    def __init__(self, size, *, centrosymmetric=False, initialization="uniform"):
        constraint = "centrosymmetric_sum_one" if centrosymmetric else "sum_one"
        super().__init__(size, constraint=constraint, initialization=initialization)


class WeakScalarDrift(nn.Module):
    def __init__(self, patch_size=15, hidden=(128, 128, 128), activation_name="softplus"):
        super().__init__()
        self.patch_size = patch_size
        self.context = UnitSumKernel(patch_size, initialization="uniform")
        self.mlp = MLP(2, hidden, 1, activation_name)

    def forward(self, patches):
        center = patches[:, :, self.patch_size // 2, self.patch_size // 2]
        context = (patches * self.context.weight).sum(dim=(-2, -1))
        return self.mlp(torch.cat((center, context), dim=1)).squeeze(1)


class PatchCNNDrift(nn.Module):
    def __init__(self, channels=(32, 32), activation_name="tanh"):
        super().__init__()
        layer = activation(activation_name)
        sizes = [1, *channels, 1]
        modules = []
        for left, right in pairwise(sizes[:-1]):
            modules.extend((nn.Conv2d(left, right, 3, padding=1), layer()))
        modules.append(nn.Conv2d(sizes[-2], sizes[-1], 3, padding=1))
        self.net = nn.Sequential(*modules)

    def forward(self, patches):
        result = self.net(patches)
        center = result.shape[-1] // 2
        return result[:, 0, center, center]
