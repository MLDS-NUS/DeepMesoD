"""Legacy-compatible learnable circular kernels."""

import torch
import torch.nn.functional as F
from torch import nn


class KernelCNN(nn.Module):
    """A shared spatial kernel returning ``Jm - m`` for unit-sum kernels."""

    def __init__(self, in_ch=1, out_ch=1, kernel_size=21, normalize_mode="none"):
        super().__init__()
        self.normalize_mode = normalize_mode
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, padding=kernel_size // 2,
            padding_mode="circular", bias=False
        )

    @property
    def weight(self):
        return self.conv.weight

    def get_normalized_weight(self):
        if self.normalize_mode == "sum":
            return self.weight / (self.weight.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)
        if self.normalize_mode == "softmax":
            return F.softmax(self.weight.flatten(1), dim=1).reshape_as(self.weight)
        if self.normalize_mode == "none":
            return self.weight
        raise ValueError(f"unknown kernel normalization '{self.normalize_mode}'")

    def set_weight(self, weight):
        with torch.no_grad():
            self.weight.copy_(weight)

    def forward(self, field):
        padding = self.kernel_size // 2
        padded = F.pad(field, (padding,) * 4, mode="circular")
        result = F.conv2d(padded, self.get_normalized_weight())
        return result - field if self.in_channels == self.out_channels else result

    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels


class MatrixKernel(nn.Module):
    """Wrapper retaining the old ``kernel.kernel.conv.weight`` state keys."""

    def __init__(self, in_ch=1, out_ch=1, kernel_size=21, normalize_mode="none"):
        super().__init__()
        self.kernel = KernelCNN(in_ch, out_ch, kernel_size, normalize_mode)

    @property
    def normalize_mode(self):
        return self.kernel.normalize_mode

    @property
    def raw_weight(self):
        return self.kernel.weight

    @property
    def final_weight(self):
        return self.kernel.get_normalized_weight()

    def set_weight(self, weight):
        self.kernel.set_weight(weight)

    def forward(self, field):
        return self.kernel(field)


def create_kernel(kernel_type="matrix", kernel_size=21, normalize_mode="none"):
    if kernel_type != "matrix":
        raise ValueError("the migrated ferromagnet pipeline currently supports matrix kernels")
    return MatrixKernel(kernel_size=kernel_size, normalize_mode=normalize_mode)
