"""Vector-spin drift and free-energy parameterizations for pipeline_r1."""

import torch
from torch import nn

from .kernels import MatrixKernel


def _mlp(input_size, hidden, output_size):
    layers = []
    previous = input_size
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


class VectorWeakPriorDrift(nn.Module):
    """Stage-1 ``WeakPriorParam`` with explicit beta and external-field inputs."""

    def __init__(self, patch_size=13, kernel_size=13, kernel_normalize_mode="sum",
                 mlp_hidden=64, mlp_layers=3, n_components=3):
        super().__init__()
        if kernel_size > patch_size:
            raise ValueError("kernel_size cannot exceed patch_size")
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.n_components = n_components
        self.kernel = MatrixKernel(
            kernel_size=kernel_size, normalize_mode=kernel_normalize_mode
        )
        self.mlp = _mlp(2 * n_components + 2,
                        (mlp_hidden,) * mlp_layers, n_components)

    def forward(self, patches, beta, external_field):
        if patches.ndim != 4 or patches.shape[1] != self.n_components:
            raise ValueError("patches must have shape (batch, components, height, width)")
        convolved = torch.cat(
            [self.kernel(patches[:, i:i + 1]) for i in range(self.n_components)], dim=1
        )
        center = self.patch_size // 2
        values = patches[:, :, center, center]
        interaction = convolved[:, :, convolved.shape[-2] // 2, convolved.shape[-1] // 2]
        return self.mlp(torch.cat((values, interaction, beta[:, None], external_field[:, None]), dim=1))


class VectorPsiMLP(nn.Module):
    def __init__(self, hidden=(64, 64, 64), n_components=3):
        super().__init__()
        self.mlp = _mlp(2 * n_components + 2, hidden, 1)

    def forward(self, state, interaction, beta, external_field):
        return self.mlp(
            torch.cat((state, interaction, beta[:, None], external_field[:, None]), dim=1)
        )


class VectorPsiJModel(nn.Module):
    """Stage-3 scalar density ``Psi(m, Jm, beta, h)`` model."""

    def __init__(self, patch_size=13, kernel_size=13, kernel_normalize_mode="sum",
                 mlp_hidden=(64, 64, 64), n_components=3, jm_mode="raw"):
        super().__init__()
        if jm_mode not in {"raw", "deviation"}:
            raise ValueError("jm_mode must be 'raw' or 'deviation'")
        if kernel_size > patch_size:
            raise ValueError("kernel_size cannot exceed patch_size")
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.n_components = n_components
        self.jm_mode = jm_mode
        self.kernel = MatrixKernel(kernel_size=kernel_size, normalize_mode=kernel_normalize_mode)
        self.psi_mlp = VectorPsiMLP(mlp_hidden, n_components)

    def interaction_field(self, field):
        deviation = torch.cat(
            [self.kernel(field[:, i:i + 1]) for i in range(self.n_components)], dim=1
        )
        return deviation if self.jm_mode == "deviation" else deviation + field

    def density(self, field, beta, external_field):
        interaction = self.interaction_field(field)
        sites = field.shape[0] * field.shape[2] * field.shape[3]
        state = field.permute(0, 2, 3, 1).reshape(sites, self.n_components)
        interaction = interaction.permute(0, 2, 3, 1).reshape(sites, self.n_components)
        beta = beta.reshape(-1, 1).expand(-1, field.shape[2] * field.shape[3]).reshape(-1)
        external_field = external_field.reshape(-1, 1).expand(
            -1, field.shape[2] * field.shape[3]
        ).reshape(-1)
        return self.psi_mlp(state, interaction, beta, external_field).reshape(
            field.shape[0], field.shape[2], field.shape[3]
        )

    def functional_force(self, field, beta, external_field, create_graph=True):
        state = field.detach().clone().requires_grad_(True)
        energy = self.density(state, beta, external_field).sum()
        return torch.autograd.grad(energy, state, create_graph=create_graph)[0]

    def forward(self, patches, beta, external_field):
        center = self.patch_size // 2
        state = patches[:, :, center, center]
        interaction_field = torch.cat(
            [self.kernel(patches[:, i:i + 1]) for i in range(self.n_components)], dim=1
        )
        interaction = interaction_field[:, :, interaction_field.shape[-2] // 2,
                                        interaction_field.shape[-1] // 2]
        if self.jm_mode == "raw":
            interaction = interaction + state
        return self.psi_mlp(state, interaction, beta, external_field)


# Names retained as lightweight migration aliases for the original scripts.
WeakPriorParam = VectorWeakPriorDrift
PsiJModel = VectorPsiJModel
