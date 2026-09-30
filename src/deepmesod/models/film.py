"""CrBr3 film models: factorized 3D interaction plus shared local MLPs."""

import torch
import torch.nn.functional as F
from torch import nn


class FactorizedFilmKernel(nn.Module):
    """J(z_out,z_in,dy,dx) = J_z(z_out,z_in) w_xy(dy,dx).

    Parameter names intentionally match the archived MesoscopicModel
    ``FactorizedKernel`` checkpoints.
    """

    def __init__(self, num_layers, kernel_size, normalize_mode="sum", jz_normalize_mode="none"):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.normalize_mode = normalize_mode
        self.jz_normalize_mode = jz_normalize_mode
        self.w_xy = nn.Parameter(torch.empty(kernel_size, kernel_size))
        nn.init.uniform_(self.w_xy, -0.01, 0.01)
        n = num_layers * (num_layers + 1) // 2
        initial = torch.zeros(n)
        diagonal = torch.arange(num_layers)
        flat_diagonal = diagonal * num_layers - diagonal * (diagonal - 1) // 2
        initial[flat_diagonal] = 0.01
        self.J_z_flat = nn.Parameter(initial)
        indices = torch.triu_indices(num_layers, num_layers)
        self.register_buffer("triu_row", indices[0])
        self.register_buffer("triu_col", indices[1])

    def _jz(self):
        result = self.J_z_flat.new_zeros(self.num_layers, self.num_layers)
        result[self.triu_row, self.triu_col] = self.J_z_flat
        return result + result.triu(1).transpose(0, 1)

    def _w(self):
        if self.normalize_mode == "sum":
            return self.w_xy / (self.w_xy.sum() + 1e-8)
        if self.normalize_mode == "softmax":
            return F.softmax(self.w_xy.reshape(-1), dim=0).reshape_as(self.w_xy)
        if self.normalize_mode == "none":
            return self.w_xy
        raise ValueError(f"unknown spatial normalization: {self.normalize_mode}")

    def _normalized_jz(self):
        jz = self._jz()
        if self.jz_normalize_mode == "sum":
            return jz / (jz.sum() + 1e-8)
        if self.jz_normalize_mode == "softmax":
            return F.softmax(jz.reshape(-1), dim=0).reshape_as(jz)
        if self.jz_normalize_mode == "none":
            return jz
        raise ValueError(f"unknown z normalization: {self.jz_normalize_mode}")

    def get_normalized_weight(self):
        return torch.einsum("oi,yx->oiyx", self._normalized_jz(), self._w())

    def forward(self, field):
        if field.ndim != 4 or field.shape[1] != self.num_layers:
            raise ValueError("film kernel expects (batch, Lz, height, width)")
        pad = self.kernel_size // 2
        b, lz, h, w = field.shape
        spatial = F.conv2d(
            F.pad(field.reshape(b * lz, 1, h, w), (pad,) * 4, mode="circular"),
            self._w().reshape(1, 1, self.kernel_size, self.kernel_size),
        ).reshape(b, lz, h, w)
        return torch.einsum("oi,biyx->boyx", self._normalized_jz(), spatial)


def _mlp(input_size, hidden, output_size):
    layers = []
    previous = input_size
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, output_size))
    return nn.Sequential(*layers)


class FilmWeakPriorDrift(nn.Module):
    """Archived CrBr3 Stage-1 weak3d drift model."""

    def __init__(self, num_layers=28, num_components=3, patch_size=7,
                 kernel_size=7, kernel_normalize_mode="sum", jz_normalize_mode="none",
                 mlp_hidden=64, mlp_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.num_components = num_components
        self.patch_size = patch_size
        self.kernel = FactorizedFilmKernel(num_layers, kernel_size,
                                            kernel_normalize_mode, jz_normalize_mode)
        self.mlp = _mlp(2 * num_components, (mlp_hidden,) * mlp_layers, num_components)

    def forward(self, m_patch):
        if m_patch.ndim != 5 or m_patch.shape[1:3] != (self.num_components, self.num_layers):
            raise ValueError("film drift expects (batch, components, Lz, patch_y, patch_x)")
        interactions = torch.stack([self.kernel(m_patch[:, c]) for c in range(self.num_components)], dim=1)
        centre = self.patch_size // 2
        state = m_patch[:, :, :, centre, centre]
        jm = interactions[:, :, :, interactions.shape[-2] // 2, interactions.shape[-1] // 2]
        inputs = torch.cat((state, jm), dim=1).permute(0, 2, 1).reshape(-1, 2 * self.num_components)
        return self.mlp(inputs).reshape(m_patch.shape[0], self.num_layers, self.num_components).permute(0, 2, 1)


class FilmPsiMLP(nn.Module):
    def __init__(self, hidden=(64, 64, 64), n_components=3):
        super().__init__()
        self.mlp = _mlp(2 * n_components, tuple(hidden), 1)

    def forward(self, state, interaction):
        return self.mlp(torch.cat((state, interaction), dim=-1))


class FilmPsiJModel(nn.Module):
    """Archived CrBr3 Stage-3 ``Psi/J`` model with a functional force."""

    def __init__(self, num_layers=28, num_components=3, patch_size=7,
                 kernel_normalize_mode="sum", jz_normalize_mode="none",
                 mlp_hidden=(64, 64, 64), spin_flip_even=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_components = num_components
        self.patch_size = patch_size
        self.spin_flip_even = bool(spin_flip_even)
        self.kernel = FactorizedFilmKernel(num_layers, patch_size,
                                            kernel_normalize_mode, jz_normalize_mode)
        self.psi_mlp = FilmPsiMLP(mlp_hidden, num_components)

    def local_energy_from_input(self, inputs):
        value = self.psi_mlp.mlp(inputs)
        if not self.spin_flip_even:
            return value
        return 0.5 * (value + self.psi_mlp.mlp(-inputs))

    def interaction_field(self, field):
        return torch.stack([self.kernel(field[:, c]) for c in range(self.num_components)], dim=1)

    def density(self, field):
        interaction = self.interaction_field(field)
        b, c, lz, h, w = field.shape
        state = field.permute(0, 2, 3, 4, 1).reshape(-1, c)
        jm = interaction.permute(0, 2, 3, 4, 1).reshape(-1, c)
        return self.local_energy_from_input(torch.cat((state, jm), dim=-1)).reshape(b, lz, h, w)

    def functional_force(self, field, h=0.0, create_graph=True):
        variable = field.detach().clone().requires_grad_(True)
        energy = self.density(variable).sum()
        if abs(float(h)) > 0:
            energy = energy - float(h) * variable[:, -1].sum()
        return torch.autograd.grad(energy, variable, create_graph=create_graph)[0]

