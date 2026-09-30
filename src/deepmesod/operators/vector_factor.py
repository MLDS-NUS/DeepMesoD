"""Finite-range factorized mobility for vector fields.

The implementation mirrors the full ``pipeline_r1`` factor: a 3x3 stencil of
dense 3x3 blocks, circular convolutions, spin-flip-even parameterization, and
an exact local Ito divergence.  It deliberately lives beside the scalar
operator so both model classes expose the same ``B(B* f)`` contract.
"""

import torch
import torch.nn.functional as F
from torch import nn


def square_offsets(radius):
    return [(dy, dx) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)]


class VectorFiniteRangeFactor(nn.Module):
    def __init__(self, patch_size=15, factor_radius=1, likelihood_radius=1,
                 n_components=3, hidden_channels=32, n_layers=3, activation="tanh",
                 structure="full", min_scale=1e-5, spin_flip_even=True,
                 symmetry="none", axial=True, projector_epsilon=0.05):
        super().__init__()
        if patch_size % 2 != 1 or patch_size < 4 * factor_radius + 1:
            raise ValueError("patch_size must be odd and at least 4*factor_radius+1")
        if structure not in {"uniaxial", "local_uniaxial", "full"}:
            raise ValueError("unknown vector factor structure")
        if structure != "full" and n_components != 3:
            raise ValueError("uniaxial vector factors require three components")
        if symmetry not in {"none", "c4", "d4"}:
            raise ValueError("symmetry must be none, c4, or d4")
        if symmetry != "none":
            raise ValueError("the migrated pipeline factor only supports symmetry='none'")
        if n_layers < 2:
            raise ValueError("n_layers must be at least two")
        self.patch_size = patch_size
        self.factor_radius = factor_radius
        self.likelihood_radius = likelihood_radius
        self.n_components = n_components
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.activation_name = activation
        self.structure = structure
        self.min_scale = min_scale
        self.spin_flip_even = spin_flip_even
        self.symmetry = symmetry
        self.d4_equivariant = symmetry != "none"
        self.axial = axial
        self.projector_epsilon = projector_epsilon
        self.factor_offsets = square_offsets(factor_radius)
        self.center_factor_index = self.factor_offsets.index((0, 0))
        self.n_factors = len(self.factor_offsets)
        self.per_offset = 2 if structure != "full" else n_components ** 2
        self.n_outputs = self.n_factors * self.per_offset
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh,
               "softplus": nn.Softplus}[activation]
        layers = []
        channels = n_components
        for layer in range(n_layers):
            output = hidden_channels if layer < n_layers - 1 else self.n_outputs
            layers.append(nn.Conv2d(channels, output, 3, padding=1,
                                    padding_mode="circular"))
            if layer < n_layers - 1:
                layers.append(act())
            channels = output
        self.net = nn.Sequential(*layers)
        self.force_radius = 2 * factor_radius
        self.force_offsets = square_offsets(self.force_radius)
        self.center_force_index = self.force_offsets.index((0, 0))
        self._register_geometry()

    def _register_geometry(self):
        rows = square_offsets(self.likelihood_radius)
        columns = square_offsets(self.likelihood_radius + self.factor_radius)
        lookup = {offset: i for i, offset in enumerate(columns)}
        indices = [[lookup[(ry + fy, rx + fx)] for fy, fx in self.factor_offsets]
                   for ry, rx in rows]
        self.register_buffer("local_column_index", torch.tensor(indices), persistent=False)
        self.n_likelihood_rows = len(rows)
        self.n_likelihood_columns = len(columns)

    def configuration(self):
        return {"patch_size": self.patch_size, "factor_radius": self.factor_radius,
                "likelihood_radius": self.likelihood_radius,
                "n_components": self.n_components, "hidden_channels": self.hidden_channels,
                "n_layers": self.n_layers, "activation": self.activation_name,
                "structure": self.structure, "min_scale": self.min_scale,
                "spin_flip_even": self.spin_flip_even, "symmetry": self.symmetry,
                "axial": self.axial, "projector_epsilon": self.projector_epsilon}

    def _raw_components(self, patches):
        centre = self.patch_size // 2
        raw = self.net(patches)[:, :, centre, centre]
        raw = raw.reshape(-1, self.n_factors, self.per_offset)
        index = self.center_factor_index
        if self.structure == "full":
            diagonal = [i * self.n_components + i for i in range(self.n_components)]
            centre_values = raw[:, index].clone()
            centre_values[:, diagonal] = F.softplus(centre_values[:, diagonal]) + self.min_scale
            raw = torch.cat((raw[:, :index], centre_values[:, None], raw[:, index + 1:]), 1)
        else:
            mask = raw.new_zeros(self.n_factors, 1)
            mask[index] = 1
            raw = mask * (F.softplus(raw) + self.min_scale) + (1 - mask) * raw
        return raw

    def components(self, patches):
        raw = self._raw_components(torch.cat((patches, -patches), 0)) if self.spin_flip_even else self._raw_components(patches)
        if self.spin_flip_even:
            raw = raw.reshape(2, patches.shape[0], self.n_factors, self.per_offset).mean(0)
        return raw

    def _to_blocks(self, components, patches=None):
        C = self.n_components
        if self.structure == "local_uniaxial":
            if patches is None:
                raise ValueError("local_uniaxial requires patches")
            centre = self.patch_size // 2
            vector = patches[:, :, centre, centre]
            denominator = vector.square().sum(1, keepdim=True) + self.projector_epsilon ** 2
            projector = vector[:, :, None] * vector[:, None, :] / denominator[:, :, None]
            identity = torch.eye(C, device=patches.device, dtype=patches.dtype)
            return (components[:, :, 0, None, None] * (identity - projector)[:, None]
                    + components[:, :, 1, None, None] * projector[:, None])
        blocks = components.new_zeros(components.shape[0], self.n_factors, C, C)
        if self.structure == "full":
            return components.reshape(components.shape[0], self.n_factors, C, C)
        blocks[:, :, 0, 0] = components[:, :, 0]
        blocks[:, :, 1, 1] = components[:, :, 0]
        blocks[:, :, 2, 2] = components[:, :, 1]
        return blocks

    def forward(self, patches):
        return self._to_blocks(self.components(patches), patches)

    def _field_patches(self, field):
        pad = self.patch_size // 2
        padded = F.pad(field, (pad,) * 4, mode="circular")
        patches = F.unfold(padded, self.patch_size).transpose(1, 2)
        return patches.reshape(-1, self.n_components, self.patch_size, self.patch_size)

    def blocks_on_patch_grid(self, patches):
        shape = patches.shape
        blocks = self.forward(patches.reshape(-1, *shape[2:]))
        return blocks.reshape(shape[0], shape[1], self.n_factors,
                             self.n_components, self.n_components)

    def blocks_on_field(self, field):
        if field.ndim != 4 or field.shape[1] != self.n_components:
            raise ValueError("field must have shape (batch, components, height, width)")
        batch, _, height, width = field.shape
        if self.symmetry != "none" or self.n_layers > self.patch_size // 2:
            blocks = self.forward(self._field_patches(field))
            return blocks.reshape(batch, height, width, self.n_factors,
                                  self.n_components, self.n_components).permute(0, 3, 4, 5, 1, 2)
        stacked = torch.cat((field, -field), 0) if self.spin_flip_even else field
        raw = self.net(stacked).reshape(-1, self.n_factors, self.per_offset, height, width)
        index = self.center_factor_index
        centre = raw[:, index].clone()
        if self.structure == "full":
            diagonal = [i * self.n_components + i for i in range(self.n_components)]
            centre[:, diagonal] = F.softplus(centre[:, diagonal]) + self.min_scale
        else:
            centre = F.softplus(centre) + self.min_scale
        raw = torch.cat((raw[:, :index], centre[:, None], raw[:, index + 1:]), 1)
        if self.spin_flip_even:
            raw = raw.reshape(2, batch, self.n_factors, self.per_offset, height, width).mean(0)
        return self._to_blocks_field(raw, field)

    def _to_blocks_field(self, components, field):
        C = self.n_components
        if self.structure == "full":
            return components.reshape(components.shape[0], self.n_factors, C, C,
                                     field.shape[-2], field.shape[-1])
        vector = field
        denominator = vector.square().sum(1, keepdim=True) + self.projector_epsilon ** 2
        projector = vector[:, :, None] * vector[:, None, :] / denominator[:, :, None]
        identity = torch.eye(C, device=field.device, dtype=field.dtype)
        transverse = identity[None, None, :, :, None, None] - projector[:, None]
        alpha = components[:, :, 0][:, :, None, None]
        gamma = components[:, :, 1][:, :, None, None]
        return alpha * transverse + gamma * projector[:, None]

    def _check(self, blocks, field):
        expected = (field.shape[0], self.n_factors, self.n_components,
                    self.n_components, *field.shape[-2:])
        if tuple(blocks.shape) != expected:
            raise ValueError(f"blocks must have shape {expected}, got {tuple(blocks.shape)}")

    def apply_factor(self, blocks, noise):
        self._check(blocks, noise)
        result = torch.zeros_like(noise)
        for i, (dy, dx) in enumerate(self.factor_offsets):
            shifted = torch.roll(noise, (-dy, -dx), (-2, -1))
            result += torch.einsum("nacyx,ncyx->nayx", blocks[:, i], shifted)
        return result

    def apply_adjoint(self, blocks, force):
        self._check(blocks, force)
        result = torch.zeros_like(force)
        for i, (dy, dx) in enumerate(self.factor_offsets):
            local = torch.einsum("nacyx,nayx->ncyx", blocks[:, i], force)
            result += torch.roll(local, (dy, dx), (-2, -1))
        return result

    def apply_mobility(self, blocks, force):
        return self.apply_factor(blocks, self.apply_adjoint(blocks, force))

    def local_factor_matrix(self, blocks):
        if tuple(blocks.shape[1:]) != (self.n_likelihood_rows, self.n_factors,
                                        self.n_components, self.n_components):
            raise ValueError("block grid does not match likelihood geometry")
        batch, rows = blocks.shape[:2]
        matrix = blocks.new_zeros(batch, rows, self.n_components,
                                  self.n_likelihood_columns, self.n_components)
        for factor_index in range(self.n_factors):
            columns = self.local_column_index[:, factor_index]
            for row in range(rows):
                matrix[:, row, :, columns[row], :] += blocks[:, row, factor_index]
        return matrix.reshape(batch, rows * self.n_components,
                              self.n_likelihood_columns * self.n_components)

    def local_covariance(self, blocks):
        matrix = self.local_factor_matrix(blocks)
        return matrix @ matrix.transpose(-1, -2)

    def _jacobian(self, patches, chunk_size):
        C = self.n_components
        entries = ([(r, c) for r in range(C) for c in range(C)]
                   if self.structure == "full" else [(i, i) for i in range(C)])
        outputs = []
        radius = self.patch_size // 2
        for start in range(0, patches.shape[0], chunk_size):
            with torch.enable_grad():
                state = patches[start:start + chunk_size].detach().requires_grad_(True)
                blocks = self.forward(state)
                result = blocks.new_zeros(blocks.shape[0], self.n_factors, C, C,
                                          len(self.force_offsets), C)
                total = self.n_factors * len(entries)
                completed = 0
                for factor_index in range(self.n_factors):
                    for row, col in entries:
                        completed += 1
                        gradient = torch.autograd.grad(
                            blocks[:, factor_index, row, col].sum(), state,
                            retain_graph=completed < total, create_graph=False
                        )[0]
                        result[:, factor_index, row, col] = torch.stack(
                            [gradient[:, :, radius + dy, radius + dx] for dy, dx in self.force_offsets], 1
                        )
            outputs.append(result)
        return torch.cat(outputs)

    def divergence_on_field_exact(self, field, chunk_size=512, method="patch", **_):
        if field.ndim != 4 or field.shape[1] != self.n_components:
            raise ValueError("field must have shape (batch, components, height, width)")
        if method not in {"patch", "auto", "fast"}:
            raise ValueError("method must be patch, auto, or fast")
        batch, _, height, width = field.shape
        patches = self._field_patches(field)
        blocks = self.forward(patches).reshape(
            batch, height, width, self.n_factors, self.n_components, self.n_components
        ).permute(0, 3, 4, 5, 1, 2)
        jacobian = self._jacobian(patches, chunk_size).reshape(
            batch, height, width, self.n_factors, self.n_components, self.n_components,
            len(self.force_offsets), self.n_components
        ).permute(0, 3, 4, 5, 6, 7, 1, 2)
        divergence = field.new_zeros(batch, self.n_components, height, width)
        lookup = {offset: i for i, offset in enumerate(self.factor_offsets)}
        for d_index, (dy, dx) in enumerate(self.force_offsets):
            for first, (fy, fx) in enumerate(self.factor_offsets):
                second = lookup.get((fy - dy, fx - dx))
                if second is None:
                    continue
                pulled = torch.roll(blocks[:, second], (-dy, -dx), (-2, -1))
                first_derivative = jacobian[:, first, :, :, d_index]
                divergence += torch.einsum("naceyx,necyx->nayx", first_derivative, pulled)
                centre = jacobian[:, second, :, :, self.center_force_index]
                traced = torch.diagonal(centre, dim1=1, dim2=3).sum(-1)
                traced = torch.roll(traced, (-dy, -dx), (-2, -1))
                divergence += torch.einsum("nacyx,ncyx->nayx", blocks[:, first], traced)
        return divergence
