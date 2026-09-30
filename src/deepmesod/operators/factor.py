import torch
import torch.nn.functional as F
from torch import nn


def square_offsets(radius):
    return [
        (dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
    ]


def transform_offset(offset, rotations, reflected):
    dy, dx = offset
    if reflected:
        dx = -dx
    for _ in range(rotations):
        dy, dx = -dx, dy
    return dy, dx


class FiniteRangeFactor(nn.Module):
    """Coordinate finite-range factor B with K = B B^T.

    For scalar fields on a periodic grid,

        [B(phi) xi]_i = sum_s b_i(s) xi_{i+s}.

    The network returns the discrete coefficients b_i(s). For the continuum
    dictionary used in the paper, an off-center regular coefficient equals
    w b_r(phi; x_i, x_{i+s}), while a delta coefficient carries no factor w.
    """

    def __init__(
        self,
        patch_size=15,
        factor_radius=1,
        likelihood_radius=1,
        hidden_channels=32,
        n_layers=3,
        activation="tanh",
        dilations=None,
        min_center=1e-5,
        spin_flip_even=True,
        parameterization="direct",
        d4_equivariant=False,
    ):
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        if factor_radius < 0 or likelihood_radius < 0:
            raise ValueError("radii must be non-negative")
        if n_layers < 2:
            raise ValueError("n_layers must be at least two")
        if parameterization not in {"direct", "amplitude_shape"}:
            raise ValueError("parameterization must be 'direct' or 'amplitude_shape'")

        self.patch_size = patch_size
        self.factor_radius = factor_radius
        self.likelihood_radius = likelihood_radius
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.activation_name = activation
        self.dilations = tuple([1] * n_layers if dilations is None else dilations)
        if len(self.dilations) != n_layers:
            raise ValueError("dilations must contain one value per layer")
        if any(value < 1 for value in self.dilations):
            raise ValueError("dilations must be positive")
        self.receptive_radius = sum(self.dilations)
        self.min_center = min_center
        self.spin_flip_even = spin_flip_even
        self.parameterization = parameterization
        self.d4_equivariant = d4_equivariant
        self.factor_offsets = square_offsets(factor_radius)
        self.center_factor_index = self.factor_offsets.index((0, 0))
        self.n_factors = len(self.factor_offsets)

        activation_class = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "softplus": nn.Softplus,
        }[activation]
        layers = []
        channels = 1
        for layer_index, dilation in enumerate(self.dilations):
            output_channels = (
                hidden_channels
                if layer_index < n_layers - 1
                else self.n_factors + int(self.parameterization == "amplitude_shape")
            )
            layers.append(
                nn.Conv2d(
                    channels,
                    output_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    padding_mode="circular",
                )
            )
            if layer_index < n_layers - 1:
                layers.append(activation_class())
            channels = output_channels
        self.net = nn.Sequential(*layers)
        self._register_symmetry()
        self._register_geometry()

    def configuration(self):
        return {
            "patch_size": self.patch_size,
            "factor_radius": self.factor_radius,
            "likelihood_radius": self.likelihood_radius,
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "activation": self.activation_name,
            "dilations": list(self.dilations),
            "min_center": self.min_center,
            "spin_flip_even": self.spin_flip_even,
            "parameterization": self.parameterization,
            "d4_equivariant": self.d4_equivariant,
        }

    def _register_symmetry(self):
        lookup = {offset: index for index, offset in enumerate(self.factor_offsets)}
        permutations = []
        transformations = []
        for reflected in (False, True):
            for rotations in range(4):
                permutations.append(
                    [
                        lookup[transform_offset(offset, rotations, reflected)]
                        for offset in self.factor_offsets
                    ]
                )
                transformations.append((rotations, reflected))
        self.register_buffer(
            "d4_permutations",
            torch.tensor(permutations, dtype=torch.long),
            persistent=False,
        )
        self.d4_transformations = transformations

    def _register_geometry(self):
        row_offsets = square_offsets(self.likelihood_radius)
        column_offsets = square_offsets(self.likelihood_radius + self.factor_radius)
        column_lookup = {offset: index for index, offset in enumerate(column_offsets)}
        local_column_index = [
            [
                column_lookup[(row_y + factor_y, row_x + factor_x)]
                for factor_y, factor_x in self.factor_offsets
            ]
            for row_y, row_x in row_offsets
        ]
        self.register_buffer(
            "local_column_index",
            torch.tensor(local_column_index, dtype=torch.long),
            persistent=False,
        )
        self.n_likelihood_rows = len(row_offsets)
        self.n_likelihood_columns = len(column_offsets)

        self.force_radius = 2 * self.factor_radius
        self.force_offsets = square_offsets(self.force_radius)
        self.n_force_rows = len(self.force_offsets)
        self.center_force_index = self.force_offsets.index((0, 0))
        factor_lookup = {offset: index for index, offset in enumerate(self.factor_offsets)}
        force_factor_index = []
        force_factor_mask = []
        for row_y, row_x in self.force_offsets:
            indices = []
            mask = []
            for latent_y, latent_x in self.factor_offsets:
                relative = (latent_y - row_y, latent_x - row_x)
                indices.append(factor_lookup.get(relative, 0))
                mask.append(relative in factor_lookup)
            force_factor_index.append(indices)
            force_factor_mask.append(mask)
        self.register_buffer(
            "force_factor_index",
            torch.tensor(force_factor_index, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "force_factor_mask",
            torch.tensor(force_factor_mask, dtype=torch.bool),
            persistent=False,
        )

    def _raw_components(self, patches):
        center = self.patch_size // 2
        raw = self.net(patches)[:, :, center, center]
        raw_direction = raw[:, : self.n_factors]
        positive_center = (
            F.softplus(raw_direction[:, self.center_factor_index]) + self.min_center
        )
        selector = torch.zeros_like(raw_direction)
        selector[:, self.center_factor_index] = 1.0
        direction = raw_direction * (1.0 - selector) + positive_center[:, None] * selector
        if self.parameterization == "direct":
            return direction, None
        amplitude = F.softplus(raw[:, self.n_factors]) + self.min_center
        return direction, amplitude

    @staticmethod
    def _transform_patches(patches, rotations, reflected):
        if reflected:
            patches = torch.flip(patches, dims=(-1,))
        if rotations:
            patches = torch.rot90(
                patches,
                k=rotations,
                dims=(-2, -1),
            )
        return patches

    def _even_components(self, patches):
        if not self.spin_flip_even:
            return self._raw_components(patches)
        batch = patches.shape[0]
        direction, amplitude = self._raw_components(torch.cat([patches, -patches], dim=0))
        direction = direction.reshape(2, batch, self.n_factors).mean(dim=0)
        if amplitude is not None:
            amplitude = amplitude.reshape(2, batch).mean(dim=0)
        return direction, amplitude

    def _raw_components_on_field(self, field):
        raw = self.net(field)
        raw_direction = raw[:, : self.n_factors]
        positive_center = (
            F.softplus(raw_direction[:, self.center_factor_index]) + self.min_center
        )
        selector = torch.zeros_like(raw_direction)
        selector[:, self.center_factor_index] = 1.0
        direction = raw_direction * (1.0 - selector) + positive_center[:, None] * selector
        if self.parameterization == "direct":
            return direction, None
        amplitude = F.softplus(raw[:, self.n_factors]) + self.min_center
        return direction, amplitude

    def _even_components_on_field(self, field):
        if not self.spin_flip_even:
            return self._raw_components_on_field(field)
        batch = field.shape[0]
        direction, amplitude = self._raw_components_on_field(
            torch.cat([field, -field], dim=0)
        )
        direction = direction.reshape(
            2,
            batch,
            self.n_factors,
            *field.shape[-2:],
        ).mean(dim=0)
        if amplitude is not None:
            amplitude = amplitude.reshape(
                2,
                batch,
                *field.shape[-2:],
            ).mean(dim=0)
        return direction, amplitude

    def forward(self, patches):
        if not self.d4_equivariant:
            direction, amplitude = self._even_components(patches)
        else:
            group_size = len(self.d4_transformations)
            batch = patches.shape[0]
            transformed = torch.cat(
                [
                    self._transform_patches(patches, rotations, reflected)
                    for rotations, reflected in self.d4_transformations
                ],
                dim=0,
            )
            direction, amplitude = self._even_components(transformed)
            direction = direction.reshape(
                group_size,
                batch,
                self.n_factors,
            )
            permutations = self.d4_permutations[:, None, :].expand(
                -1,
                batch,
                -1,
            )
            direction = direction.gather(2, permutations).mean(dim=0)
            if amplitude is not None:
                amplitude = amplitude.reshape(group_size, batch).mean(dim=0)
        if self.parameterization == "direct":
            return direction
        shape = (
            direction / direction.square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        )
        return amplitude[:, None] * shape

    def coefficients_on_patch_grid(self, patches):
        batch, rows = patches.shape[:2]
        coefficients = self.forward(patches.reshape(batch * rows, *patches.shape[2:]))
        return coefficients.reshape(batch, rows, self.n_factors)

    def local_factor_matrix(self, coefficients):
        expected = (self.n_likelihood_rows, self.n_factors)
        if coefficients.shape[1:] != expected:
            raise ValueError("coefficient grid does not match likelihood geometry")
        batch = coefficients.shape[0]
        matrix = coefficients.new_zeros(
            batch,
            self.n_likelihood_rows,
            self.n_likelihood_columns,
        )
        indices = self.local_column_index.unsqueeze(0).expand(batch, -1, -1)
        matrix.scatter_(2, indices, coefficients)
        return matrix

    def local_covariance(self, coefficients):
        factor = self.local_factor_matrix(coefficients)
        return factor @ factor.transpose(-1, -2)

    def coefficients_on_field(self, field):
        if field.ndim != 4 or field.shape[1] != 1:
            raise ValueError("field must have shape (batch, 1, height, width)")
        if not self.d4_equivariant and self.patch_size // 2 >= self.receptive_radius:
            direction, amplitude = self._even_components_on_field(field)
            if self.parameterization == "direct":
                return direction
            shape = (
                direction
                / direction.square()
                .sum(
                    dim=1,
                    keepdim=True,
                )
                .clamp_min(1e-12)
                .sqrt()
            )
            return amplitude[:, None] * shape
        return self._coefficients_on_field_unfold(field)

    def _coefficients_on_field_unfold(self, field):
        padding = self.patch_size // 2
        padded = F.pad(
            field,
            (padding, padding, padding, padding),
            mode="circular",
        )
        patches = F.unfold(padded, kernel_size=self.patch_size)
        batch, _, sites = patches.shape
        patches = patches.transpose(1, 2).reshape(
            batch * sites,
            1,
            self.patch_size,
            self.patch_size,
        )
        coefficients = self.forward(patches)
        height, width = field.shape[-2:]
        return coefficients.reshape(
            batch,
            height,
            width,
            self.n_factors,
        ).permute(0, 3, 1, 2)

    def divergence_coloring_probes(self, field):
        if field.ndim != 4 or field.shape[1] != 1:
            raise ValueError("field must have shape (batch, 1, height, width)")
        dependency_radius = self.receptive_radius + 2 * self.factor_radius
        height, width = field.shape[-2:]
        periods = [
            period
            for period in range(dependency_radius + 1, min(height, width) + 1)
            if height % period == 0 and width % period == 0
        ]
        if not periods:
            raise ValueError("field dimensions do not admit an exact periodic coloring")
        period = periods[0]
        y = torch.arange(height, device=field.device)[:, None]
        x = torch.arange(width, device=field.device)[None, :]
        probes = [
            ((y % period == color_y) & (x % period == color_x)).to(field.dtype)
            for color_y in range(period)
            for color_x in range(period)
        ]
        probes = torch.stack(probes)[:, None, None]
        return probes.expand(-1, field.shape[0], -1, -1, -1).contiguous()

    def divergence_on_field_colored_exact(
        self,
        field,
        probes=None,
        probe_chunk_size=64,
    ):
        if self.d4_equivariant or self.patch_size // 2 < self.receptive_radius:
            return self.divergence_on_field_exact(field)
        if probes is None:
            probes = self.divergence_coloring_probes(field)
        if probes.shape[1:] != (
            field.shape[0],
            1,
            *field.shape[-2:],
        ):
            raise ValueError("color probes do not match the field")

        def one_probe(probe):
            def action(state):
                coefficients = self.coefficients_on_field(state)
                return self.apply_dissipation(coefficients, probe[:, 0])

            return torch.func.jvp(
                action,
                (field,),
                (probe,),
            )[1]

        divergence = field.new_zeros(
            field.shape[0],
            *field.shape[-2:],
        )
        for start in range(0, probes.shape[0], probe_chunk_size):
            divergence = divergence + torch.vmap(one_probe)(
                probes[start : start + probe_chunk_size]
            ).sum(dim=0)
        return divergence

    def apply_factor(self, coefficients, noise):
        if coefficients.ndim != 4 or coefficients.shape[1] != self.n_factors:
            raise ValueError("coefficient field has the wrong shape")
        if noise.shape != coefficients.shape[:1] + coefficients.shape[2:]:
            raise ValueError("noise field has the wrong shape")
        result = torch.zeros_like(noise)
        for index, (dy, dx) in enumerate(self.factor_offsets):
            result = result + coefficients[:, index] * torch.roll(
                noise,
                shifts=(-dy, -dx),
                dims=(-2, -1),
            )
        return result

    def apply_adjoint(self, coefficients, force):
        if coefficients.ndim != 4 or coefficients.shape[1] != self.n_factors:
            raise ValueError("coefficient field has the wrong shape")
        if force.shape != coefficients.shape[:1] + coefficients.shape[2:]:
            raise ValueError("force field has the wrong shape")
        result = torch.zeros_like(force)
        for index, (dy, dx) in enumerate(self.factor_offsets):
            result = result + torch.roll(
                coefficients[:, index] * force,
                shifts=(dy, dx),
                dims=(-2, -1),
            )
        return result

    def apply_dissipation(self, coefficients, force):
        return self.apply_factor(
            coefficients,
            self.apply_adjoint(coefficients, force),
        )

    def _center_covariance_row(self, coefficients):
        if coefficients.shape[1:] != (self.n_force_rows, self.n_factors):
            raise ValueError("coefficient grid does not match force geometry")
        indices = self.force_factor_index.unsqueeze(0).expand(coefficients.shape[0], -1, -1)
        neighboring = coefficients.gather(2, indices)
        neighboring = neighboring * self.force_factor_mask.unsqueeze(0)
        center = coefficients[:, self.center_force_index]
        return (neighboring * center.unsqueeze(1)).sum(dim=-1)

    def divergence_at_center_exact(self, context):
        expected_side = self.patch_size + 2 * self.force_radius
        if context.shape[-2:] != (expected_side, expected_side):
            raise ValueError("context size does not match operator geometry")
        center = expected_side // 2
        state = context.detach().requires_grad_(True)
        patches = F.unfold(state, kernel_size=self.patch_size)
        patches = patches.transpose(1, 2).reshape(
            state.shape[0],
            self.n_force_rows,
            1,
            self.patch_size,
            self.patch_size,
        )
        covariance_row = self._center_covariance_row(
            self.coefficients_on_patch_grid(patches)
        )
        divergence = state.new_zeros(state.shape[0])
        for index, (dy, dx) in enumerate(self.force_offsets):
            gradient = torch.autograd.grad(
                covariance_row[:, index].sum(),
                state,
                retain_graph=index + 1 < self.n_force_rows,
            )[0]
            divergence = divergence + gradient[:, 0, center + dy, center + dx]
        return divergence

    def divergence_on_field_exact(self, field, chunk_size=512):
        if field.ndim != 4 or field.shape[1] != 1:
            raise ValueError("field must have shape (batch, 1, height, width)")
        if self.factor_radius == 0:
            return self._pointwise_divergence_exact(field, chunk_size)

        patch_radius = self.patch_size // 2
        padded = F.pad(
            field,
            (patch_radius, patch_radius, patch_radius, patch_radius),
            mode="circular",
        )
        patches = F.unfold(padded, kernel_size=self.patch_size).transpose(1, 2)
        batch, sites, _ = patches.shape
        patches = patches.reshape(
            batch * sites,
            1,
            self.patch_size,
            self.patch_size,
        )
        coefficients = (
            self.forward(patches)
            .reshape(
                batch,
                *field.shape[-2:],
                self.n_factors,
            )
            .permute(0, 3, 1, 2)
        )

        selected_jacobians = []
        for start in range(0, patches.shape[0], chunk_size):
            state = patches[start : start + chunk_size].detach().requires_grad_(True)
            values = self.forward(state)
            output_jacobians = []
            for factor_index in range(self.n_factors):
                gradient = torch.autograd.grad(
                    values[:, factor_index].sum(),
                    state,
                    retain_graph=factor_index + 1 < self.n_factors,
                )[0]
                output_jacobians.append(
                    torch.stack(
                        [
                            gradient[:, 0, patch_radius + dy, patch_radius + dx]
                            for dy, dx in self.force_offsets
                        ],
                        dim=-1,
                    )
                )
            selected_jacobians.append(torch.stack(output_jacobians, dim=1))
        selected_jacobians = (
            torch.cat(selected_jacobians)
            .reshape(
                batch,
                *field.shape[-2:],
                self.n_factors,
                self.n_force_rows,
            )
            .permute(0, 3, 4, 1, 2)
        )

        factor_lookup = {offset: index for index, offset in enumerate(self.factor_offsets)}
        divergence = field.new_zeros(batch, *field.shape[-2:])
        for force_index, (dy, dx) in enumerate(self.force_offsets):
            for first_index, (factor_y, factor_x) in enumerate(self.factor_offsets):
                second_index = factor_lookup.get((factor_y - dy, factor_x - dx))
                if second_index is None:
                    continue
                shifted_second = torch.roll(
                    coefficients[:, second_index],
                    shifts=(-dy, -dx),
                    dims=(-2, -1),
                )
                shifted_center_derivative = torch.roll(
                    selected_jacobians[:, second_index, self.center_force_index],
                    shifts=(-dy, -dx),
                    dims=(-2, -1),
                )
                divergence = divergence + (
                    selected_jacobians[:, first_index, force_index] * shifted_second
                    + coefficients[:, first_index] * shifted_center_derivative
                )
        return divergence

    def _pointwise_divergence_exact(self, field, chunk_size):
        padding = self.patch_size // 2
        padded = F.pad(
            field,
            (padding, padding, padding, padding),
            mode="circular",
        )
        patches = F.unfold(padded, kernel_size=self.patch_size)
        batch, _, sites = patches.shape
        patches = patches.transpose(1, 2).reshape(
            batch * sites,
            1,
            self.patch_size,
            self.patch_size,
        )
        center = self.patch_size // 2
        derivatives = []
        for start in range(0, patches.shape[0], chunk_size):
            local = patches[start : start + chunk_size]
            direction = torch.zeros_like(local)
            direction[:, 0, center, center] = 1.0

            def diagonal_dissipation(state):
                return self.forward(state)[:, 0].square()

            _, derivative = torch.func.jvp(
                diagonal_dissipation,
                (local,),
                (direction,),
            )
            derivatives.append(derivative)
        return torch.cat(derivatives).reshape(batch, *field.shape[-2:])
