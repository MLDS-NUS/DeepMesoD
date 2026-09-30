import torch
import torch.nn.functional as F


def square_offsets(radius):
    return [
        (dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
    ]


def shift_at(field, offset):
    dy, dx = offset
    return torch.roll(field, shifts=(-dy, -dx), dims=(-2, -1))


def _convolution_offsets(layer):
    height, width = layer.kernel_size
    dilation_y, dilation_x = layer.dilation
    center_y = height // 2
    center_x = width // 2
    return [
        (
            (kernel_y - center_y) * dilation_y,
            (kernel_x - center_x) * dilation_x,
            kernel_y,
            kernel_x,
        )
        for kernel_y in range(height)
        for kernel_x in range(width)
    ]


def validate_exact_local_divergence(factor):
    if factor.d4_equivariant:
        raise ValueError("analytic divergence does not support D4 averaging")
    if factor.n_layers != 3:
        raise ValueError("analytic divergence requires three convolution layers")
    if factor.activation_name != "tanh":
        raise ValueError("analytic divergence currently requires tanh activations")
    if len(factor.net) != 5:
        raise ValueError("unexpected factor network structure")
    if factor.factor_radius != 1:
        raise ValueError("analytic divergence currently requires factor radius one")


def _raw_output_and_selected_jacobian(factor, field, selected_offsets):
    first_convolution = factor.net[0]
    second_convolution = factor.net[2]
    third_convolution = factor.net[4]

    first_pre_activation = first_convolution(field)
    first_activation = torch.tanh(first_pre_activation)
    second_pre_activation = second_convolution(first_activation)
    second_activation = torch.tanh(second_pre_activation)
    raw_output = third_convolution(second_activation)

    first_activation_derivative = 1.0 - first_activation.square()
    second_activation_derivative = 1.0 - second_activation.square()

    first_offsets = [(dy, dx) for dy, dx, _, _ in _convolution_offsets(first_convolution)]
    first_derivatives = torch.stack(
        [
            first_activation_derivative
            * first_convolution.weight[:, 0, kernel_y, kernel_x][None, :, None, None]
            for _, _, kernel_y, kernel_x in _convolution_offsets(first_convolution)
        ],
        dim=2,
    )

    second_offsets = sorted(
        {
            (first_y + second_y, first_x + second_x)
            for first_y, first_x in first_offsets
            for second_y, second_x, _, _ in _convolution_offsets(second_convolution)
        }
    )
    second_lookup = {offset: index for index, offset in enumerate(second_offsets)}
    second_derivatives = field.new_zeros(
        field.shape[0],
        second_convolution.out_channels,
        len(second_offsets),
        *field.shape[-2:],
    )
    for second_y, second_x, kernel_y, kernel_x in _convolution_offsets(second_convolution):
        shifted = shift_at(first_derivatives, (second_y, second_x))
        contribution = torch.einsum(
            "ba,naohw->nbohw",
            second_convolution.weight[:, :, kernel_y, kernel_x],
            shifted,
        )
        indices = torch.tensor(
            [
                second_lookup[(first_y + second_y, first_x + second_x)]
                for first_y, first_x in first_offsets
            ],
            device=field.device,
        )
        second_derivatives[:, :, indices] += contribution
    second_derivatives *= second_activation_derivative[:, :, None]

    selected_lookup = {offset: index for index, offset in enumerate(selected_offsets)}
    raw_derivatives = field.new_zeros(
        field.shape[0],
        third_convolution.out_channels,
        len(selected_offsets),
        *field.shape[-2:],
    )
    for third_y, third_x, kernel_y, kernel_x in _convolution_offsets(third_convolution):
        valid = [
            (
                selected_lookup[selected],
                second_lookup[(selected[0] - third_y, selected[1] - third_x)],
            )
            for selected in selected_offsets
            if (
                selected[0] - third_y,
                selected[1] - third_x,
            )
            in second_lookup
        ]
        if not valid:
            continue
        output_indices = torch.tensor(
            [output for output, _ in valid],
            device=field.device,
        )
        input_indices = torch.tensor(
            [source for _, source in valid],
            device=field.device,
        )
        shifted = shift_at(
            second_derivatives[:, :, input_indices],
            (third_y, third_x),
        )
        contribution = torch.einsum(
            "cb,nbqhw->ncqhw",
            third_convolution.weight[:, :, kernel_y, kernel_x],
            shifted,
        )
        raw_derivatives[:, :, output_indices] += contribution
    return raw_output, raw_derivatives


def coefficients_and_selected_jacobian(factor, field):
    validate_exact_local_divergence(factor)
    selected_offsets = square_offsets(2 * factor.factor_radius)
    if factor.spin_flip_even:
        raw, raw_derivatives = _raw_output_and_selected_jacobian(
            factor,
            torch.cat([field, -field], dim=0),
            selected_offsets,
        )
        batch = field.shape[0]
        raw = raw.reshape(2, batch, *raw.shape[1:])
        raw_derivatives = raw_derivatives.reshape(
            2,
            batch,
            *raw_derivatives.shape[1:],
        )
        derivative_sign = raw_derivatives.new_tensor([1.0, -1.0]).reshape(2, 1, 1, 1, 1, 1)
    else:
        batch = field.shape[0]
        raw, raw_derivatives = _raw_output_and_selected_jacobian(
            factor,
            field,
            selected_offsets,
        )
        raw = raw[None]
        raw_derivatives = raw_derivatives[None]
        derivative_sign = raw_derivatives.new_ones(1, 1, 1, 1, 1, 1)

    raw_direction = raw[:, :, : factor.n_factors]
    direction_derivatives = raw_derivatives[:, :, : factor.n_factors].clone()
    center = factor.center_factor_index
    direction = raw_direction.clone()
    direction[:, :, center] = F.softplus(raw_direction[:, :, center]) + factor.min_center
    direction_derivatives[:, :, center] *= torch.sigmoid(raw_direction[:, :, center])[
        :, :, None
    ]

    direction = direction.mean(dim=0)
    direction_derivatives = (direction_derivatives * derivative_sign).mean(dim=0)
    if factor.parameterization == "direct":
        return direction, direction_derivatives, selected_offsets

    raw_amplitude = raw[:, :, factor.n_factors]
    amplitude = F.softplus(raw_amplitude) + factor.min_center
    amplitude_derivatives = (
        raw_derivatives[:, :, factor.n_factors] * torch.sigmoid(raw_amplitude)[:, :, None]
    )
    amplitude = amplitude.mean(dim=0)
    amplitude_derivatives = (amplitude_derivatives * derivative_sign[:, :, 0]).mean(dim=0)

    norm_squared = direction.square().sum(dim=1, keepdim=True)
    clamped = norm_squared.clamp_min(1e-12)
    inverse_norm = clamped.rsqrt()
    shape = direction * inverse_norm
    directional_projection = (direction[:, :, None] * direction_derivatives).sum(
        dim=1, keepdim=True
    )
    active = (norm_squared > 1e-12).to(direction.dtype)
    shape_derivatives = (
        direction_derivatives * inverse_norm[:, :, None]
        - direction[:, :, None]
        * directional_projection
        * inverse_norm[:, :, None].pow(3)
        * active[:, :, None]
    )
    coefficients = amplitude[:, None] * shape
    coefficient_derivatives = (
        amplitude_derivatives[:, None] * shape[:, :, None]
        + amplitude[:, None, None] * shape_derivatives
    )
    return coefficients, coefficient_derivatives, selected_offsets


def divergence_on_field_local_exact(factor, field, return_coefficients=False):
    coefficients, derivatives, derivative_offsets = coefficients_and_selected_jacobian(
        factor, field
    )
    derivative_lookup = {offset: index for index, offset in enumerate(derivative_offsets)}
    center_derivative = derivative_lookup[(0, 0)]
    factor_lookup = {offset: index for index, offset in enumerate(factor.factor_offsets)}
    divergence = field.new_zeros(field.shape[0], *field.shape[-2:])
    for first_offset, first_index in factor_lookup.items():
        for second_offset, second_index in factor_lookup.items():
            row_offset = (
                first_offset[0] - second_offset[0],
                first_offset[1] - second_offset[1],
            )
            row_derivative = derivative_lookup[row_offset]
            divergence += derivatives[:, first_index, row_derivative] * shift_at(
                coefficients[:, second_index], row_offset
            ) + coefficients[:, first_index] * shift_at(
                derivatives[:, second_index, center_derivative],
                row_offset,
            )
    if return_coefficients:
        return divergence, coefficients
    return divergence
