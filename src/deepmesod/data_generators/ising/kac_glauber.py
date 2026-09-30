import json
from pathlib import Path

import numpy as np

from deepmesod.data_generators.registry import register_data_generator


def _array_module(device):
    if device == "cpu":
        return np
    try:
        import cupy
    except ImportError as error:
        raise RuntimeError("GPU Ising generation requires the ising-gpu extra") from error
    return cupy


def _spectral_field(size, correlation_length, amplitude, seed):
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(size, size))
    kx = 2 * np.pi * np.fft.fftfreq(size)
    ky = 2 * np.pi * np.fft.rfftfreq(size)
    wave_x, wave_y = np.meshgrid(kx, ky, indexing="ij")
    spectrum = amplitude * np.exp(-0.25 * correlation_length**2 * (wave_x**2 + wave_y**2))
    coefficients = spectrum * np.fft.rfftn(white)
    coefficients[0, 0] = 0
    field = np.fft.irfftn(coefficients, s=(size, size), axes=(0, 1)).real
    standard_deviation = field.std()
    if standard_deviation > 0:
        field *= amplitude / standard_deviation
    return field


def multiscale_initial_field(
    size,
    seed,
    *,
    large_length=256.0,
    small_length=24.0,
    large_amplitude=1.5,
    small_amplitude=2.0,
    small_weight=0.4,
    saturation=0.2,
):
    large = _spectral_field(size, large_length, large_amplitude, seed)
    small = _spectral_field(size, small_length, small_amplitude, seed + 10000)
    return np.tanh((large + small_weight * small) / saturation)


def sample_spins(mean_field, seed):
    rng = np.random.default_rng(seed)
    probability = 0.5 * (1 + np.clip(mean_field, -1, 1))
    return np.where(rng.random(mean_field.shape) < probability, 1, -1).astype(np.int8)


def block_average(field, block):
    if field.shape[-1] % block or field.shape[-2] % block:
        raise ValueError("fine-grid dimensions must be divisible by the block width")
    height = field.shape[-2] // block
    width = field.shape[-1] // block
    return field.reshape(height, block, width, block).mean(axis=(1, 3))


def fourier_kernel(size, epsilon, xp, domain_length_x=1.0, domain_length_y=1.0):
    spacing_x = domain_length_x / size
    spacing_y = domain_length_y / size
    kx = xp.fft.fftfreq(size, spacing_x) * 2 * xp.pi
    ky = xp.fft.fftfreq(size, spacing_y) * 2 * xp.pi
    wave_x, wave_y = xp.meshgrid(kx, ky, indexing="ij")
    kernel = xp.exp(-0.5 * epsilon**2 * (wave_x**2 + wave_y**2))
    return kernel / kernel[0, 0]


def simulate_tau_leaping(
    initial_spins,
    *,
    temperature,
    interaction_strength,
    external_field,
    epsilon,
    steps,
    block,
    tau_fraction,
    seed,
    device,
    domain_length_x=1.0,
    domain_length_y=1.0,
):
    xp = _array_module(device)
    if device != "cpu":
        xp.random.seed(seed)
    else:
        np.random.seed(seed)
    spins = xp.asarray(initial_spins)
    kernel = fourier_kernel(
        spins.shape[0],
        epsilon,
        xp,
        domain_length_x,
        domain_length_y,
    )
    real_kernel = xp.fft.ifft2(kernel).real
    self_weight = real_kernel[0, 0] / real_kernel.sum()
    beta = 1 / temperature
    coarse_size = spins.shape[0] // block
    fields = np.empty((steps + 1, coarse_size, coarse_size), dtype=np.float32)
    times = np.empty(steps + 1, dtype=np.float64)
    coarse = block_average(spins, block)
    fields[0] = np.asarray(coarse.get() if device != "cpu" else coarse)
    times[0] = 0
    for step in range(steps):
        convolved = xp.fft.ifft2(kernel * xp.fft.fft2(spins)).real
        local_field = external_field + interaction_strength * (
            convolved - self_weight * spins
        )
        rates = 1 / (1 + xp.exp(xp.clip(2 * beta * local_field * spins, -30, 30)))
        dt = tau_fraction / float(rates.max())
        flips = xp.random.poisson(rates * dt)
        spins *= xp.where(flips % 2, -1, 1)
        coarse = block_average(spins, block)
        if device != "cpu":
            coarse = coarse.get()
        fields[step + 1] = np.asarray(coarse)
        times[step + 1] = times[step] + dt
    return fields, times


@register_data_generator("ising_kac_tau_leaping")
def generate_trajectory(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    initial_mean = multiscale_initial_field(config["size"], config["seed"])
    spins = sample_spins(initial_mean, config["seed"])
    fields, times = simulate_tau_leaping(
        spins,
        temperature=config["temperature"],
        interaction_strength=config.get("interaction_strength", 1.0),
        external_field=config.get("external_field", 0.0),
        epsilon=config["epsilon"],
        steps=config["steps"],
        block=config["block"],
        tau_fraction=config.get("tau_fraction", 0.01),
        seed=config["seed"],
        device=config.get("device", "cuda"),
        domain_length_x=config.get("domain_length_x", 1.0),
        domain_length_y=config.get("domain_length_y", 1.0),
    )
    metadata = {
        "schema_version": 1,
        "system": "ising_kac",
        "dynamics": "continuous_time_glauber_tau_leaping",
        "boundary": "periodic",
        "temperature": config["temperature"],
        "beta": 1 / config["temperature"],
        "coarse_volume": config["block"] ** 2,
        "fine_size": config["size"],
        "block": config["block"],
        "epsilon": config["epsilon"],
        "domain_length_x": config.get("domain_length_x", 1.0),
        "domain_length_y": config.get("domain_length_y", 1.0),
        "tau_fraction": config.get("tau_fraction", 0.01),
        "interaction_strength": config.get("interaction_strength", 1.0),
        "external_field": config.get("external_field", 0.0),
        "seed": config["seed"],
    }
    np.save(output_dir / "field.npy", fields)
    np.save(output_dir / "times.npy", times)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output_dir
