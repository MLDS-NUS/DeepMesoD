import numpy as np

from deepmesod.data import load_trajectory
from deepmesod.data_generators.ising import generate_trajectory
from deepmesod.data_generators.ising.kac_glauber import fourier_kernel


def test_fourier_kernel_uses_physical_domain_length():
    kernel = fourier_kernel(8, 0.2, np, 1.0, 1.0)
    expected = np.exp(-0.5 * (2 * np.pi * 0.2) ** 2)
    assert np.isclose(kernel[1, 0], expected)


def test_cpu_ising_generator_emits_valid_trajectory(tmp_path):
    output = generate_trajectory(
        {
            "size": 8,
            "block": 2,
            "steps": 2,
            "temperature": 1.5,
            "interaction_strength": 1.0,
            "external_field": 0.0,
            "epsilon": 0.125,
            "domain_length_x": 1.0,
            "domain_length_y": 1.0,
            "tau_fraction": 0.01,
            "seed": 7,
            "device": "cpu",
        },
        tmp_path / "trajectory",
    )
    trajectory = load_trajectory(output)
    assert trajectory.field.shape == (3, 4, 4)
    assert trajectory.metadata["coarse_volume"] == 4
    assert trajectory.metadata["external_field"] == 0.0
