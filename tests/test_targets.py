import torch

from deepmesod.training.targets import batched_conjugate_gradient


def test_batched_conjugate_gradient_solves_spd_systems():
    diagonal = torch.tensor([1.0, 2.0, 4.0])[None, :, None]
    right_hand_side = torch.tensor([[[1.0], [2.0], [8.0]]])

    def action(values):
        return diagonal * values

    solution, _, relative = batched_conjugate_gradient(
        action, right_hand_side, iterations=10, tolerance=1e-8
    )
    expected = torch.tensor([[[1.0], [1.0], [2.0]]])
    assert torch.allclose(solution, expected, atol=1e-6)
    assert float(relative.max()) < 1e-6
