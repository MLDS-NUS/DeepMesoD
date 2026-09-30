import torch
from torch import nn

from .drift import MLP, InteractionKernel


class EvenScalarFreeEnergy(nn.Module):
    def __init__(self, hidden=(128, 128, 128), activation_name="softplus"):
        super().__init__()
        self.mlp = MLP(1, hidden, 1, activation_name)

    def forward(self, state):
        positive = self.mlp(state[..., None]).squeeze(-1)
        negative = self.mlp((-state)[..., None]).squeeze(-1)
        return 0.5 * (positive + negative)

    def derivative(self, state, *, create_graph):
        state = state.detach().requires_grad_(True)
        with torch.enable_grad():
            energy = self(state)
            return torch.autograd.grad(
                energy.sum(), state, create_graph=create_graph, retain_graph=create_graph
            )[0]


class ScalarFreeEnergyInteraction(nn.Module):
    def __init__(
        self,
        patch_size=15,
        hidden=(128, 128, 128),
        activation_name="softplus",
        external_field=0.0,
        interaction_constraint="centrosymmetric_sum_one",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.free_energy = EvenScalarFreeEnergy(hidden, activation_name)
        self.interaction = InteractionKernel(
            patch_size,
            constraint=interaction_constraint,
            initialization="uniform",
        )
        self.register_buffer("external_field", torch.tensor(float(external_field)))

    def force_on_patches(self, patches, *, create_graph):
        center_index = self.patch_size // 2
        state = patches[:, 0, center_index, center_index]
        derivative = self.free_energy.derivative(state, create_graph=create_graph)
        convolved = (patches * self.interaction.weight).sum(dim=(1, 2, 3))
        interaction_action = convolved - state
        return derivative - interaction_action - self.external_field
