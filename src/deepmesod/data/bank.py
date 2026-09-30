import torch

from .sampling import BinnedFieldSampler, UniformFieldSampler, periodic_patches
from .schema import inverse_temperature, load_trajectories


class ScalarTrajectoryBank:
    def __init__(
        self,
        paths,
        transitions_per_trajectory,
        *,
        bins=10,
        sampling="joint",
        patch_size=15,
        seed=42,
    ):
        self.trajectories = load_trajectories(paths, transitions_per_trajectory)
        settings = {
            (
                item.field.shape[-2:],
                inverse_temperature(item.metadata),
                float(item.metadata["coarse_volume"]),
                float(item.metadata.get("external_field", 0)),
            )
            for item in self.trajectories
        }
        if len(settings) != 1:
            raise ValueError(
                "all trajectories in a bank must share shape and physical metadata"
            )
        shape, self.beta, self.coarse_volume, self.external_field = settings.pop()
        self.height, self.width = shape
        self.generator = torch.Generator().manual_seed(seed)
        if sampling not in {"none", "joint", "center"}:
            raise ValueError("sampling must be 'none', 'joint', or 'center'")
        if sampling == "none":
            self.samplers = [
                UniformFieldSampler(trajectory.field[:-1], generator=self.generator)
                for trajectory in self.trajectories
            ]
        else:
            local_mean_size = patch_size if sampling == "joint" else None
            self.samplers = [
                BinnedFieldSampler(
                    trajectory.field[:-1],
                    bins,
                    local_mean_size=local_mean_size,
                    generator=self.generator,
                )
                for trajectory in self.trajectories
            ]

    def sample(self, batch_size, patch_size):
        trajectory_index = int(
            torch.randint(len(self.trajectories), (1,), generator=self.generator).item()
        )
        trajectory = self.trajectories[trajectory_index]
        time, row, column = self.samplers[trajectory_index].sample(batch_size)
        current, _, _, rate = trajectory.transitions
        return (
            periodic_patches(current, time, row, column, patch_size),
            rate[time, row, column],
        )

    def sample_window(self, batch_size, patch_size, window_radius):
        trajectory_index = int(
            torch.randint(len(self.trajectories), (1,), generator=self.generator).item()
        )
        trajectory = self.trajectories[trajectory_index]
        time, row, column = self.samplers[trajectory_index].sample(batch_size)
        current, following, dt, _ = trajectory.transitions
        context_size = patch_size + 2 * window_radius
        context = periodic_patches(current, time, row, column, context_size)
        window_size = 2 * window_radius + 1
        current_window = periodic_patches(current, time, row, column, window_size).flatten(
            1
        )
        following_window = periodic_patches(
            following, time, row, column, window_size
        ).flatten(1)
        return context, current_window, following_window, dt[time].float()
