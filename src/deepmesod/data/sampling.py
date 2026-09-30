import torch
import torch.nn.functional as F


def flat_coordinates(flat, height, width):
    plane = height * width
    time = flat // plane
    within = flat % plane
    return time, within // width, within % width


class BinnedFieldSampler:
    def __init__(self, field, bins=10, *, local_mean_size=None, generator=None):
        if field.ndim != 3:
            raise ValueError(
                "BinnedFieldSampler currently expects scalar (time, y, x) fields"
            )
        self.generator = generator
        center_code = ((field + 1) * 0.5 * bins).floor().clamp(0, bins - 1).long()
        if local_mean_size is None:
            code = center_code
            number_of_codes = bins
        else:
            radius = local_mean_size // 2
            local_mean = F.avg_pool2d(
                F.pad(field[:, None], (radius, radius, radius, radius), mode="circular"),
                kernel_size=local_mean_size,
                stride=1,
            )[:, 0]
            mean_code = ((local_mean + 1) * 0.5 * bins).floor().clamp(0, bins - 1).long()
            code = center_code * bins + mean_code
            number_of_codes = bins * bins
        code = code.reshape(-1)
        counts = torch.bincount(code, minlength=number_of_codes)
        self.order = torch.argsort(code)
        self.starts = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        self.valid = torch.where(counts > 0)[0]
        self.height = field.shape[-2]
        self.width = field.shape[-1]

    def sample(self, batch_size):
        choice = torch.randint(self.valid.numel(), (batch_size,), generator=self.generator)
        code = self.valid[choice]
        starts = self.starts[code]
        counts = self.starts[code + 1] - starts
        offsets = torch.floor(
            torch.rand(batch_size, generator=self.generator) * counts
        ).long()
        flat = self.order[starts + offsets]
        return flat_coordinates(flat, self.height, self.width)


class UniformFieldSampler:
    def __init__(self, field, *, generator=None):
        if field.ndim != 3:
            raise ValueError("UniformFieldSampler expects scalar (time, y, x) fields")
        self.generator = generator
        self.time = field.shape[0]
        self.height = field.shape[-2]
        self.width = field.shape[-1]

    def sample(self, batch_size):
        flat = torch.randint(
            self.time * self.height * self.width,
            (batch_size,),
            generator=self.generator,
        )
        return flat_coordinates(flat, self.height, self.width)


def periodic_patches(field, time, row, column, size):
    radius = size // 2
    offsets = torch.arange(-radius, radius + 1)
    rows = (row[:, None, None] + offsets[None, :, None]) % field.shape[-2]
    columns = (column[:, None, None] + offsets[None, None, :]) % field.shape[-1]
    return field[time[:, None, None], rows, columns].unsqueeze(1)
