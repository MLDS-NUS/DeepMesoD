import torch


def trajectory_metrics(prediction, reference):
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes differ")
    spatial_dims = tuple(range(1, prediction.ndim))
    absolute = (prediction - reference).abs().mean(dim=spatial_dims)
    return {
        "mae_by_time": absolute,
        "mean_mae": float(absolute.mean()),
        "final_mae": float(absolute[-1]),
        "finite": bool(torch.isfinite(prediction).all()),
    }
