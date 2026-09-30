import json

import numpy as np
import pandas as pd
import torch

from experiments.paper.figure3_2D_ferromagnet import generate_figure3_rollout
from deepmesod.models import VectorPsiJModel
from deepmesod.operators import VectorFiniteRangeFactor


def test_figure3_evaluation_writes_visualization_schema(tmp_path):
    torch.manual_seed(7)
    factor = VectorFiniteRangeFactor(
        patch_size=5,
        factor_radius=1,
        likelihood_radius=1,
        hidden_channels=2,
        n_layers=2,
    )
    factor_path = tmp_path / "factor.pt"
    torch.save(
        {
            "model_config": factor.configuration(),
            "model_state_dict": factor.state_dict(),
            "beta": 0.1,
            "coarse_volume": 4.0,
            "external_field": 0.0,
        },
        factor_path,
    )

    thermodynamics = VectorPsiJModel(
        patch_size=3, kernel_size=3, mlp_hidden=(4,), jm_mode="raw"
    )
    stage3_path = tmp_path / "stage3.pt"
    torch.save(
        {
            "model_state_dict": thermodynamics.state_dict(),
            "beta": 0.1,
            "coarse_volume": 4.0,
            "external_field": 0.0,
        },
        stage3_path,
    )

    initial_path = tmp_path / "initial.npy"
    np.save(initial_path, np.random.default_rng(4).normal(size=(6, 6, 3)).astype("f4"))
    result = generate_figure3_rollout(
        stage3_path,
        factor_path,
        initial_path,
        tmp_path / "figure3",
        t_start=20,
        t_end=22,
        dt=1,
        divk_every=0,
        device="cpu",
        progress_every=0,
    )

    assert result["trajectory"].name == "pred_traj_t20_22.npy"
    assert result["trajectory"].parent == tmp_path / "figure3/data/rollout_deepmesoD"
    assert np.load(result["trajectory"]).shape == (3, 6, 6, 3)
    assert len(pd.read_csv(result["macro_statistics"])) == 3
    metadata = json.loads(result["metadata"].read_text(encoding="utf-8"))
    assert metadata["stage3_objective"] == "direct"
    assert metadata["divk_every"] == 0
