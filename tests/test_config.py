from deepmesod.config import apply_overrides


def test_override_supports_ordered_stage_lists(tmp_path):
    config = {
        "_config_dir": str(tmp_path),
        "training": {"procedure": [{"loss": "first"}, {"loss": "second"}]},
    }
    updated = apply_overrides(config, ["training.procedure.1.loss=replacement"])
    assert updated["training"]["procedure"][1]["loss"] == "replacement"
