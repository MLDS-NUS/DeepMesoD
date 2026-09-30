from deepmesod.classes import register_loss, register_trainer
from deepmesod.training import RunContext, TrainingProcedure


@register_loss("test_identity_loss")
def identity_loss(prediction, target, weight=None):
    return prediction - target


@register_trainer("test_artifact_trainer")
def artifact_trainer(context, stage):
    output = context.stage_dir(stage.name) / "artifact.txt"
    output.write_text(stage.config["value"], encoding="utf-8")
    return {"artifact": str(output)}


def test_configurable_procedure_respects_dependencies(tmp_path):
    procedure = TrainingProcedure(
        [
            {
                "name": "first",
                "trainer": "test_artifact_trainer",
                "loss": "test_identity_loss",
                "config": {"value": "first"},
            },
            {
                "name": "second",
                "trainer": "test_artifact_trainer",
                "loss": "test_identity_loss",
                "requires": ["first"],
                "config": {"value": "second"},
            },
        ]
    )
    context = RunContext({"run": {"seed": 1}}, tmp_path)
    result = procedure.run(context)
    assert set(result) == {"first", "second"}
    assert (tmp_path / "second" / "artifact.txt").read_text() == "second"
    assert (tmp_path / "provenance.json").exists()
