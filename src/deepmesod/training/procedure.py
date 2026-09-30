import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from deepmesod.classes import get_trainer, load_extensions


def config_sha256(config):
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_revision(path):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@dataclass(frozen=True)
class StageSpec:
    name: str
    trainer: str
    loss: str
    requires: tuple[str, ...]
    config: dict

    @classmethod
    def from_mapping(cls, value):
        return cls(
            name=value["name"],
            trainer=value["trainer"],
            loss=value["loss"],
            requires=tuple(value.get("requires", ())),
            config=value.get("config", {}),
        )


class RunContext:
    def __init__(self, config, output_dir, *, resume=False):
        self.config = config
        load_extensions(config.get("extensions"), config.get("_config_dir"))
        self.output_dir = Path(output_dir)
        self.artifacts = {}
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            if not resume:
                raise FileExistsError(f"run output is not empty: {self.output_dir}")
            provenance_path = self.output_dir / "provenance.json"
            if not provenance_path.is_file():
                raise FileNotFoundError(
                    f"cannot resume without provenance: {provenance_path}"
                )
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("config_sha256") != config_sha256(config):
                raise ValueError("resume configuration does not match the existing run")
            self.artifacts.update(provenance.get("artifacts", {}))
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, name):
        directory = self.output_dir / name
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"stage output is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def publish(self, stage, artifacts):
        self.artifacts[stage] = artifacts


class TrainingProcedure:
    def __init__(self, stages):
        self.stages = tuple(StageSpec.from_mapping(stage) for stage in stages)
        self._validate()

    def _validate(self):
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("training stage names must be unique")
        available = set()
        for stage in self.stages:
            missing = set(stage.requires) - available
            if missing:
                raise ValueError(
                    f"stage '{stage.name}' requires unavailable stages {sorted(missing)}"
                )
            available.add(stage.name)

    def run(self, context, *, only=None):
        selected = set(only or [stage.name for stage in self.stages])
        resolved = context.output_dir / "resolved_config.json"
        resolved.write_text(
            json.dumps(context.config, indent=2, sort_keys=True), encoding="utf-8"
        )
        for stage in self.stages:
            if stage.name not in selected:
                continue
            available = set(context.artifacts) | set(context.config.get("artifacts", {}))
            missing = set(stage.requires) - available
            if missing:
                raise RuntimeError(
                    f"stage '{stage.name}' is missing artifacts from {sorted(missing)}"
                )
            trainer = get_trainer(stage.trainer)
            artifacts = trainer(context, stage)
            context.publish(stage.name, artifacts)
        self._write_provenance(context)
        return context.artifacts

    def _write_provenance(self, context):
        payload = {
            "config_sha256": config_sha256(context.config),
            "run_seed": context.config.get("run", {}).get("seed"),
            "source_revision": source_revision(context.config.get("_config_dir", ".")),
            "python": platform.python_version(),
            "stages": [stage.name for stage in self.stages],
            "artifacts": context.artifacts,
        }
        (context.output_dir / "provenance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
