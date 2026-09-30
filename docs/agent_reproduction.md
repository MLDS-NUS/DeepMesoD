# Agent workflow

An agent can operate this repository without author data or checkpoints. The user
supplies trajectories satisfying `docs/data_contract.md` and chooses a YAML file.

```bash
deepmesod validate --config CONFIG
deepmesod reproduce --config CONFIG
```

The second command requires a fresh output directory and runs the declared
training procedure, inference, evaluation, and diagnostic report. Private
artifacts stay under the configured output directory and are excluded from the
source release.

Read the resulting files in this order:

1. `resolved_config.json` for the actual data, model, sampling, constraints, seed,
   and Stage 3 objective.
2. `provenance.json` for the configuration hash and final artifacts.
3. `report/report.json` for machine-readable diagnostics.
4. `report/report.md` and `report/diagnostics.pdf` for interpretation.

The report can establish whether the run remained finite, which checkpoint was
used, whether declared interaction constraints were satisfied, how the complete
dissipation response varies over sampled states, whether CG met its tolerance,
and how the rollout compares with the supplied test ensemble. It cannot establish
that the selected field, free-energy ansatz, or dissipation representation is
physically complete.

For publication claims, add the relevant microscopic, analytical, or baseline
reference through an experiment-specific manifest. The generic report covers one
model seed and its dissipation ribbon is within-bin site variation, not
training-seed uncertainty.
