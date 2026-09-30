---
name: use-deepmesod
description: Configure, train, evaluate, extend, or audit DeepMesoD mesoscopic field models. Use for preparing trajectory data, selecting a dynamical class, choosing direct versus CG thermodynamic training, adding Model A/B/H or vector-field components, and interpreting learned operators and generated reports.
---

# Use DeepMesoD

## Start safely

1. Read the repository `AGENTS.md` and the guide relevant to the requested task.
2. Inspect the requested YAML and its declared manifest. Do not infer scientific
   settings from neighboring experiments.
3. Preflight the entire run with `deepmesod validate --config CONFIG`.
4. Run `configs/smoke.yaml` after changes to scientific code.
5. Use a fresh private output directory. Never add trajectories, inverse targets,
   checkpoints, rollouts, generated arrays, figures, or reports to Git.

## Choose the task

- For a new dataset, read `references/data.md`.
- For a new equation, field type, conservation law, or training order, read
  `references/extensions.md`.
- For training or comparing Stage 3 modes, read `references/training.md`.
- For interpreting `J`, `B`, `K`, free energy, or the Itô term, read
  `references/operators.md`.
- For running a complete workflow or interpreting its generated report, read
  `references/report.md`.

## Preserve scientific invariants

- Form single forward increments within each trajectory before combining data.
- Treat the configured dynamical class as fixed during a matched experiment.
- Keep `K = B B*` and evaluate the complete operator, not only its diagonal.
- Retain the configured finite-cell Itô state divergence in both training and
  rollout unless the run is explicitly labeled as an ablation.
- Do not introduce symmetry, positivity, radial, locality, or normalization
  constraints that the selected class does not declare.
- Use all declared model seeds and the final scheduled checkpoint. Do not choose
  a checkpoint or seed from the plotted rollout.

## Finish

For a complete run, use `deepmesod reproduce --config CONFIG`. Read the resolved
configuration, provenance, JSON report, Markdown report, and rendered diagnostic
figure in that order. Run tests and scan the source release for private paths,
credentials, datasets, checkpoints, results, figures, caches, and non-English
repository text.
