# Agent instructions

Read the selected configuration and the relevant guide under `docs/` before
changing scientific code or launching an experiment.

## Scientific invariants

- Treat the configured dynamical class and equations as ground truth.
- Keep data preparation independent from model construction and training.
- Form only within-trajectory single forward increments unless a configuration
  explicitly defines another estimator.
- Keep the finite-range dissipation operator as `K = B B*`.
- Retain the configured finite-cell Itô state divergence in training and rollout.
- Do not replace the full operator by its diagonal or an averaged scalar.
- Do not add symmetry, positivity, radial, or locality constraints that are absent
  from the selected model class.
- Use the final scheduled checkpoint unless the protocol explicitly declares a
  validation split before training.
- Never choose a checkpoint, seed, or plotting range from the final rollout.

## Engineering invariants

- Add a new physical hypothesis through the dynamical-class registry. Do not fork
  generic training loops for each application.
- Keep trajectory I/O, model components, training, inference, evaluation, and
  visualization in their existing packages.
- Use batched tensor operations over fields and patches. Do not add Python loops
  over grid cells or samples to a training or rollout hot path.
- Preserve run directories and provenance. Do not overwrite existing outputs.
- Run `pytest` and the synthetic smoke workflow after scientific code changes.
- Use `deepmesod validate` before a full run and `deepmesod reproduce` only with a
  fresh private output directory.
- Treat the generated report as a numerical audit. Physical conclusions require
  the relevant microscopic, analytical, or baseline reference.
- Scan staged files for absolute server paths, secrets, datasets, checkpoints,
  generated figures, and caches before public release.

## Example workflows

Treat every manifest as an explicit protocol. Do not infer data, checkpoints, or
scientific settings from neighboring examples or remembered commands.
