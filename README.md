# DeepMesoD

<p align="center">
  <img src="assets/branding/deepmesod-logo.png" alt="DeepMesoD logo" width="720">
</p>

DeepMesoD learns an effective mesoscopic field equation from increments of a
coarse-grained trajectory. The repository separates the trajectory format, the
chosen dynamical class, trainable component parameterizations, training stages,
inference, evaluation, and visualization.

The first complete implementation is the scalar Model-A-type equation used for
the Ising--Kac experiment. The same interfaces are intended for vector fields and
other dynamical classes. A new application supplies trajectories and a registered
model-class adapter rather than editing the training loops.

## Repository map

```text
configs/                 Experiment and example configurations
src/deepmesod/data/      Trajectory schema, Vampire coarse-graining, sampling
src/deepmesod/classes/   Dynamical-class interface and registry (scalar/vector)
src/deepmesod/models/    Trainable drift, free-energy, kernels, and checkpoints
src/deepmesod/operators/ Finite-range scalar/vector B, K=BB*, exact divergence
src/deepmesod/training/  Scalar and vector Stage 1/2/direct Stage 3 trainers
src/deepmesod/inference/ Deterministic and stochastic scalar/vector rollouts
src/deepmesod/evaluation Metrics and physical-operator diagnostics
src/deepmesod/visualization/ Generic plotting primitives and shared style
experiments/paper/       Experiment-specific evaluation and paper plotting
examples/                Synthetic data and custom-model templates
docs/                    Data, model-class, training, and interpretation guides
```

Data and outputs are not Python modules. Training code reads paths from a YAML
configuration and writes a self-contained run directory containing the resolved
configuration, random seed, source revision when available, checkpoints, metrics,
and provenance.

The public repository contains no trajectories, cached inverse targets, model
checkpoints, rollouts, or generated results. Root-level `data/`, `outputs/`,
`results/`, `checkpoints/`, `artifacts/`, and `reports/` directories are ignored.
Supply data outside Git and point to it through environment variables or YAML.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
deepmesod --help
```

GPU training requires a PyTorch build compatible with the local CUDA runtime.
The package itself does not download data or checkpoints.

## Prepare a trajectory

Each trajectory is a directory with three files:

```text
trajectory/
  field.npy       shape (time, *space) or (time, channels, *space)
  times.npy       shape (time,), strictly increasing
  metadata.json   physical and coarse-graining metadata
```

The field stored at index `n` is the conditioning state. Its target is always the
single forward increment

```text
(field[n + 1] - field[n]) / (times[n + 1] - times[n]).
```

Transitions are formed within each trajectory before trajectories are combined.
No increment is ever formed across a trajectory boundary.

Validate the complete experiment before training:

```bash
deepmesod validate --config configs/ising/t075-cg.yaml
```

This checks trajectory structure, transition counts, physical metadata, test
ensemble compatibility, registered components, Stage 3 objective, exact local
divergence support, and output separation before a long job starts.

See `docs/data_contract.md` for the metadata schema and a converter template.
The built-in Ising generator is selected by `generator: ising_kac_tau_leaping`
in its YAML file. Another simulator registers a generator or writes the same
trajectory contract without importing any trainer.

For the VAMPIRE 2D ferromagnet, use `deepmesod data coarse-grain-vampire` after
the atomistic run. The block-mean conversion and the expected raw snapshot
layouts are documented in `docs/2D_ferromagnet_data.md`.

## Select a model class

The YAML field `model.class` selects a registered dynamical class. The built-in
`model_a_scalar` adapter defines the state, thermodynamic force, finite-range
factor, complete Itô drift, and compatible Stage 1--3 components for the current
Ising calculation.

```yaml
model:
  class: model_a_scalar
  thermodynamics:
    interaction_constraint: centrosymmetric_sum_one
```

`interaction_constraint` is a model choice, not a property of DeepMesoD. The
built-in scalar implementation supports `none`, `sum_one`, `centrosymmetric`,
and `centrosymmetric_sum_one`. The included Ising example selects the last option.

A new Model B, Model H, or application-specific class implements the interface in
`deepmesod.classes.base` and registers one builder. The generic data loader, run
management, evaluation, and visualization utilities remain unchanged. See
`docs/custom_model_class.md` and `examples/custom_model_class/`.

External model classes, trainers, losses, and generators can be loaded without
editing the package:

```yaml
extensions:
  - ../../examples/custom_model_class/model.py
```

## Train

Run the three stages independently so their inputs and outputs remain auditable:

```bash
deepmesod train drift --config configs/ising/t075-cg.yaml
deepmesod train dissipation --config configs/ising/t075-cg.yaml
deepmesod train thermodynamics --config configs/ising/t075-cg.yaml
```

The first command creates a fresh run directory. Later stage commands resume only
when the configuration hash matches the stored provenance. A nonempty stage
directory is never overwritten.

Stage 3 has two supported objectives selected by `training.stage3.objective`:

- `direct` minimizes the complete forward-rate residual.
- `cg` constructs a regularized thermodynamic-force target with a matrix-free
  full-field conjugate-gradient solve, then fits the free energy and interaction.

Both modes use the same fixed Stage 2 factor and the same complete finite-cell
Itô drift. They are alternatives, not different physical model classes.

Run the complete CPU smoke workflow before a full experiment:

```bash
python examples/synthetic/generate.py
deepmesod train --config configs/smoke.yaml
deepmesod infer --config configs/smoke.yaml
deepmesod evaluate --config configs/smoke.yaml
deepmesod report --config configs/smoke.yaml
```

For a fresh end-to-end run, one command performs preflight, all declared training
stages, inference, evaluation, and diagnostic reporting:

```bash
deepmesod reproduce --config configs/ising/t075-cg.yaml
```

It refuses a nonempty output directory. The ignored private run directory receives
JSON and Markdown reports, vector and raster diagnostics, checkpoints, targets,
rollouts, and prepared arrays. The report covers the complete conditional
response `[K[m]1]_x`, shifted `g(m)`, signed interaction weights, declared
constraint residuals, rollout error, and CG residuals when present. Numerical
integrity checks are kept separate from claims requiring an external reference.

## Infer and evaluate

```bash
deepmesod infer --config configs/ising/t075-cg.yaml
deepmesod evaluate --config configs/ising/t075-cg.yaml
```

The optimized rollout evaluates the factor CNN once per field step and contracts
the exact local Jacobian needed by the state divergence. It does not loop over
spatial cells in Python and does not use a stochastic trace approximation.

## Plot the Ising example

The Ising figure example uses a manifest under `experiments/paper/`. The manifest
records the data protocol, model seeds, uncertainty definition, plotted operator
functional, and plotting entry point. Supply the plot-data arrays described in
`docs/figure2_data.md` before running it.

```bash
deepmesod plot figure2 --manifest experiments/paper/figure2-ising.yaml
```

The plotter does not select a best seed or validation checkpoint. Lines and bands
use all model seeds declared by the supplied manifest.

Audit the source-only release tree with:

```bash
python tools/audit_release.py
```

## Agent use

An agent must read `AGENTS.md` before changing scientific code. The repository also
contains a loadable skill at `.agents/skills/use-deepmesod/`. The intended agent
workflow is:

1. run the full configuration preflight;
2. resolve and freeze a model class;
3. run the smallest smoke configuration;
4. use `deepmesod reproduce` with a fresh private output directory;
5. inspect interaction action, complete K response, CG residuals, and rollout;
6. read the generated report with the repository skill before making a physical
   claim.

## Current release boundary

The Ising finite-range implementation and example configurations are included.
The ASD namespace and vector-field interfaces are present for collaborator code.
The `2D-ferromagnet` vector implementation is documented in
`docs/2D-ferromagnet.md` and configured by `configs/2D-ferromagnet/pipeline-2D.yaml`.
The archived CrBr3 finite-thickness film is organized under the `model_a_film`
adapter, documented in `docs/CrBr3.md` and `docs/CrBr3_pipeline.md`, and
configured by `configs/CrBr3/pipeline.yaml` (run name: `CrBr3`).
Its Vampire simulation inputs are kept under
`configs/CrBr3/data-file/`; Vampire generates the atomistic spin
trajectories from these inputs, which are then coarse-grained for training.
The `2D-ferromagnet` Vampire inputs are kept under
`configs/2D-ferromagnet/data-files/` and are documented in
`docs/2D-ferromagnet.md`.
Collaborator data, unpublished checkpoints, and generated results are not bundled;
the config uses `DEEPMESOD_DATA_ROOT` for those private inputs.
Add the author-approved license and citation metadata before public release.
