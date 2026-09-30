# Training procedures

`training.procedure` is an ordered list. Each entry names a trainer, a loss, its
artifact dependencies, and stage-local options.

```yaml
training:
  procedure:
    - name: drift
      trainer: local_drift
      loss: forward_rate_mse
      config: {epochs: 500, batches_per_epoch: 500}
    - name: dissipation
      trainer: finite_range_factor
      loss: window_gaussian_nll
      requires: [drift]
      config: {epochs: 500, batches_per_epoch: 100}
    - name: thermodynamics
      trainer: free_energy_interaction
      loss: cg_force_target
      requires: [dissipation]
      config: {objective: cg, steps: 400000}
```

The names are labels, not hard-coded stage numbers. A different dynamical class
may use fewer stages, add a stage, or register another trainer and loss. The
dependency list is checked before a stage starts.

## Built-in scalar Model-A-type procedure

- `local_drift` estimates the conditional mean of the single forward increment.
- `finite_range_factor` estimates the conditional increment covariance through
  the factor `B`, with `K = B B*`.
- `free_energy_interaction` learns the homogeneous free-energy density and the
  unit-sum interaction kernel while keeping the Stage 2 factor fixed.

The scalar implementation uses state-balanced sampling when requested. This
changes the optimization measure, not the physical trajectory or target.

## Stage 3 choice

`direct_forward_rate` minimizes the complete drift residual

```text
K[m] G[m] - ((beta Vc)^-1 div_m K[m] - observed_forward_rate).
```

`cg_force_target` first solves the regularized full-field system

```text
(K[m] + lambda I) G_target = (beta Vc)^-1 div_m K[m] - observed_forward_rate
```

with a matrix-free batched conjugate-gradient solve, then fits the thermodynamic
force. Both use the same single forward increments, factor, and exact state
divergence. The mode must be declared before training and recorded in figure
provenance.

## Custom stages

Register a function with `register_trainer(name)` and a callable with
`register_loss(name)`. A trainer receives `(RunContext, StageSpec)`, writes only
inside `context.stage_dir(stage.name)`, and returns a JSON-serializable artifact
mapping. Load the extension through the top-level `extensions` list in YAML.
