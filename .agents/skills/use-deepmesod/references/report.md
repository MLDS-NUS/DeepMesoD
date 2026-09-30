# Run and report audit

Use `deepmesod validate --config CONFIG` before any long run. A failed preflight
must be resolved before training. Use `deepmesod reproduce --config CONFIG` only
with a fresh output directory. The command intentionally refuses to overwrite an
existing run.

Read `resolved_config.json` before interpreting results. Confirm the trajectory
count and transition window, model class, interaction constraint, state sampling,
seed, Stage 3 objective, and final-checkpoint policy. Then read `provenance.json`
and verify that reported artifact paths belong to the same run.

The generic report is a numerical audit of one model seed. Check that histories,
rollout, and operator arrays are finite. Inspect rollout MAE over time, not only
its final value. For CG, compare the recorded relative residual with the declared
tolerance. For `J`, interpret unit sum or centrosymmetry only when the resolved
constraint declares it. For `K`, inspect the complete response `[K[m]1]_x` and
state-bin coverage. The displayed ribbon is within-bin variation over field sites,
not training-seed uncertainty. The free-energy curve is shifted by its own
minimum.

Do not conclude that a dynamical class is physically correct from finite losses,
a stable rollout, `K=B B*`, or small constraint residuals. Calibration and paper
claims require the experiment-specific microscopic, analytical, or baseline
reference and all declared model seeds.
