# CrBr3

`configs/CrBr3/pipeline.yaml` is the DeepMesoD organization of the archived
CrBr3 experiment. The experiment name is exactly `CrBr3`.

The coarse field is stored by the archive as `(T, My, Mx, Lz, 3)` with
`(251, 37, 40, 28, 3)`. DeepMesoD loads it as `(T, 3, 28, 37, 40)`. x/y are
periodic and z is a finite, nonperiodic film axis. The 3x3 Stage-2 factor is
applied independently to each z slab; Stage 1 and Stage 3 retain the symmetric
28x28 `J_z` coupling.

The manifest reads private processed trajectories through `DEEPMESOD_DATA_ROOT`
and writes new outputs below `DEEPMESOD_OUTPUT_ROOT/CrBr3`. It does not copy
large `.npy`, `.pth`, or generated figure artifacts into the package. Existing
weights can be loaded by constructing the model from the manifest and calling
`load_state_dict` on the corresponding private checkpoints.

The VAMPIRE input package is kept at
`configs/CrBr3/data-file/`. `input`, `CrBr3.mat`, and `CrBr3.ucf` define the
84-layer atomistic CrBr3 system and its mid-plane stacking fault. VAMPIRE
generates the raw spin trajectories; the processing scripts then coarse-grain
those trajectories into the 28-slab mesoscopic fields used by this pipeline.

The new absolute-position converter writes `times.npy` in ns by default. Keep
the configured frame spacing `data.dt=0.002`; for the archived materialized
training view, the same nanosecond convention applies.

See [`CrBr3_pipeline.md`](CrBr3_pipeline.md) for the complete VAMPIRE →
absolute-position coarse-graining → three-stage training → evaluation and
visualization workflow.
