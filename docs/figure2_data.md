# Figure 2 plot-data contract

`experiments/paper/figure2-ising.yaml` points to one NPZ file per temperature.
Each file contains:

```text
times
mae_deepmesod                (model_seed, time)
mae_hydrodynamic             (time,)
mae_allen_cahn               (time,)
mae_cnn                      (model_seed, time)
mean_reference               (microscopic_round, time)
mean_deepmesod               (model_seed, time)
mean_hydrodynamic            (time,)
mean_allen_cahn              (time,)
energy_state                 (state,)
energy_reference             (state,)
energy_deepmesod             (model_seed, state)
dissipation_state            (state_bin,)
dissipation_reference        (state_bin,)
dissipation_deepmesod        (model_seed, state_bin)
kernel_radius                (radial_shell,)
kernel_reference             (radial_shell,)
kernel_deepmesod             (model_seed, radial_shell)
```

Set `DEEPMESOD_PAPER_DATA` to the directory containing the exported NPZ files.
The portable plotter never reads training directories or selects checkpoints.
