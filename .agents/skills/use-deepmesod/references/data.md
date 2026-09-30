# Data workflow

Each trajectory directory contains `field.npy`, `times.npy`, and
`metadata.json`. The field has shape `(time, *space)` or `(time, channels,
*space)`. Times are strictly increasing. Required metadata includes periodic
boundary conditions, coarse-cell volume, and inverse temperature or temperature.

Generate Ising data with `deepmesod data generate --config CONFIG`. For another
simulator, register a generator or write a converter that emits the same files.
Never put simulator logic inside a trainer.

For the authors' legacy Ising directories containing `m.npy`, `times.npy`, and
`meta.json`, use `deepmesod data convert-legacy-ising --source SOURCE --output
OUTPUT`. The default creates links and normalized metadata without copying the
trajectory. Use `--mode copy` only for a self-contained private data directory.

Validate every split before training. Confirm the number of transitions selected
per trajectory and that no transition crosses a trajectory boundary.
