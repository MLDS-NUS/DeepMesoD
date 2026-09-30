# Data contract

DeepMesoD consumes trajectories, not simulator-specific objects. Keep microscopic
generation, coarse-graining, conversion, and learning as separate commands.

## Files

`field.npy` stores either `(time, *space)` for a scalar field or
`(time, channels, *space)` for a multicomponent field. `times.npy` stores strictly
increasing physical times. `metadata.json` must contain:

```json
{
  "schema_version": 1,
  "system": "user_defined_name",
  "boundary": "periodic",
  "coarse_volume": 1024,
  "beta": 1.3333333333333333
}
```

`temperature` may replace `beta`. Record channel order, spatial units, time units,
coarse-cell geometry, simulator name and version, and generation seed when they
apply.

## Transition rule

For each trajectory independently, state `n` conditions the single forward
increment to state `n+1`. Truncate each trajectory before constructing increments.
Concatenating states first would create an invalid increment from the end of one
trajectory to the beginning of another.

## New simulators

Write a converter that produces these three files. Do not import training modules
inside a generator. A callable registered with `register_data_generator` can be
selected by the `generator` field in a generation YAML. Run `deepmesod data
validate` before starting a procedure.

The authors' older Ising data use `m.npy`, `times.npy`, and `meta.json`. Convert
one trajectory without copying its arrays using:

```bash
deepmesod data convert-legacy-ising --source SOURCE --output OUTPUT
```

The command writes contract-compliant metadata and links the arrays. It derives
the coarse-cell volume as the square of the recorded block size and records the
temperature and external field from the legacy dataset metadata.
