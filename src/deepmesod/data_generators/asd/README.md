# ASD adapter boundary

Collaborator ASD generators should write `(time, 3, y, x)` magnetization fields
through `write_asd_trajectory`. Preserve the original time units, coarse-cell
definition, boundary condition, temperature, cell volume, field normalization,
and source-simulator version in `metadata.json`.

This directory deliberately contains no synthetic replacement for the unpublished
ASD generator. Add the final generator as a separate module and keep conversion
independent from the vector-field dynamical-class implementation.
