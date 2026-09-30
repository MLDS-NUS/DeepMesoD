# Extension workflow

A registered dynamical class fixes the state, component builders,
thermodynamic-force convention, and complete drift. Load external registrations
through the YAML `extensions` list.

Reuse a built-in trainer only when its tensor and physical loss contract matches
the new class exactly. Otherwise register a trainer and loss, then place them in
the ordered `training.procedure`. Stage names are labels; custom procedures may
add, remove, or reorder stages subject to declared artifact dependencies.

Use batched tensor operations over fields and patches. Python loops over a small
number of stages, stencil offsets, or CG iterations are acceptable. Do not add
Python loops over grid cells or samples to hot paths.
