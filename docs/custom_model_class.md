# Add a dynamical class

A dynamical class specifies the state and physical equation before training. It
does not prescribe the application data source.

Implement `DynamicalClass` with five operations:

1. build the conditional-drift model;
2. build the dissipation factor;
3. build the thermodynamic model;
4. evaluate the thermodynamic force;
5. evaluate the complete drift, including the discretization-consistent thermal
   term required by that class.

Register the implementation and load it from YAML:

```yaml
extensions:
  - ../../examples/custom_model_class/model.py
model:
  class: example_scalar_model_a
```

The built-in trainers assume a scalar field and the scalar Model-A-type
factorization. A Model B, Model H, vector-field, or application-specific class may
reuse them only when their input and loss contracts are exact. Otherwise register
new trainers while retaining the generic procedure, run, and provenance layers.

Do not encode a new physical assumption by silently changing a generic trainer.
The registered class and its configuration must expose every conservation,
symmetry, locality, and normalization constraint.
