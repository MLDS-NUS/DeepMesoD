# Interpreting learned operators

## Interaction kernel

The interaction constraints belong to the selected model class. The built-in
scalar implementation supports no constraint, unit sum, centrosymmetry, or both.
The formal Ising configuration uses a signed `15 x 15` kernel `J` with both
centrosymmetry and unit sum. Its center coefficient is not interpreted
separately. Compare the identifiable action `Jm - m`, and omit the center from
shape plots when the center/off-center decomposition is not identifiable. If
centrosymmetry is not imposed, the direct force
`g'(m) + m - Jm - h_ext` is not the strict gradient of the corresponding
quadratic free energy unless the antisymmetric part has no effect.

## Dissipation operator

Stage 2 learns a local factor `B[m]`; the dissipation operator is `K[m] =
B[m]B[m]*`. Positive semidefiniteness follows from this representation and is not
evidence of statistical calibration.

For a scalar hydrodynamic comparison, plot the complete constant-force response

```text
[K[m] 1]_x = sum_y K_xy[m]
```

conditioned on the coarse-grained field. Do not substitute `K_xx`, an ordinary
mean of factor values, or the center factor coefficient. Also report conditional
covariance calibration when judging Stage 2.

## Free-energy density

The displayed homogeneous density `g(m)` is defined only up to an additive
constant. Subtracting each curve's own minimum is a plotting convention and must
be stated in the caption. The thermodynamic force, not the vertical offset of
`g`, enters the field equation.

## Itô state divergence

When `K` depends on the field, the configured finite-cell Itô drift contains
`(beta Vc)^-1 div_m K`. Omitting this term changes the inferred thermodynamic
force. Treat omission as a labeled ablation, not an alternative rendering of the
same model.
