# Training workflow

The built-in scalar procedure estimates conditional drift, then a finite-range
factor, then thermodynamics. Each stage writes an independent final checkpoint.

Use `direct_forward_rate` to optimize the complete forward-rate equation. Use
`cg_force_target` to construct a regularized full-field force target with batched
matrix-free CG before fitting thermodynamics. Both modes use the same forward
increments, fixed Stage 2 factor, and exact state divergence. Declare the choice
before training and never select between them from the final plot.

Run one stage with `deepmesod train STAGE --config CONFIG`, or the complete
ordered procedure with `deepmesod train --config CONFIG`.
