"""Lift the in-plane vector mobility operator to a finite-thickness film."""

import torch


def _slabs(field):
    b, c, lz, h, w = field.shape
    return field.permute(0, 2, 1, 3, 4).reshape(b * lz, c, h, w)


def film_blocks_on_field(factor, field):
    """Return per-slab blocks with shape ``(B,Lz,S,C,C,H,W)``."""
    b, _, lz, _, _ = field.shape
    blocks = factor.blocks_on_field(_slabs(field))
    return blocks.reshape(b, lz, factor.n_factors, factor.n_components,
                         factor.n_components, field.shape[-2], field.shape[-1])


def film_apply_mobility(factor, field, force):
    blocks = film_blocks_on_field(factor, field)
    values = factor.apply_mobility(blocks.reshape(-1, *blocks.shape[2:]), _slabs(force))
    b, c, lz, h, w = force.shape
    return values.reshape(b, lz, c, h, w).permute(0, 2, 1, 3, 4)


def film_divergence_exact(factor, field, chunk_size=512):
    """Exact Itô divergence, independently in each z slab."""
    b, c, lz, h, w = field.shape
    value = factor.divergence_on_field_exact(_slabs(field), chunk_size=chunk_size)
    return value.reshape(b, lz, c, h, w).permute(0, 2, 1, 3, 4)

