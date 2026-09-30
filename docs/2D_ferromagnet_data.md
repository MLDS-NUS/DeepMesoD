# `2D-ferromagnet` data generation

The data path has two separate operations:

```text
VAMPIRE atomistic simulation
        ↓
raw spin snapshots
        ↓ periodic B×B block mean
coarse-grained 2D vector trajectory
        ↓
DeepMesoD training
```

## 1. Generate raw data with VAMPIRE

The experiment input files are in
[`configs/2D-ferromagnet/data-files/`](../configs/2D-ferromagnet/data-files/):

- `input` contains the time-series simulation settings;
- `FM2D.mat` contains the ferromagnet material parameters.

VAMPIRE itself is an external executable. Run it using this input directory and
keep each seed's output in a separate directory. The exact executable command
depends on the local VAMPIRE build and cluster launcher; DeepMesoD does not
assume a VAMPIRE installation.

The converter accepts either of the snapshot layouts used by the original
MesoscopicModel workflow:

```text
<raw>/*_snaps.npy
<raw>/*_times.npy
<raw>/*_meta.json
```

or:

```text
<raw>/*_snaps.bin
<raw>/*_times.npy
<raw>/*_meta.json
```

The bit-packed format is for scalar ±1 snapshots. The floating-point format
also supports vector snapshots shaped `(T,L,L,3)` or `(T,L,L,1,3)`.

## 2. Coarse-grain the VAMPIRE snapshots

For the published 2D-ferromagnet setup, use a fine grid of `1024×1024`, a
coarse grid of `64×64`, and a block width `B=16`:

```bash
PYTHONPATH=src python -m deepmesod.cli data coarse-grain-vampire \
  --input-root /path/to/vampire/2D-4-large-damping/seed1 \
  --output-dir /path/to/coarse-grained/2D-ferromagnet/seed1/M64_B16 \
  --block-size 16 \
  --coarse-size 64 \
  --batch-size 10
```

The same command is available through the installed `deepmesod` executable:

```bash
deepmesod data coarse-grain-vampire \
  --input-root /path/to/vampire/seed1 \
  --output-dir /path/to/coarse-grained/2D-ferromagnet/seed1/M64_B16
```

The output follows the DeepMesoD data contract:

```text
round0/
  field.npy       (T, 3, 64, 64) for the vector ferromagnet
  times.npy
  metadata.json
```

`metadata.json` records the VAMPIRE source, periodic boundary condition, block
size, coarse-cell volume (`16²=256` for this experiment), temperature/beta when
present in the source metadata, and component order. Run the normal data
validation command before training after adding these directories to
`data.train` and `data.test` in the experiment config.

## Provenance

The implementation is a package-level migration of
`MesoscopicModel/scripts/train/spins_to_m.py` and
`MesoscopicModel/src/data.py::spins_to_block_means`. It keeps the original
periodic block-centre convention but writes the canonical DeepMesoD field
layout, so training code does not depend on MesoscopicModel imports.
