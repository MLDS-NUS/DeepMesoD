# Migrated `2D-ferromagnet` vector pipeline

The `2D-ferromagnet` implementation is exposed through the same package
boundaries as the scalar Ising implementation. The old scripts remain in
`MesoscopicModel` as historical provenance; new code should import DeepMesoD.

| Old responsibility | DeepMesoD location |
| --- | --- |
| `GlobalMagnetizationData`, `vector_adapter.py` | `deepmesod.data.vector` |
| `WeakPriorParam` and `create_kernel` | `deepmesod.models.vector`, `models.kernels` |
| `VectorFiniteRangeFactor` | `deepmesod.operators.vector_factor` |
| windowed vector Gaussian likelihood | `deepmesod.training.vector_likelihood` |
| direct target preparation | `deepmesod.training.vector_targets` |
| Stage 1/2/3 trainers | `deepmesod.training.stage{1,2,3}_vector` |
| direct vector rollout | `deepmesod.inference.vector_rollout` |
| Figure 3-A direct rollout export | `experiments/paper/figure3_2D_ferromagnet.py` |
| Figure 3 composite renderer | `experiments/paper/figure3_2D_ferromagnet_plot.py` |

The model class is registered as `model_a_vector`. It keeps the pipeline's
three-component `(batch, 3, height, width)` internal layout, circular boundary
conditions, dense 3x3 factor blocks, `K = B B*`, spin-flip-even factor, and
functional-derivative convention for the direct Stage 3 model. This experiment
uses the raw interaction convention from `pipeline-2D.yaml`.

Existing pipeline checkpoints can be loaded directly:

```python
from deepmesod.models import load_vector_drift, load_vector_thermodynamics
from deepmesod.operators import load_vector_factor

drift, _ = load_vector_drift(".../stage1/best_model.pth")
factor, _ = load_vector_factor(".../stage2/factor_last.pth")
thermodynamics, _ = load_vector_thermodynamics(".../stage3/checkpoints/step020000.pth")
```

The source-only repository does not copy `.pth`, trajectory, cache, or figure
artifacts. Use `configs/2D-ferromagnet/pipeline-2D.yaml` with private environment
paths to reproduce a run.

The Vampire input files for this experiment are kept at
`configs/2D-ferromagnet/data-files/`. `input` and `FM2D.mat` define the
single-material atomistic ferromagnet, including the simulation temperature,
exchange, anisotropy, random initialization, and periodic x/y boundaries.
Vampire generates the atomistic spin trajectories; the existing preprocessing
then produces the coarse-grained 2D vector fields consumed by DeepMesoD.

## Data → coarse-graining → training → evaluation → visualization

The experiment-specific paths stay under the same `2D-ferromagnet` namespace:

1. Vampire source input is `configs/2D-ferromagnet/data-files/`. Vampire
   generates the atomistic spin trajectories.
2. The processed coarse-grained vector trajectories are supplied by the
   `data.train` and `data.test` paths in the config. Generate them with the
   converter documented in [`docs/2D_ferromagnet_data.md`](2D_ferromagnet_data.md);
   their public reader is `deepmesod.data.vector`.
3. Stage 1, dissipation and direct Stage 3 write checkpoints below
   `run.output_dir`.
4. Figure 3-A evaluation writes
   `run.output_dir/evaluation/figure3/data/rollout_deepmesoD/`.
5. Figure 3 visualization reads that exact evaluation root. The historical
   export root is only a reference source for the unchanged baseline, CNN and
   B-panel tables.

Run the evaluation and then the renderer with the same config:

```bash
PYTHONPATH=src:. python -m experiments.paper.figure3_2D_ferromagnet \
  --config configs/2D-ferromagnet/pipeline-2D.yaml

PYTHONPATH=src:. python -m experiments.paper.figure3_2D_ferromagnet_plot \
  --config configs/2D-ferromagnet/pipeline-2D.yaml
```

The evaluator uses only the formal `stage3` direct checkpoint.

## Figure 3

The v15 composite Figure 3 drawing process is implemented in
`experiments/paper/figure3_2D_ferromagnet_plot.py`. With `--config`, it
reads the generated evaluation root and the reference data root from the same
experiment config, then writes PDF, JPG, PNG, and provenance JSON:

```bash
PYTHONPATH=src:. python -m experiments.paper.figure3_2D_ferromagnet_plot \
  --config configs/2D-ferromagnet/pipeline-2D.yaml
```

The plot contains the mean-`m_z` rollout panel, six mobility/free-energy panels,
and the 2×4 magnetisation snapshot grid from `composite_figure_v15.pdf`. When
the evaluator output is separate from the historical export, pass
`--reference-data-root` as shown above.
