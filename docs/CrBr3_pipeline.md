# CrBr3 全流程

CrBr3 使用独立于 `2D-ferromagnet` 的数据路径。原因是 CrBr3 的原子位于蜂窝晶格，不能把原子自旋直接 reshape 成方形 block；coarse-graining 必须读取 VAMPIRE 的绝对 `x/y/z` 坐标，识别晶格周期后再做映射。

这部分迁移自原仓库的
`scripts/honeycomb_stacking_fault/data-generation/honeycomb_spins_to_m.py`
（使用其中的 `lattice` 模式）；Jz/Jxy 和两块 free-energy 的参考实现分别是
`exports/fig4_Jz_Jxy_asd/` 与
`scripts/honeycomb_stacking_fault/stage3/two_block_free_energy_2d.py`。

代码边界如下，CLI 只负责组装配置，不承载具体科学逻辑：

```text
data/
  crbr3_preprocessing.py       VAMPIRE I/O + absolute-coordinate geometry
  coarse_graining.py           Ising/2D/CrBr3 coarse-graining entry points
evaluation/
  kernels.py                   reusable factorized-kernel extraction
  free_energy.py              reusable two-block energy scan
visualization/
  ...                           generic plotting primitives
experiments/paper/
  crbr3.py                     CrBr3 evaluation orchestration
  figure4_crbr3.py             CrBr3 Jz/Jxy/free-energy rendering
```

数据预处理和 coarse-graining 因为属于同一条数据入口，没有继续拆成多个
CrBr3 文件；evaluation 和 visualization 则按最终产物分开。

## 1. 用 VAMPIRE 生成原子数据

输入模板在 [`configs/CrBr3/data-file`](../configs/CrBr3/data-file/)：

- `input`：VAMPIRE 的 coarsening-dynamics 输入模板；
- `CrBr3.mat`：CrBr3 材料参数；
- `CrBr3.ucf`：84 层、含中间 stacking fault 的蜂窝 unit cell。

VAMPIRE 运行后，每个 seed/run 目录需要至少包含：

```text
atoms-coords.data
spins-00000000.data
spins-00000000.meta
spins-00000001.data
spins-00000001.meta
...
```

所以 `data` 的来源是 VAMPIRE；DeepMesoD 只负责把这些 atomistic snapshots 转成训练用的 mesoscopic trajectories。

## 2. 绝对位置 coarse-graining

```bash
python -m deepmesod.cli data coarse-grain-crbr3 \
  --input-root /path/to/vampire/seed1 \
  --output-dir /path/to/CrBr3/data/train_ns/seed1/latt_b16x30_s16x30_z3 \
  --stride-A 50 50 --block-A 50 50 --z-group 3 --time-unit ns
```

`--stride-A 50 50` 会自动 snap 到能整齐铺满周期盒子的晶格 repeat；对于当前 200 nm 盒子，实际网格是约 `37 × 40 × 28`，每个 coarse cell 含 480 个原子。`--z-group 3` 从中间 stacking fault 两侧向外分组，不会让一个 slab 穿过 fault。

输出是：

```text
m.npy          (T, My, Mx, Lz, 3)
counts.npy     (My, Mx, Lz)
times.npy      ns（默认；训练配置 dt=0.002）
z_layers.npy
meta.json      包含 VAMPIRE 来源、绝对位置映射和 lattice geometry
```

## 3. 三阶段训练

训练配置为 [`configs/CrBr3/pipeline.yaml`](../configs/CrBr3/pipeline.yaml)，顺序固定为：

1. `film_weak_prior`：学习 coarse drift；
2. `film_finite_range_factor`：学习有限范围 mobility；
3. `film_free_energy_interaction`：冻结 mobility，学习 `Psi/J` thermodynamics。

```bash
python -m deepmesod.cli data validate --config configs/CrBr3/pipeline.yaml
python -m deepmesod.cli train --config configs/CrBr3/pipeline.yaml
```

如果只重跑某一阶段：

```bash
python -m deepmesod.cli train thermodynamics --config configs/CrBr3/pipeline.yaml
```

这里没有 inverse model；CrBr3 流程使用 direct `Psi/J` Stage-3 模型。

## 4. Jz/Jxy 与 free-energy evaluation

Stage-3 训练完成后，以下实验脚本从同一个 `thermodynamics_last.pt` 生成 kernel 表和两块自由能数据：

```bash
PYTHONPATH=src:. python -m experiments.paper.crbr3 \
  --config configs/CrBr3/pipeline.yaml
```

结果位于：

```text
${DEEPMESOD_OUTPUT_ROOT}/CrBr3/evaluation/kernels/
  Jz.csv  w_xy.csv  kernel_metadata.json
${DEEPMESOD_OUTPUT_ROOT}/CrBr3/evaluation/free_energy/
  two_block_free_energy_2d.npz
  two_block_free_energy_2d.csv
  minimum_path.csv
  free_energy_metadata.json
```

二维 free-energy 坐标是

```text
mean_mz = (m1 + m2) / 2
ell     = (m1 - m2) / 2
```

其中前 14 个 z slabs 取 `m1`，后 14 个取 `m2`；整个 28 层 uniform in-plane field 会直接经过 DeepMesoD 的 `FilmPsiJModel`。

绘图：

```bash
PYTHONPATH=src:. python -m experiments.paper.figure4_crbr3 \
  --config configs/CrBr3/pipeline.yaml
```

这会生成与原始
`exports/fig4_Jz_Jxy_asd/Jz_Jxy_styled.pdf` 和
`exports/crbr3_two_block_free_energy_2d/two_block_free_energy_2d.pdf`
对应的 PDF/PNG，但输出位置改为 CrBr3 自己的 visualization 目录。
