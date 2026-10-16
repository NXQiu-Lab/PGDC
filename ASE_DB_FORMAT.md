# 本项目 ASE 数据库文件格式说明

> 本文档描述 `material-e` 项目中 `.db` 文件（ASE SQLite Database）的实际存储格式，以及 MACE 读取这些文件时需要注意的适配要点。
>
> 最后更新：2026-07-14

---

## 1. 文件格式总览

本项目所有 `.db` 文件均为 **ASE SQLite Database** 格式（`ase.db`），可通过以下方式识别：

```python
import ase.db
db = ase.db.connect('data/sn2_reactions.db')
print(db.count())  # -> 452709
```

关键特征：
- 后端为 SQLite，文件头包含 `SQLite format 3` 标识
- 每一行对应一个 `ase.Atoms` 构型
- **能量、力、应力等物理量存储在 `row.data` 字段中**，而非标准 ASE 的 `row.energy`、`atoms.calc.results` 或 `atoms.arrays`

---

## 2. 数据字段存储位置对照表

| 物理量 | 标准 ASE 位置 | **本项目实际位置** | 备注 |
|--------|--------------|-------------------|------|
| 原子种类 | `atoms.numbers` | `atoms.numbers` / `row.numbers` | 标准位置，未改变 |
| 原子坐标 | `atoms.positions` | `atoms.positions` / `row.positions` | 标准位置，未改变 |
| 晶胞 | `atoms.cell` | `atoms.cell` / `row.cell` | 标准位置，未改变 |
| 能量 | `row.energy` / `atoms.calc.results['energy']` | **`row.data['energy']`** | ⚠️ 非标准位置 |
| 力 | `atoms.arrays['forces']` / `atoms.calc.results['forces']` | **`row.data['forces']`** | ⚠️ 非标准位置 |
| 应力 | `atoms.calc.results['stress']` | **`row.data['stress']`** | ⚠️ 部分数据集包含；schnetpack 训练时须为 **(1, 3, 3)**，见第 6 节 |
| 其他属性 | `atoms.info` / `row.key_value_pairs` | **`row.data['xxx']`** | 如 `HOMO_LUMO_gap`、`dipole` 等 |

### 2.1 为什么需要关注这一点

标准 ASE 的 `row.toatoms()` **不会自动将 `row.data` 中的 `energy` 和 `forces` 附加到返回的 `Atoms` 对象上**。这意味着：

```python
row = list(db.select(limit=1))[0]
atoms = row.toatoms()

# 以下调用会失败或返回 None
atoms.get_potential_energy()   # RuntimeError: Atoms object has no calculator
atoms.arrays.get('forces')     # None
```

必须显式地从 `row.data` 中提取：

```python
energy = row.data['energy']      # float, eV
forces = row.data['forces']      # ndarray, shape (natoms, 3), eV/Å
```

---

## 3. 典型数据集格式示例

### 3.1 `sn2_reactions.db`

```python
{
    'natoms': 2 or 6,           # 反应物/过渡态原子数
    'data': {
        'energy': float,        # 总能量，单位 eV
        'forces': ndarray,      # shape (natoms, 3)，单位 eV/Å
    }
}
```

- 总行数：452,709
- 注意：存在约 58,000 个重复实例（详见第 5 节）

### 3.2 `MD17/*.db`

```python
{
    'natoms': 12 ~ 21,          # 取决于分子
    'data': {
        'energy': float,
        'forces': ndarray,      # shape (natoms, 3)
    }
}
```

- 能量和力同样仅在 `row.data` 中

### 3.3 `qm7x/*.db`

```python
{
    'natoms': ~20,
    'data': {
        'energy': float,
        'forces': ndarray,
        'sRMSD': float,
        'atomization_energy': float,
        'HOMO_LUMO_gap': float,
        'dipole': ndarray,
        'polarizability': ndarray,
    }
}
```

- 包含更多物理属性，全部存于 `row.data`

### 3.4 `HfO2.db`（少数例外）

```python
{
    'natoms': 96,
    'data': {
        'energy': float,
        'forces': ndarray,
        'stress': ndarray,
        'free_energy': float,
    }
}
```

- 调用 `row.toatoms()` 后，ASE 会自动附加 `SinglePointCalculator`
- `atoms.calc.results` 包含 `energy`、`forces`、`stress`
- **这是 ASE 的标准行为，但仅在 `row.data` 中包含这些键时才会触发**（见 `ase.db.row.AtomsRow.toatoms()` 实现）

---

## 4. 与 MACE 标准读取路径的差异

MACE 的标准训练流程（`mace/cli/run_train.py`）对于 ASE 可读文件会调用：

```python
# mace/data/utils.py -> load_from_xyz()
atoms_list = ase.io.read(file_path, index=":")
```

然后提取能量/力：

```python
# mace/data/utils.py -> config_from_atoms()
for name, atoms_key in key_specification.info_keys.items():
    properties[name] = atoms.info.get(atoms_key, None)

for name, atoms_key in key_specification.arrays_keys.items():
    properties[name] = atoms.arrays.get(atoms_key, None)
```

### 4.1 问题

由于本项目的 `.db` 文件将 `energy`/`forces` 存放在 `row.data` 而非 `atoms.info` / `atoms.arrays`，**MACE 默认路径无法直接读取这些属性**。

### 4.2 建议的适配策略

若要在 MACE 中支持本项目 `.db` 文件，推荐以下任一方式：

#### 方案 A：在 `load_from_xyz` 中增加 `row.data` 回退（推荐）

在 `mace/data/utils.py` 的 `config_from_atoms()` 中，增加对 `row.data`（或 `atoms.info['data']`）的回退读取：

```python
def config_from_atoms(atoms, ...):
    ...
    for name, atoms_key in key_specification.arrays_keys.items():
        val = atoms.arrays.get(atoms_key, None)
        # 回退：从 data 字典读取
        if val is None and "data" in atoms.info:
            val = atoms.info["data"].get(atoms_key, None)
        properties[name] = val
        ...
```

> 注意：若通过 `ase.io.read(..., format='db')` 读取，ASE 可能将 `row.data` 以某种方式暴露；需根据实际读取方式调整。

#### 方案 B：自定义 Dataset 类（如 `AseDBDataset`）

本项目已引入 `mace/tools/fairchem_dataset/lmdb_dataset_tools.py` 中的 `AseDBDataset`，它通过 `ase.db.connect()` 直接连接数据库，并在 `get_atoms()` 中将 `row.data` 正确映射到 `atoms.info` 和 `atoms.arrays`：

```python
# AseDBDataset.get_atoms() 中的关键逻辑
data_dict = _decode_ndarrays(row.data) if isinstance(row.data, dict) else {}
atoms.info.update(data_dict)

extra_arrays = data_dict.pop("__arrays__", {})
for name, arr in extra_arrays.items():
    atoms.new_array(name, np.asarray(arr))
```

**若使用 `AseDBDataset`，需确保 `.db` 文件在 MACE 中被识别为 LMDB/非 ASE 可读路径**，否则会优先走 `load_from_xyz` 路径而绕过 `AseDBDataset`。

#### 方案 C：预处理为 `.xyz` 或 `.h5`

在训练前将 `.db` 导出为 MACE 原生支持的格式：

```python
import ase.io
atoms = ase.io.read('data/sn2_reactions.db', index=':', format='db')
# 确保能量/力已正确附加到 atoms.info / atoms.arrays
ase.io.write('data/sn2_reactions.xyz', atoms, format='extxyz')
```

---

## 5. 数据质量注意事项

### 5.1 重复构型

以 `sn2_reactions.db` 为例：

| 指标 | 数值 |
|------|------|
| 总构型数 | 452,709 |
| 唯一构型数 | 394,684 |
| 重复实例数 | ~58,000 |
| 最大重复次数 | 15 次 |

**影响**：MACE 的 `random_train_valid_split` 仅对索引做随机分割，不识别物理重复。若同一构型出现多次，其副本可能分别落入训练集和验证集，造成**数据泄漏**。建议在训练前进行去重，或在读取时保留唯一构型。

### 5.2 力的分布范围（抽样检查）

| 统计量 | 数值 |
|--------|------|
| 最小值 | 0.0 eV/Å |
| 最大值 | ~17.5 eV/Å |
| 平均值 | ~0.89 eV/Å |
| 中位数 | ~0.39 eV/Å |
| 95% 分位 | ~3.40 eV/Å |

力分布正常，不存在整体趋近于零的异常情况。

### 5.3 单位统一

所有 `.db` 文件中的能量和力均使用 **eV** 和 **eV/Å** 作为单位，与 MACE 内部默认单位一致，无需额外缩放。

---

## 6. 应力数据格式（schnetpack / PaiNN 训练）

当需要训练含应力输出的机器学习势（如本项目 `painn.py` 的 `--stress-weight` 模式）时，`.db` 文件必须满足以下额外要求：

1. **存储位置**：应力张量必须存放在 `row.data['stress']` 中。
2. **形状**：schnetpack 要求 `row.data['stress']` 的形状为 **`(1, 3, 3)`**（携带 leading batch 维的完整张量），而不是 ASE 常用的 6 分量 Voigt 形式。
   - ASE Voigt 顺序为 `[xx, yy, zz, yz, xz, xy]`，需先转换为 3×3 张量：
     ```python
     [[xx, xy, xz],
      [xy, yy, yz],
      [xz, yz, zz]]
     ```
   - 再在最前面增加一个 batch 维度，得到形状 `(1, 3, 3)`，以便 `schnetpack.data._atoms_collate_fn` 按第 0 轴拼接后得到 `(batch, 3, 3)`。
3. **单位**：`eV/Å³`（即 `eV / Angstrom / Angstrom / Angstrom`）。
4. **元数据**：数据库 `metadata["_property_unit_dict"]` 中必须包含：
   ```python
   "stress": "eV/Angstrom/Angstrom/Angstrom"
   ```
   这一格式可被 schnetpack 的 `units._parse_unit` 正确解析。

### 6.1 使用 `convert_db_for_schnetpack.py`

如果原始数据库的应力以 6 分量 Voigt 形式存放在 `row.data['stress']`，`scripts/paper/convert_db_for_schnetpack.py` 会自动将其转换为 schnetpack 所需的 `(1, 3, 3)` 张量，并在元数据中写入正确的单位。转换后的张量可直接被 `schnetpack.data.ASEAtomsData` 读取。

对于没有应力的数据库，可以使用 `scripts/amorphous_carbon/compute_stress_for_db.py` 通过远程 CP2K `ENERGY_FORCE` 单点批量补算应力（支持断点续算、不修改原始计算文件），补算完成后再用 `convert_db_for_schnetpack.py` 确认格式即可。

## 7. 快速验证脚本

在将 `.db` 文件接入 MACE 或 schnetpack 训练前，建议运行以下脚本验证数据读取是否正确：

```python
import ase.db
import numpy as np

def inspect_db(db_path, sample_limit=1000):
    db = ase.db.connect(db_path)
    print(f"Database: {db_path}")
    print(f"Total rows: {db.count()}")

    energies = []
    forces_flat = []
    has_forces = 0
    has_energy = 0
    has_stress = 0

    for row in db.select(limit=sample_limit):
        e = row.data.get('energy')
        f = row.data.get('forces')
        s = row.data.get('stress')
        if e is not None:
            has_energy += 1
            energies.append(e)
        if f is not None:
            has_forces += 1
            forces_flat.extend(f.flatten().tolist())
        if s is not None:
            has_stress += 1

    print(f"Sampled: {sample_limit}")
    print(f"  Has energy: {has_energy}/{sample_limit}")
    print(f"  Has forces: {has_forces}/{sample_limit}")
    print(f"  Has stress: {has_stress}/{sample_limit}")

    if energies:
        print(f"  Energy range: {min(energies):.4f} ~ {max(energies):.4f} eV")
    if forces_flat:
        forces_flat = np.array(forces_flat)
        print(f"  Force magnitude: min={np.min(np.abs(forces_flat)):.6f}, "
              f"max={np.max(np.abs(forces_flat)):.6f}, "
              f"mean={np.mean(np.abs(forces_flat)):.6f} eV/Å")

# 示例
inspect_db('data/sn2_reactions.db')
```

---

## 8. 修订记录

| 日期 | 修订内容 |
|------|----------|
| 2026-06-02 | 初始版本：基于 `sn2_reactions.db`、`MD17`、`qm7x`、`HfO2` 等数据集的格式分析 |
| 2026-07-14 | 增加应力张量存储规范、schnetpack 训练要求及 `convert_db_for_schnetpack.py` 转换说明 |
