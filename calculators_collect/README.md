# 新增／修改公式汇总：Metric、BCP、IMD 与 IMQ

本目录只收录本轮核对后的自定义实现与数据需求检查，不复制完整 WannierBerri。静态入口统一叫 **Metric**，不保留历史 QMetric 别名。原始 v1.7、v1.8 归档不修改。

这是可导入的扩展包，依赖已安装的 WannierBerri；**不能用这里的文件整文件覆盖上游同名文件**。以后合并进上游时，应按类／函数迁移并调整 import。

## 本轮收录范围

| 文件 | 新增／修改定义 | 数学含义和手册依据 |
|---|---|---|
| `formula/covariant.py` | `Metric` | 子空间矩阵 `G_mn^(ab)`；`eq:Finterpolation-components`、`eq:projected-geometry-components` |
| 同上 | `DerMetric` | `G_mn^(ab:d)`；`eq:Ffirst-components`、`eq:geometry-derivative-components` |
| 同上 | `VelMetric`、`MetricVel`、`VelDerMetric` | 能量梯度与上述矩阵的有序乘积；`eq:energy-ingredients`、`eq:product-components` |
| `calculators/static.py` | `Metric` | 零温 `∫[dk] Tr_occ G^(ab)`；直接使用上游非可加积分 |
| 同上 | `MetricDipole_FermiSea` | 零温 `∫[dk] Tr_occ G^(ab:d)`；核求导不包含占据阶跃的导数 |
| 同上 | `MetricQuadrupole_FermiSurf` | `VelDerMetric` 的累积子空间 trace 差分配合上游 `fder=1`；原始轴 `(c,a,b,d)` |
| `calculators/tabulate.py` | `Metric`、`DerMetric` | 单带量；简并组输出组 trace / 组内带数 |
| `system/needed_data.py` | `metric_required_matrices`、`validate_metric_data` | 由手册 AA、FF、D 依赖导出的需求声明与检查 |
| `formula/covariant.py` | `BCP`、`DerBCP`、`Der2BCP` | `mathcal G^(ab[:d[:e]])`；`eq:bcp-code-components`、`eq:sylvester1`、`eq:sylvester2` |
| 同上 | `VelBCP`、`VelDerBCP` | 速度与 BCP／导数的子空间矩阵乘积；trace 非可加 |
| `formula/covariant_basic.py` | `_BCPConnections`、`_bcp_product`、`_commutator` | 新的 Sylvester 连接导数和乘积求导辅助函数；不复制旧 basic 类 |
| `calculators/static.py` | `BCPD_FermiSea`、`BCPQ_FermiSea` | BCP 一／二阶导数的海积分；BCPQ 默认含三阶响应 SI 转换因子（tau=1 fs） |
| 同上 | `BCPD_FermiSurf`、`BCPQ_FermiSurf` | 速度乘积核的非可加累积 trace 配合 `fder=1`；见下文定义边界 |
| `calculators/tabulate.py` | `BCP`、`DerBCP`、`BCPQuadrupole` | 单带／简并组 trace 平均 |
| `system/needed_data.py` | `bcp_required_matrices`、`validate_bcp_data` | 检查 Ham/AA；无 FF 需求 |
| `formula/covariant.py` | `DerYln`、`Der2W`、`Der2V`、`Der4E` | 解析四阶能量导数；轴 `(a,b,c,d)` |
| 同上 | `Der3EVel` | 三阶能量导数与能量梯度的 band/group 乘积；IMQ surface 核 |
| `calculators/static.py` | `IMD_FermiSea/Surf`、`IMQ_FermiSea/Surf` | 二／三阶非线性 Drude 的海面实现和一飞秒 SI 因子 |
| `calculators/tabulate.py` | `IMD`、`IMQ` | 三／四阶能量导数的单带／简并组 trace 平均 |
| `system/needed_data.py` | `imd_required_matrices`、`imq_required_matrices` | 两者都只需要 `Ham`；分别生成至三／四阶导数 |

`Formula_ln`、`FormulaProduct`、`Matrix_GenDer_ln`、`Dcov`、`DerDcov`、
`Der2Dcov`、`Der3E`、`InvMass`、`MassVel`、`StaticCalculator`、`Tabulator`
等上游基础定义均通过 import 复用，没有复制进本目录。

本轮不收录历史 `QuantumMetric`（与标准 G 的归一化／数据范围不一致）、`QMetric_test`、自定义 `tildeFab` 测试分支及轨道四极矩代码。BCP 独立收录，不作为 Metric 的同义入口。

## Metric 修正及语义

1. **DerMetric 外部减项**：中间态必须是 n′∈A：

   `−Abar_mn′^(a:d) Abar_n′n^b − Abar_mn′^a Abar_n′n^(b:d)`。

   已把这两个乘积的四个 `.nl/.ln` 访问修正为 `.nn`。其他内部项、四个交叉项、1/2 系数及 band-space Hermitian 对称化保持推导含义。

2. **符号和单位**：G 是 quantum metric；calligraphic G 是 BCP，BCP 另用 `BCP` 系列入口。Metric kernel 是 Å²，DerMetric kernel 是 Å³。三维 BZ 积分后，静态 Metric 是 Å⁻¹、MetricDipole 是无量纲、MetricQuadrupole_FermiSurf 是 Å。没有附加电荷或 ℏ 的响应 prefactor；不再把 Metric 标作 S/m。

3. **张量顺序**：Metric `(a,b)`，DerMetric `(a,b,d)`，VelMetric `(c,a,b)`，MetricVel `(a,b,c)`，VelDerMetric `(c,a,b,d)`。Ham 一阶导数为 `E^c=∂c ε=ℏ v^c`，不是 SI 速度。费米面类沿用上游 `fder=1` 的 `−f′` 权重。

4. **非可加积分（2026-10-05 修正）**：Metric、DerMetric 及其速度乘积公式保留 `additive=False`。三个静态类直接继承 `StaticCalculator`，不再包装固定组求和。令 `A_r` 为最低 r 条能带组成的子空间、`T_r=Tr_(A_r) X`，上游给区间 `[a,b)` 分配 `T_b−T_a`；零温累加后得到整个占据子空间的 trace。这个过程正确处理非可加性。之前将固定组之和设为唯一 reference，并据此声称需要 `_BandGroupSum` / `_MetricStatic`，属于计算对象判断错误；这两个类已移除。

   对本轮完整 internal 两带模型，全满时子空间 Metric 为零，逐带度规之和却一般非零。两者不是同一物理量。`Tabulator` 仍输出单带或简并组平均；不能将其逐带加权和直接当作新的静态入口的参考。

   `MetricQuadrupole_FermiSurf` 保留原有公式和 `fder=1` 绑定，使用相同的非可加分支。手册的单带积分分部恒等式本身不证明它等同于任意占据子空间的二阶 k 导数；本轮验证其核、指标和上游积分语义，不额外宣称这项物理等价性。
5. **项开关**：`internal_terms` 控制 DD；`external_terms` 控制 FF−AA 以及是否允许外部交叉项；在 external_terms=True 时，`cross_terms=False` 可单独去掉 D–Abar 项。默认三者 True；`external_terms=False` 给内部固定基模型。强制内部模式继续服从 data_K 的 `force_internal_terms_only`。

6. **积分选项**：简并分组、能窗、tetra、`k_resolved` 等交给上游处理；本目录不再另加分组或 `hole_like` 限制。默认 `hole_like=False`，下文的占据子空间结论和测试均采用此默认值；其他占据权重应按上游对应路径理解。
## 数据需求

```python
from calculators_collect.system.needed_data import metric_required_matrices

metric_required_matrices()                       # {'Ham', 'AA', 'FF'}
metric_required_matrices(external_terms=False)   # {'Ham'}
```

DerMetric 所需的 Ham 一二阶导数、AA/FF 的一阶协变导数由 `data_K.covariant` 生成，无需登记为新的独立 R-space 矩阵。

**这里没有伪造 `NeededData(metric=True)` 或 FF loader。** 检查到的上游 System_w90 路径尚不支持直接构造 FF；仅设置 `_getFF=True` 或追加一个文件名映射并不能补齐实现。完整外部项应使用已经提供一致 Ham/AA/FF 的系统，缺少 R-space 矩阵时本目录的校验会明确报错。若只能提供 Ham，可以显式采用内部模型；这不等于完整 ab-initio 几何。

## 使用

将本仓库根目录加入 Python 搜索路径，保留完整 WannierBerri 安装：

```python
import numpy as np
from calculators_collect.calculators import static, tabulate

# 完整模型：system 必须提供一致的 Ham/AA/FF。
calculators = {
    'metric': static.Metric(Efermi=np.linspace(-1.0, 1.0, 101)),
    'metric_dipole': static.MetricDipole_FermiSea(Efermi=np.linspace(-1.0, 1.0, 101)),
}
metric_k = tabulate.Metric()
der_metric_k = tabulate.DerMetric()

# 只有 Ham 的固定基模型应显式选择：
metric_internal = static.Metric(
    Efermi=np.linspace(-1.0, 1.0, 101),
    kwargs_formula={'external_terms': False},
)
```

命名静态类等价于直接将相应公式交给 `StaticCalculator`：Metric / DerMetric 使用 `fder=0`，VelDerMetric 使用 `fder=1`。公式的 `additive=False` 直接传给上游。默认静态结果是占据子空间量，不能解释为固定单带之和。

## 验证

```sh
python -m unittest discover -s calculators_collect/tests -v
```

测试覆盖 8 种项开关组合、单带／多带／空／完整 active space、解析两带模型、光滑四带有限差分、简并组内 gauge 协变、独立 basic 参考式、物理态导数的直接几何、速度乘积轴顺序和真实 StaticCalculator/Tabulator 调用。积分回归直接选择每个费米能下的整个占据子空间，覆盖宽窄窗口一致性、满带零值、精确简并、受控 tetra 权重和费米面差分。

详细依据、版本与限制见 `VALIDATION.md`。测试证明的是列明模型、数据接口及数学约定下的一致性，不替代真实材料数据和 k 网格收敛检查。

## BCP 的定义、修正与使用

详见 [BCP_VALIDATION.md](BCP_VALIDATION.md)。BCP 采用手册 `mathcal G` 的归一化（无额外因子 2），不是 quantum metric `G`。

- `DerBCP` 修正旧的对角速度差分母导数，保留组内完整矩阵乘积。
- `Der2BCP` 按 Sylvester 二阶规则重建，以有限差分和规范协变性验证；不复制旧的 bug TODO 实现。
- `BCP`、`DerBCP`、`Der2BCP` 的 trace 可加，采用 `additive=True`；其速度乘积不可加，采用 `False`，不能套用统一开关。
- `BCPQ_FermiSea` 默认使用 `factor_BCPQ3rd`，即把三维几何积分 Å/eV 转为 m/J，并乘以 `e^4 tau/hbar^2`，其中 `tau=1 fs`。其余 BCP 静态入口仍默认 `constant_factor=1`。
- `BCPD/Q_FermiSurf` 当前定义为非可加速度子空间核的累积差分积分。它们不是自动成立的“孤立带响应”实现，不能未经核对就与相应 sea 类作积分分部等价比较。

```python
from calculators_collect.calculators import static, tabulate
from calculators_collect.system.needed_data import bcp_required_matrices

bcp_required_matrices()  # {'Ham', 'AA'}
# ef 是均匀递增的费米能数组；Ham-only 模型：
bcp_dipole = static.BCPD_FermiSea(
    Efermi=ef, kwargs_formula={'external_terms': False})
bcp_quadrupole = static.BCPQ_FermiSea(
    Efermi=ef, kwargs_formula={'external_terms': False})
bcp_k = tabulate.BCP(kwargs_formula={'external_terms': False})
```

一阶 sea 输出轴为 `(a,b,d)`，二阶 sea 输出轴为 `(a,b,d,e)`；转常用响应排列时只置换末尾 tensor 轴。BCP 的 AA 项仅覆盖已有 Wannier 空间，不能恢复空间外物理能带的分母贡献。全部公式的测试均使用实际上游基础类。
`BCPQ_FermiSea` 返回 SI 缩放后的原始 `Q^(abde)`；具体三阶输运公式要求的三个指标排列仍需由该原始张量组合。

## IMD 与 IMQ

`IMD_FermiSea` / `IMD_FermiSurf` 分别使用上游已有的 `Der3E` 与
`MassVel`。`IMQ_FermiSea` 使用本目录验证后的解析 `Der4E`，
`IMQ_FermiSurf` 使用新增的 `Der3EVel`。原始张量轴统一为
`(a,b,c)` 和 `(a,b,c,d)`：

```python
from calculators_collect.calculators import static, tabulate

imd_sea = static.IMD_FermiSea(Efermi=ef)
imd_surface = static.IMD_FermiSurf(Efermi=ef)
imq_sea = static.IMQ_FermiSea(Efermi=ef)
imq_surface = static.IMQ_FermiSurf(Efermi=ef)

third_energy_derivative = tabulate.IMD()
fourth_energy_derivative = tabulate.IMQ()
```

四个静态入口都按能带或精确简并组求和，保持公式默认的
`additive=True`。`Der3E` 与 `Der4E` 的 group trace 本身满足加性；
两个 surface product 则必须按物理定义逐 band/group 计算。把 surface
product 强制改成 `additive=False` 会得到占据子空间矩阵乘积并引入组间交叉项，
不再是 IMD/IMQ 的带求和。

默认因子采用 `tau=1 fs`。`factor_IMD2nd` 与 WannierBerri 的
`factor_nldrude` 完全相同；`factor_IMQ3rd` 将三维原始积分的
`eV Angstrom` 转为 `J m`。二阶响应中由电场指标约定产生的数值对称化因子
在组合最终电导张量时另行施加。若弛豫时间以 fs 给出，IMD 和 IMQ 的
`constant_factor` 分别再乘 `tau_fs**2` 和 `tau_fs**3`。推导、量纲和验证见
[IMDQ_VALIDATION.md](IMDQ_VALIDATION.md)。
