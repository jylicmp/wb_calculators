# WannierBerri 1.7 / 1.8 自定义改动报告

## 1. 报告范围与结论

本报告比较以下两个已安装环境与对应的官方 PyPI 发布包：

| 归档 | 来源 | 官方基线 | 全包结果 |
|---|---|---|---|
| 1.7 | `SUSTech-Zhu:~/bin/.conda/env/wberri_v1.7_test/.../wannierberri` | PyPI 1.7.0 wheel | 98 个文件中 5 个不同，无新增或缺失文件 |
| 1.8 | `SUSTech-Phys:~/.conda/envs/wberri_v1.8/.../wannierberri` | PyPI 1.8.0 wheel | 94 个文件中 3 个不同，无新增或缺失文件 |

比较排除了 `__pycache__`、`*.pyc` 和 `*.pyo`。所有归档文件均与服务器源文件逐字节一致，详细哈希见 [1.7 清单](v1.7/manifest.json) 和 [1.8 清单](v1.8/manifest.json)。

1.7 的主要新增内容是量子度规、Berry connection polarizability（BCP）及其导数、Berry 曲率/BCP 四极项、三阶电导和一组轨道磁四极矩试验实现。1.8 的主要新增内容是 Gao–Xiao 轨道磁四极矩四项完整分解、总量计算器和有限温度二阶热电张量 `Beta_GaoXiao`。

“新增”均指相对同版本官方发布包新增。两个官方版本自身的差异没有计入用户改动。

### 1.1 重要：FermiSurf 与 FermiSea 的张量指标顺序不一定一致

> **不能直接按相同的 NumPy 数组下标比较部分 Fermi-surface 和 Fermi-sea 计算器的结果。** 两种写法可能表示相同的物理张量，却把速度指标或动量求导指标放在不同的轴上。比较、相加或检查分部积分关系之前，必须先按统一的物理指标约定转置。

根源是 `FormulaProduct` 按公式因子的先后顺序拼接张量轴，而协变导数类把新增的求导指标附加在原张量指标之后。以 BCP 为例（以下顺序不包括结果数组中可能存在的 k 点和费米能前导轴）：

| 公式 | 原始张量轴顺序 | 说明 |
|---|---|---|
| `VelBCP = FormulaProduct([Velocity, BCP])` | `[速度 alpha, BCP beta, BCP gamma]` | 速度指标位于第一个张量轴。 |
| `DerBCP` | `[BCP beta, BCP gamma, 求导 alpha]` | `nn()` 中的输出下标为 `...abd`；求导指标 `d` 位于最后一个张量轴。 |

因此，如果统一采用物理约定 `chi_{alpha,beta,gamma}`，`BCPD_FermiSurf` 的最后三个轴已经是 `[alpha,beta,gamma]`，而 `BCPD_FermiSea` 的最后三个轴是 `[beta,gamma,alpha]`。后者需要把最后一轴移到这三个张量轴的最前面，例如：

```python
# 适用于前面还带有 Efermi 或 k 点轴的结果数组
chi_sea_aligned = np.moveaxis(chi_sea_raw, -1, -3)
```

BCP 四极项存在同类问题：

- `VelDerBCP` 的原始顺序是 `[速度 alpha, BCP beta, BCP gamma, 求导 delta]`。
- `Der2BCP` 的原始顺序是 `[BCP beta, BCP gamma, 求导 alpha, 求导 delta]`。
- 若统一为 `sigma_{alpha,beta,gamma,delta}`，应将 `BCPQ_FermiSea` 的最后四个轴按 `(2, 0, 1, 3)` 重排；有前导轴时应只重排最后四个张量轴。

Berry 曲率四极项也不能直接逐分量比较：`VelDerOmega` 的原始顺序为 `[速度, Omega, 求导]`，`Der2Omega` 为 `[Omega, 求导1, 求导2]`。若采用源码文档中的 `Q_{beta,delta,eta}` 约定，FermiSurf 最后三轴需从 `[beta,eta,delta]` 排为 `[beta,delta,eta]`，FermiSea 最后三轴需从 `[eta,beta,delta]` 排为 `[beta,delta,eta]`。

这个问题不限于 BCP。凡是用 `FormulaProduct` 组合速度、度规或导数的类，例如 `VelMetric`、`MetricVel`、`VelDerMetric`、`VelDerOmega` 和 `VelDerBCP`，都应从公式因子顺序和 `einsum` 输出下标确认张量轴含义。类名相似、张量形状相同并不保证指标顺序相同。

## 2. 官方版本基线

- 1.7 以 `wannierberri-1.7.0-py3-none-any.whl` 为准，SHA-256 为 `c210f1ceffd3a66113753fa915f57b99c232fc4aa12c082366855df7443d247d`。GitHub 标签 `1.7.0` 的提交 `134a94a3188787efd5f6e2fe3ec22fe7cb26c424` 中，98 个包文件与 wheel 完全一致。
- 1.8 以 `wannierberri-1.8.0-py3-none-any.whl` 为准，SHA-256 为 `9c00e5813f88fa25d5db779d2575d62ccfad6391482b7a66de6d8bd30f595535`。GitHub 标签名为 `v1.7.0`，但对应提交 `50265d5d1cef184f8377f87ed691c8d9a2f6cdbf` 的 94 个包文件与 PyPI 1.8.0 wheel 完全一致。
- 两个版本都要求 Python `>=3.11`。源码采集环境均为 Python 3.12。

## 3. 1.7 改动

### 3.1 静态计算器

新增代码位于 [calculators/static.py](v1.7/wannierberri/calculators/static.py)。`fder=0` 表示代码按费米海权重 `f` 积分，`fder=1` 表示按费米面核 `df/dE` 积分。

| 计算器 | 公式类与秩 | 实际行为 |
|---|---|---|
| `AHC_test_Li` | `frml_basic.tildeFc`，二阶 | AHC 基础公式的测试入口，`fder=0`，没有新增单位因子。 |
| `QMetric` | `Metric`，二阶 | 对对称量子度规型矩阵做费米海积分。默认 `constant_factor=1`，输出仍是 WannierBerri 内部单位。 |
| `QMetric_test` | `tildeFab`，二阶 | 未对称的 `tildeFab` 测试入口，用于与 `QMetric` 对照。 |
| `BCPD_FermiSurf` | `VelBCP`，三阶 | 计算速度与 BCP 的乘积，`fder=1`，使用 `factor_BCPD`；原始轴顺序为 `[速度, BCP1, BCP2]`。 |
| `BCPD_FermiSea` | `DerBCP`，三阶 | 计算 BCP 的协变一阶导数，`fder=0`，使用 `factor_BCPD`；原始轴顺序为 `[BCP1, BCP2, 求导]`。 |
| `BCPD_FermiSea_test` | `DerBCP_test`，三阶 | 以拆分的 `M/N` 辅助公式复算上一项的测试实现。 |
| `BerryQuadrupole_FermiSea` | `Der2Omega`，三阶 | Berry 曲率二阶导数的费米海形式，使用 `factor_BCQ3rd`；原始轴顺序为 `[Omega, 求导1, 求导2]`。 |
| `BerryQuadrupole_FermiSea_chain` | `Der2Omega_chain`，三阶 | 与上一项相同目标，但二阶协变导数用链式展开实现，供交叉检查。 |
| `BerryQuadrupole_FermiSurf` | `VelDerOmega`，三阶 | 速度乘 Berry 曲率一阶导数的费米面形式，使用 `factor_BCQ3rd`；原始轴顺序为 `[速度, Omega, 求导]`。 |
| `BCPQ_FermiSea` | `Der2BCP`，四阶 | BCP 二阶导数的费米海形式，使用 `factor_BCPQ3rd`；原始轴顺序为 `[BCP1, BCP2, 求导1, 求导2]`。 |
| `BCPQ_FermiSea_test` | `Der2BCP_test`，四阶 | 拆分辅助公式的测试实现；当前公式函数会主动抛出 `NotImplementedError`。 |
| `BCPQ_FermiSurf` | `VelDerBCP`，四阶 | 速度乘 BCP 一阶导数的费米面形式，使用 `factor_BCPQ3rd`；原始轴顺序为 `[速度, BCP1, BCP2, 求导]`。 |
| `IMQ_FermiSea` | `Der4E`，四阶 | 能量四阶导数，即速度三阶导数对应的逆质量四极项，使用 `factor_IMQ3rd`。 |
| `MetricDipole_FermiSea` | `DerMetric`，三阶 | 量子度规一阶导数的费米海积分。 |
| `MetricQuadrupole_FermiSurf` | `VelDerMetric`，四阶 | 速度乘量子度规一阶导数的费米面测试量。 |
| `Test_InvMass_FermiSea` | `InvMass`，二阶 | 逆有效质量费米海测试入口。 |
| `Test_VelVel_FermiSurf` | `VelVel`，二阶 | 速度外积费米面测试入口。 |

三阶电流约定在源码中写为 `j_a = sigma_abgd E_b E_g E_d`。对应的几何部分包括 Berry 曲率四极项和 BCP 四极项；`IMQ_FermiSea` 则是弛豫时间三次方的 Drude/逆质量四极贡献。

### 3.2 1.7 轨道磁四极矩计算器

以下类也新增在 `calculators/static.py`：

- `Qorb_IMD`：将 `Qorb_IMD` 公式作费米海积分后乘晶胞体积，返回每晶胞的二阶轨道磁四极矩贡献。
- `Qorb_dG` 与 `Qorb_QMD`：分别计算量子度规导数基项和带 `(E-mu)` 权重的 grand-potential 组合；`Qorb_QMD` 将能量加权项减去 `mu` 乘基项，并乘晶胞体积。
- `Qorb_IMD_FermiSurf` 与 `Qorb_QMD_FermiSea`：分别用 `MassVel` 和 `MetricVel` 表示费米面/费米海分解。后者额外乘 `-2/3`。
- `Qorb_QMD_FermiSea_ndim3`、`Qorb_dG_ndim3`、`Qorb_QMD_Grand_ndim3`、`Qorb_IMD_FermiSea_ndim3`、`Qorb_IMD_FermiSurf_ndim3`：保留未做 Levi-Civita 缩并的三阶张量版本，用于检查指标顺序和费米海/费米面关系。
- `Qorb_dG_ndim3_basic`、`Qorb_QMD_Grand_ndim3_basic`：使用 `covariant_basic.py` 的独立实现作对照。

这里的 IMD 是 inverse-mass dipole，QMD 是 quantum-metric derivative。二阶版本把三阶导数张量按 Levi-Civita 张量缩并为 `Q_ij`；`ndim3` 版本保留类似 `Q_ilj` 的三指标形式。源码中的变换属性均设为时间反演奇、空间反演奇。

### 3.3 Tabulator 接口

[calculators/tabulate.py](v1.7/wannierberri/calculators/tabulate.py) 新增 8 个逐 k 点/逐能带入口：

| Tabulator | 对应公式 |
|---|---|
| `BCP` | `frml.BCP` |
| `IMD` | `frml.Der3E` |
| `Metric` | `frml.Metric` |
| `QuantumMetric` | `frml.QuantumMetric` |
| `DerMetric` | `frml.DerMetric` |
| `DerBCP` | `frml.DerBCP` |
| `BerryQuadrupole` | `frml.Der2Omega` |
| `BCPQuadrupole` | `frml.Der2BCP` |

`Metric` 与 `QuantumMetric` 是两套不同构造：前者从 `D`、`AA`、`FF` 组合并显式对称化，后者从 `D` 和 `AA` 构造量子度规。静态 `QMetric` 使用前者，名为 `QuantumMetric` 的 tabulator 使用后者。

### 3.4 协变公式和辅助函数

[formula/covariant.py](v1.7/wannierberri/formula/covariant.py) 没有改写官方已有的顶层类，而是新增以下顶层类：

- 能带间连接和导数：`Mcov`、`Ncov`、`DV_ln`、`DerMcov`、`DerNcov`、`DInvMass`、`Der2Mcov`、`Der2Ncov`、`Der2Dcov_chain`。
- 高阶 Hamiltonian 导数：`DerYln`、`Der2Wnn`、`Der2Vln`、`Der4E`。它们为四阶能量导数、BCP 二阶导数和链式检查提供中间量。
- 量子度规：`Metric`、`tildeFab`、`QuantumMetric`、`DerMetric`。
- BCP：`BCP`、`DerBCP`、`DerBCP_test`、`Der2BCP`、`Der2BCP_test`。
- Berry 曲率替代实现：`Der2Omega_chain`。
- 轨道磁四极矩：`Qorb_IMD`、`Qorb_QMD`、`Qorb_QMDen`、`Qorb_QMD_ndim3`、`Qorb_QMDen_ndim3`、`Qorb_IMD_ndim3`、`Qorb_MassVel_ndim3`。
- 公式乘积：`VelMetric`、`MetricVel`、`VelDerMetric`、`VelDerOmega`、`VelBCP`、`VelDerBCP`。

`BCP` 在代码中以 Hamiltonian-gauge 连接 `M` 和能量分母加权连接 `N` 的反对称组合构造，并对能带子空间及笛卡尔指标作对称化。`DerBCP` 和 `Der2BCP` 继续对连接、能量分母及速度差求协变导数。`Der2BCP_test` 是未完成的检查版本，其 `nn()` 计算后无条件抛出异常。

[formula/covariant_basic.py](v1.7/wannierberri/formula/covariant_basic.py) 新增 `Qorb_QMD_basic` 与 `Qorb_QMDen_basic`：前者从 `tildeFab_d` 构造三阶 QMD 基项，后者乘平均能量 `Eav`，并声明 `additive=False`。

### 3.5 新增单位系数

[factors.py](v1.7/wannierberri/factors.py) 新增四个模块常量：

| 常量 | 代码中的量纲用途 |
|---|---|
| `factor_BCPD` | BCP 偶极/二阶响应；包含 eV 到 J 的能量分母换算。 |
| `factor_BCQ3rd` | Berry 曲率四极三阶电导，含 `tau^2` 和一个长度因子。 |
| `factor_BCPQ3rd` | BCP 四极三阶电导，含 `tau` 及速度单位换算。 |
| `factor_IMQ3rd` | 逆质量四极三阶电导，含 `tau^3` 及速度单位换算。 |

代码沿用上游 `TAU_UNIT=1e-15`，即数值上以飞秒为单位。物理单位报告应依据完整张量定义再次核算；本次只记录代码实际乘入的常数，没有独立证明其 SI 量纲。

## 4. 1.8 改动

### 4.1 Gao–Xiao 轨道磁四极矩

[calculators/static.py](v1.8/wannierberri/calculators/static.py) 新增以下公开计算器。它们都返回二阶张量密度，默认 `constant_factor=1`，因此结果保留 WannierBerri 内部的能量和长度单位。

| 计算器 | 代码实现 |
|---|---|
| `Qorb_GaoXiao_CL` | 经典型项 `(2/3) Re Tr_A[A_i M_j]`，费米海权重 `f`。 |
| `Qorb_GaoXiao_IMD` | 逆质量导数项 `-(1/12) epsilon_{alpha beta j} partial_alpha Gamma_{i beta}`，费米海权重 `f`。 |
| `_Qorb_GaoXiao_ME_base` | ME 项未乘 grand-potential 权重的内部计算器。 |
| `Qorb_GaoXiao_ME` | `-2 Re Tr_A[A_i M_j/(epsilon_A-epsilon_B)]` 乘 `G=(E-mu)f`；代码用能量加权积分减 `mu` 乘基项实现。 |
| `_Qorb_GaoXiao_QMD_base` | QMD 项未乘 grand-potential 权重的内部计算器。 |
| `Qorb_GaoXiao_QMD` | `-(1/3) epsilon_{alpha beta j} partial_alpha g_{i beta}` 乘 `G=(E-mu)f`。 |
| `Qorb_GaoXiao` | `MultitermCalculator`，返回上述 CL、IMD、ME、QMD 四项之和。 |

公式实现以 `Ham` 和 `AA` 为核心输入：`BerryConnectionH` 构造 `A_H=Abar+iD`，`GaoXiaoInterbandM` 构造能带间轨道磁矩矩阵，`GaoXiaoMetricGradient` 构造量子度规协变梯度。各公式设置 `additive=True`，使每个简并能带组的虚空间包含其余全部能带，而不是只包含费米能以上的能带。

ME 和 QMD 的 `(E-mu)f` 写法在源码中明确按零温 grand potential 处理；当前实现没有把这两项推广为有限温 grand-potential 核。

### 4.2 二阶热电张量 `Beta_GaoXiao`

`Beta_GaoXiao` 是直接继承 `Calculator` 的三阶张量计算器，响应约定为

```text
J_a^(2) = beta_abd (partial_b T / T) (partial_d T / T)
```

实现的带分辨核为

```text
beta_abd = sum_n integral[dk]
           1/2 (epsilon_abc theta_cd^n + epsilon_adc theta_cb^n)
           (epsilon_n-mu)^2 df_n/d epsilon_n .
```

关键接口和行为：

- `Efermi` 必须是一维数组。
- 温度可由 `kBT`（eV）或 `T_Kelvin`（K）给出；如果两者都未提供，构造函数报错。
- `kBT` 必须为正。`_dfde()` 保留费米分布能量导数的负号。
- `maxdE` 默认取 `10 kBT`，用于限制参与计算的能带窗口。
- `Beta_GaoXiao_ThetaSym` 显式令后两个场指标 `b,d` 对称。
- `k_resolved=True` 返回 `K__Result`；否则对 k 点取平均并返回 `EnergyResult`。
- 输出除以晶胞体积。默认 `constant_factor=1`，代码未提供专用 SI 单位换算系数。

### 4.3 1.8 新增公式

[formula/covariant.py](v1.8/wannierberri/formula/covariant.py) 新增的 Gao–Xiao 支持层为：

- 连接与磁矩：`BerryConnectionH`、`DerBerryConnectionH`、`GaoXiaoInterbandM`。
- 四项 MQM：`Qorb_GaoXiao_CL`、`Qorb_GaoXiao_IMD`、`Qorb_GaoXiao_ME_base`、`Qorb_GaoXiao_ME_E`、`GaoXiaoMetricGradient`、`Qorb_GaoXiao_QMD_base`、`Qorb_GaoXiao_QMD_E`。
- 热电响应：函数 `_gx_levi_civita`、常量 `_EPS_GX`、类 `GaoXiaoTheta`、`Beta_GaoXiao_ThetaSym`。

同时保留了一批早期/通用公式：`DerYln`、`Der2Wnn`、`Der2Vln`、`Der4E`、`BCP`、`DerBCP`、`Der2BCP`、`Qorb_IMD`、`Qorb_QMD`、`Qorb_QMDen`、`Qorb_QMD_ndim3`、`Qorb_QMDen_ndim3`、`Qorb_IMD_ndim3`、`Qorb_MassVel_ndim3`、`VelDerOmega`、`VelBCP`、`VelDerBCP`。

除 `_gx_levi_civita` 外，没有修改官方已有的顶层函数；上述项目均为新增定义。

### 4.4 1.8 保留的早期轨道磁四极矩计算器

`calculators/static.py` 还新增了 `Qorb_IMD`、`Qorb_dG`、`Qorb_QMD`、`Qorb_IMD_FermiSurf`、`Qorb_QMD_FermiSea_ndim3`、`Qorb_dG_ndim3`、`Qorb_QMD_Grand_ndim3`、`Qorb_IMD_FermiSea_ndim3`、`Qorb_IMD_FermiSurf_ndim3`、`Qorb_dG_ndim3_basic` 和 `Qorb_QMD_Grand_ndim3_basic`。它们与 1.7 中同名实现的总体思路相同，主要用于早期分解和交叉检查。

其中有三项在当前 1.8 归档中缺少所引用的公式类：

- `Qorb_QMD_FermiSea_ndim3` 引用不存在的 `frml.MetricVel`。
- `Qorb_dG_ndim3_basic` 引用不存在的 `frml_basic.Qorb_QMD_basic`。
- `Qorb_QMD_Grand_ndim3_basic` 引用不存在的 `frml_basic.Qorb_QMDen_basic`。

因此这三项导入时可见，但实例化时会报 `AttributeError`。本次按“原样归档”要求没有补写依赖。

### 4.5 `NeededData` 接口变化

[system/needed_data.py](v1.8/wannierberri/system/needed_data.py) 修改了官方已有的 `NeededData` 类、`get_parameters()` 和 `__init__()`：

- 新增参数 `qorb=False`。启用后请求 `AA`，因此 Gao–Xiao MQM 可从 `Ham+AA` 的 tight-binding 数据构造。
- 新增参数 `JJ=False`，并把 `JJ` 与 `AA` 加入矩阵集合；`needed_files['JJ']` 为空。
- `get_parameters()` 会从系统构造参数中提取 `JJ` 和 `qorb`。

当前归档的其他改动文件没有读取 `JJ`。此外，官方 `System_tb` 会拒绝除 `Ham`、`AA` 外的矩阵，因此 `System_tb(..., JJ=True)` 仍会报错。`qorb=True` 已通过检查，所得矩阵集合为 `{'Ham', 'AA'}`。

## 5. 跨版本对照

| 功能 | 1.7 | 1.8 |
|---|---|---|
| 量子度规及一阶导数 | 完整新增，并有静态计算器和 tabulator | 没有以用户改动形式加入同一套公开入口 |
| BCP、BCP 偶极和二阶导数 | 有公式、静态计算器、tabulator 与测试版本 | 保留 `BCP`、`DerBCP`、`Der2BCP` 公式，未加入 1.7 的完整公开计算器集合 |
| Berry 曲率/BCP 四极三阶电导 | 有费米海、费米面及链式检查实现 | 没有对应公开静态计算器 |
| IMQ 三阶电导 | `IMQ_FermiSea` | 没有对应公开静态计算器 |
| 早期轨道磁四极矩 | 二阶、三阶、basic 和费米面/费米海检查较完整 | 保留一部分，但有三个入口缺少依赖 |
| Gao–Xiao 四项 MQM | 无 | CL、IMD、ME、QMD 及总量完整入口 |
| Gao–Xiao 二阶热电响应 | 无 | `Beta_GaoXiao`，支持有限温度和 k 分辨输出 |
| 单位换算系数 | 新增四个三阶/BCP 系数 | 新计算器默认因子为 1，没有专用 SI 系数 |
| 数据依赖接口 | 未修改 | `NeededData` 新增 `qorb` 和 `JJ` |

## 6. 代码审查中发现的限制

以下结论来自静态检查和最小导入测试，不是对物理结果的数值否定：

1. 部分 FermiSurf/FermiSea 实现的张量轴顺序不同。尤其是 `VelBCP` 的速度指标在第一轴，而 `DerBCP` 的求导指标在最后一轴；未经转置的同下标分量没有可比性。详见 1.1 节。
2. 1.7 的 `Der2BCP_test.nn()` 无条件抛出 `NotImplementedError`，所以 `BCPQ_FermiSea_test` 不能完成实际计算。
3. 1.7 的 `tildeFab.nn()` 第一项也由 `external_terms` 控制；如果只开启 internal terms，这一项不会加入。该条件与同文件其他公式的写法不一致，使用 `QMetric_test` 前应核对设计意图。
4. 1.7 的 `BCPD_FermiSea` 文档公式写有 `f'`，但代码设置 `fder=0`，实际使用费米海权重 `f`。报告以代码行为为准。
5. 1.7 的 `QMetric` 文档标注 `S/m`，但构造函数没有设置专用 `constant_factor`。1.7 的多个 MQM 测试入口以及 1.8 的 Gao–Xiao 入口也默认因子为 1。输出单位需要结合 WannierBerri 内部 eV/Å 约定另行换算。
6. 名称含 `_test`、`_chain`、`_basic` 或 `ndim3` 的类属于替代实现或中间张量检查入口，不宜在没有基准模型对照时视为稳定公共 API。
7. 1.8 的三个早期 Qorb 入口缺少公式依赖，`JJ` 参数也没有形成可运行的数据通路；具体情况见 4.4 和 4.5。
8. 本归档没有修改算法，也没有运行材料模型、k 点收敛、简并阈值敏感性、规范不变性或文献数值复现。因此符号、整体单位和物理正确性仍需独立基准验证。

## 7. 验证记录

- 服务器与归档文件：所有 8 个归档文件的 SHA-256 均与来源环境一致。
- 官方源码：GitHub 两个固定提交的包树分别与 PyPI 1.7.0、1.8.0 wheel 完全一致。
- 补丁：两个 `changes.patch` 均已从对应官方 wheel 成功重建；重建后的完整包目录与服务器包逐文件一致。
- 语法：全部 8 个归档 `.py` 文件通过不写入字节码的 `compile()` 检查。
- 1.7 远端检查：成功导入版本 `1.7.0`，并确认主要 static、tabulate 和 factor 符号存在。
- 1.8 远端检查：成功导入版本 `1.8.0`，构造 `Beta_GaoXiao(Efermi=[0], T_Kelvin=300)` 成功，`NeededData(qorb=True)` 返回 `Ham+AA`。

报告符号覆盖检查也已通过：所有相对官方基线新增的顶层类和函数都在本报告中出现。

## 8. 物理依据

- Y. Gao and D. Xiao, “Orbital Magnetic Quadrupole Moment and Nonlinear Anomalous Thermoelectric Transport,” *Phys. Rev. B* **98**, 060402(R) (2018), [DOI](https://doi.org/10.1103/PhysRevB.98.060402)。1.8 的 MQM 四项和 `Beta_GaoXiao` 按源码注释对应其式 (4) 与式 (8)。
- H. Liu et al., “Berry connection polarizability tensor and third-order Hall effect,” *Phys. Rev. B* **105**, 045118 (2022), [DOI](https://doi.org/10.1103/PhysRevB.105.045118)。用于理解 BCP 与三阶 Hall 响应的关系。
- K. Das et al., “Intrinsic nonlinear conductivities induced by the quantum metric,” *Phys. Rev. B* **108**, L201405 (2023), [DOI](https://doi.org/10.1103/PhysRevB.108.L201405)。1.7 的 BCP/量子度规注释直接引用该工作。
- X. Liu, S. S. Tsirkin, and I. Souza, “Covariant derivatives of Berry-type quantities: Application to nonlinear transport,” [arXiv:2303.10129](https://arxiv.org/abs/2303.10129)。对应协变导数、费米海表达和 Wannier 插值背景。
