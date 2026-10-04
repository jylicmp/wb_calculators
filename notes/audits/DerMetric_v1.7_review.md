# v1.7 DerMetric 与推导的一致性检查

检查日期：2026-10-04。检查对象是本仓库 v1.7 归档实现，不代表其他版本。

结论：**内部项一致；启用 external_terms 时有两项使用了错误的块，因此完整实现与推导不一致。** 问题在 `v1.7/wannierberri/formula/covariant.py:894–901`。本次没有修改归档程序或 LaTeX 手册。

## 1. 对应的数学对象

沿用手册约定：`inn` 是 A，`out` 是 B；m,n∈A，l,p∈B；a,b,d 为笛卡尔指标。

- `.nn(ik, inn, out)` → AA 块。
- `.nl(ik, inn, out)` → AB 块。
- `.ln(ik, inn, out)` → BA 块。
- `.ll(ik, inn, out)` → BB 块。

`data_K.covariant('AA')` 是旋转后的 Wannier 连接矩阵 Ā；`gender=1` 是手册中使用 D 连接的 block-covariant（冒号）导数。`FF` 对应 F̄。`DerDcov` 的能隙符号和求导项与手册一致，其内部 `Ham, gender=1` 在 1.7 基础实现中返回 bare Hamiltonian 一阶导数。

目标是

$$\widetilde g_{mn}^{ab:d}=\frac12\left(\widetilde F_{mn}^{ab:d}+\widetilde F_{mn}^{ba:d}\right).$$

参考 `notes/formulae_temp_clean.tex` 的 `eq:Ffirst`、`eq:Fext-first`、`eq:geometry-derivatives`，以及原始 `notes/Wannier.tex` 的 quantum metric 定义和 AA 投影减项。外部项应为

$$\begin{aligned}
\widetilde F_{\mathrm{ext}}^{ab:d}={}&\bar F_{AA}^{ab:d}
-\bar A_{AA}^{a:d}\bar A_{AA}^{b}-\bar A_{AA}^{a}\bar A_{AA}^{b:d}\\
&+iD_{AB}^{a:d}\bar A_{BA}^{b}+iD_{AB}^{a}\bar A_{BA}^{b:d}
+i\bar A_{AB}^{a:d}D_{BA}^{b}+i\bar A_{AB}^{a}D_{BA}^{b:d}.
\end{aligned}$$

## 2. 逐项核查

| 实现位置（covariant.py） | 对应内容 | 结论 |
|---|---|---|
| 865–873 | 内部 DD 的两个乘积导数 | 一致 |
| 877 | F̄ 的 block 导数 | 一致 |
| 878–893 | 四个 D–Ā 交叉导数 | 一致 |
| 894–897 | 第一个 Ā–Ā 减项 | 错用 AB × BA，应为 AA × AA |
| 898–901 | 第二个 Ā–Ā 减项 | 错用 AB × BA，应为 AA × AA |
| 904 | 加 band-space Hermitian conjugate | 正确；结合前面的 1/2 完成 a,b 对称化 |
| 859–861 | ndim=3，时间反演/空间反演 odd | 对度规导数的 traced tensor 一致 |

代码当前减去的是

$$-\frac12\sum_{l\in B}\left(\bar A_{ml}^{a:d}\bar A_{ln}^{b}+\bar A_{ml}^{a}\bar A_{ln}^{b:d}\right),$$

而应为

$$-\frac12\sum_{n'\in A}\left(\bar A_{mn'}^{a:d}\bar A_{n'n}^{b}+\bar A_{mn'}^{a}\bar A_{n'n}^{b:d}\right).$$

上述两式都还需加上同样的 band-space Hermitian conjugate。差异是中间态的求和空间，不是虚指标改名。两种写法形状都为 `(len(inn), len(inn), 3, 3, 3)`，所以不会靠数组形状报错暴露。

`Metric.nn` 的对应未求导减项（639–642 行）已经正确使用 `.nn × .nn`，因此现有 `DerMetric` 也不是该 `Metric` 的一般外部项导数。

正确替换应仅针对 DerMetric 最后两个乘积：

```python
summ += -0.5 * cached_einsum(
    "mkad,knb->mnabd",
    self.dA.nn(ik, inn, out),
    self.A.nn(ik, inn, out))
summ += -0.5 * cached_einsum(
    "mka,knbd->mnabd",
    self.A.nn(ik, inn, out),
    self.dA.nn(ik, inn, out))
```

这里 einsum 字符 k 只表示 `.nn` 返回数组的中间轴，即 n′∈A；它不是手册中无约束的全空间求和。也可保留原 einsum 字符串，只修改四个访问器；einsum 字符本身不决定 block。

## 3. 数值证据

执行脚本 `check_dermetric_v17.py`，结果保存在 `dermetric_v17_results.json`。脚本直接从归档 AST 加载 `Metric`、`DerMetric`、`DerDcov` 等类和 `covariant_basic.tildeFab_d`；基础 block 接口来自本机已有的 WannierBerri 1.7.0 `formula.py`。缓存 einsum 用 NumPy einsum 替代。

候选修正仅在内存中将最后两个乘积的四个 `.nl/.ln` 调用改为 `.nn`，并未写回生产文件。使用相互一致的解析输入模型，而不是随意生成独立的 Ā、F̄ 和导数：

- Wannier frame 为 S(k)=exp[−i f(k)T]，T 是常数 Hermitian 矩阵，f 为 k 的二次多项式。
- A^(W,a)=(∂a f)T，F^(W,ab)=(∂a f)(∂b f)T²。
- H^W(k) 为常数、一阶和二阶 Hermitian 矩阵组成的光滑 Hamiltonian。
- 每个 k 重新对角化、旋转矩阵，按原基础类构造 block 导数。

### 解析两能带反例

取 H^W=diag(−1,1)、T=σx、f=kx²/2，A 为第一条带，B 为第二条带。此时 D=0，Ā_AA=0，Ā_AB=kx，F̄_AA^(xx)=kx²。因此

$$g_n^{xx}=k_x^2,\qquad\partial_xg_n^{xx}=2k_x.$$

在 kx=0.7，正确值为 1.4。原代码错误减去 AB × BA 的导数，返回 0；候选修正返回 1.4。有限差分亦给出 1.4。

另取 A 为整个两能带空间、B 为空。完整的投影几何为零，导数也应为零；原代码 trace 返回 2.8，候选修正返回 0。这是对完备模型的代数检查，不要求把空补空间用于实际材料流程。

### 四能带检查

表中误差是与 `Metric.trace` 中心有限差分的最大绝对差，覆盖所有 `(a,b,d)` 分量；最终步长 h=10⁻⁴，模型为无量纲测试。每个矩阵参考比较都保留完整 AA 矩阵而不只取 trace。

| 模型/设置 | 原实现误差 | 候选修正误差 |
|---|---:|---:|
| 单 active band，仅内部项 | 2.62×10⁻¹¹ | 2.62×10⁻¹¹ |
| 单 active band，完整项 | 1.42×10⁻¹ | 5.28×10⁻¹¹ |
| 两条 active bands，仅内部项 | 9.84×10⁻¹¹ | 9.84×10⁻¹¹ |
| 两条 active bands，完整项 | 9.24×10⁻² | 8.08×10⁻¹¹ |
| A 内精确简并且与 B 有隙，完整项 | 9.02×10⁻² | 6.22×10⁻¹¹ |

h 从 10⁻² 减至 10⁻³、10⁻⁴ 时，候选修正误差按中心差分的二阶规律下降；原实现完整项误差趋于非零常数。候选修正与 `covariant_basic.tildeFab_d` 做 a,b 对称化后的完整矩阵结果，最大绝对差不超过 2.8×10⁻¹⁷。

复现：

```sh
python check_dermetric_v17.py --formula-base /path/to/wannierberri-1.7.0/wannierberri/formula/formula.py
```

需要 NumPy。归档不是完整可安装包，因此本检查是实际公式方法与解析数据适配器的数值核验，不是完整 Wannier90 文件读取、BZ 积分或材料收敛测试。

## 4. 使用范围与下游影响

- `external_terms=False` 或 data_K 强制只用内部项时，此错误分支不执行。
- 默认 external_terms=True；若外部矩阵非零，结果一般受影响。特殊模型或最终对称性积分可能恰好掩盖错误。
- 原始 axes 正确为 `(m,n,a,b,d)`；取 trace 后为 `(a,b,d)`，求导指标在最后。不存在需要额外乘 2 的问题。
- 默认 `cross_terms` 参数没有被 Metric/DerMetric 单独判断；四个交叉项跟随 `external_terms`。不能依靠 `cross_terms=False` 单独关闭它们。
- 直接调用链包括 `tabulate.DerMetric`、`MetricDipole_FermiSea`、`VelDerMetric` 及 `MetricQuadrupole_FermiSurf`；其他组合类调用 DerMetric 时同样继承此问题。
- Tabulator 对简并组使用 `trace/组内带数`，这是子空间导数的平均值，而非简并点上逐带唯一的导数。
- 此处检查的是固定 A/B 的几何公式。静态积分中的分组/可加性策略、物理响应 prefactor、实际材料输入质量不在本次数值验证范围内。

四个归档 formula/calculator 文件在检查结束时仍与 `v1.7/manifest.json` 中 SHA-256 一致。
