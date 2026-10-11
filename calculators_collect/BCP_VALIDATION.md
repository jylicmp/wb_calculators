# BCP 收录依据与验证

本轮按 `notes/formulae_temp_clean.tex` 的 BCP 定义收录有限 Wannier 模型内的几何核，符号为 `mathcal G`，不同于 quantum metric `G`。不将没有核对的旧 `_test` 分支收入 API。

## 实现与手册对应

| 定义 | 手册标签 | 实现 |
|---|---|---|
| BCP 矩阵 | `eq:bcp-product-components`、`eq:bcp-code-components` | `BCP`：`(T+T†)/4`，`T=N_AB M_BA−M_AB N_BA` |
| 一阶导数 | `eq:sylvester1`、`eq:bcp-derivative-components` | `DerBCP`：保留 A、B 内完整 Hamiltonian 导数乘积 |
| 二阶导数 | `eq:sylvester2`、`eq:bcp-second` | `Der2BCP`：两次分母导数及全部乘积项；有序 `(a,b,d,e)` |
| 连接导数 | `eq:Dfirst`、`eq:sylvester-components` | `_BCPConnections`：只求逆 A−B 能隙，组内简并允许 |
| 速度乘积 | `eq:product-components` | `VelBCP`、`VelDerBCP`，原始轴 `(c,a,b)`、`(c,a,b,d)` |

令 `N=M/Delta`。旧 DerBCP 的对角速度差只覆盖了 Sylvester 导数的对角部分；当 A 或 B 有多个带时，不能代替完整的 `H_AA^:d N−N H_BB^:d`。即使 A 只有一条带，多带 B 也需要这些矩阵元。

二阶连接导数由同一 Sylvester 规则生成，复用上游 `data_K.covariant` 的 comma 导数，不照搬带 bug TODO 的旧 `Der2BCP`，也不依赖未经本轮验证的旧 `Der2Dcov` / `Der2A`。矩阵二阶协变导数有顺序，trace 在光滑子空间上满足 d、e 对称；不能对每个矩阵元擅自强制这个对称性。

## 加性与积分对象

- `BCP`、`DerBCP`、`Der2BCP` 的 **trace 可加**。互换能带后分母反号，合并子空间所移除的成对跃迁贡献抵消。光滑子空间下其 trace 导数也可加。三类使用 `additive=True`。
- `VelBCP`、`VelDerBCP` 的 trace 一般不可加：乘上组内速度矩阵后，上述抵消不再成立。它们保留 `additive=False`。
- `BCPD_FermiSea` / `BCPQ_FermiSea` 使用相应导数和 `fder=0`。
- `BCPD_FermiSurf` / `BCPQ_FermiSurf` 使用速度矩阵乘积和上游 `fder=1`，因此是累积子空间 trace 差分的权重积分。**这些入口不自动等于固定单带 `sum_n (-f'_n) v_n mathcal G_n` 或对应的海积分。** 若用于某篇论文的具体电响应，需先匹配论文的求和对象及指标组合；本轮没有把这两种对象混成同一参考。

`BCPQ_FermiSea` 默认采用 `factor_BCPQ3rd`：将三维 BCPQ 几何积分从 Å/eV 转为 m/J，并乘以 `e^4 tau/hbar^2`，取 `tau=1 fs`。其余 BCP 静态类保持 `constant_factor=1`。BCPQ 的默认单位因此为 `A m/V^3`，但输出仍是原始 `Q^(abde)`；特定三阶输运表达式中的多个指标排列需要另行组合。外部仍可通过 `constant_factor` 覆盖默认值。

BCP 核单位为 Å²/eV；一、二阶导数为 Å³/eV、Å⁴/eV。三维 k 积分后 dipole 为 eV⁻¹，quadrupole 为 Å/eV。Sea dipole 原始轴 `(a,b,d)`，转常用 `(d,a,b)` 用 `(2,0,1)`；sea quadrupole 转 `(d,a,b,e)` 用 `(2,0,1,3)`。这些置换不改前导 k/EF 轴。

## 数据与项开关

- 全部有限模型项需要 `Ham`、`AA`；没有 `FF` 需求。导数由实空间数据生成，Der2BCP 需要 Ham 三阶、AA 二阶 comma 导数。
- `internal_terms` 控制 DD，`external_terms` 控制 AA，`external_terms and cross_terms` 控制两个 DA 混合项，和 Metric 的开关约定一致。修正旧代码把所有 BCP 项嵌套在 `internal_terms` 分支中的问题。
- `force_internal_terms_only` 沿用上游开关；缺 AA 时报错或显式选 internal 模型。
- 跨 A、B 的能隙若被上游 cutoff 置零，明确拒绝除以该能隙；应把简并带放在同一子空间。本轮未引入额外正则化参数。
- AA 的加入不恢复截断 Wannier 空间之外所有物理能带的能量分母贡献。

## 本地验证

`tests/test_bcp.py` 的 11 项测试与原 Metric 的 15 项测试共同运行。使用固定随机种子的一致移动基模型，并检查：

1. 独立显式谱求和与完整 BCP 矩阵的 1/4 归一化。
2. 8 种项开关、单带／多带／非连续 active space 的一、二阶 trace 有限差分。
3. BCP 本身的二阶差分，避免只验证两条彼此依赖的导数实现。
4. 用子空间 polar 对齐的完整矩阵平行输运有限差分。
5. 全部子集的 trace 加性；速度乘积的非加性反例。
6. 精确简并组内随机 unitary 旋转协变性；跨组零能隙报错。
7. 解析两带结果 `BCP_xx=-0.245`、`DerBCP_xxx=-0.7`、`Der2BCP_xxxx=-1`。
8. 真实 StaticCalculator／Tabulator、宽窄 EF 窗口、表面中心差分、归一化、张量轴和项数据需求。

完整物理响应的有限温度、tetra 收敛和真实材料数据不在这组解析模型测试的结论范围内。真实 Haldane/FeS 测试记录另附于本节之后。

## Haldane / FeS 独立模型验证（2026-10-05）

SUSTech-Zhu 的 Slurm 数组 `980440_[0-1]` 均正常完成（exit 0），使用 `256G56c/qos1h`、每任务 1 节点 1 CPU。全部脚本、源代码快照、原始张量和 PNG/PDF/SVG 图保存在：

`/share/home/zhuhaiyuan/NLC/test_wb_calculators_v1.7/bcp/`

报告为 `REPORT.md`，完整误差为 `results/validation.json`。模型沿用既有 PythTB Haldane / FeS 参数，仅计算 internal terms。独立 NumPy reference 由 GPT-5.6 Luna 编写，经主代理检查接口和导数轴顺序；不导入 WannierBerri 或本目录公式。BCP 从谱求和得到，一阶导数由本征态微扰得到，二阶参考由独立一阶导数做笛卡尔中心差分（h=1e-5）。

21² 和 41² 网格全部通过。以下为 41² 全张量逐点最大绝对差除以参考张量的最大绝对值（不是每个元素的相对误差）：

|模型|BCP|DerBCP|Der2BCP（有限差分参考）|
|---|---:|---:|---:|
|Haldane|2.115e-15|4.608e-15|6.743e-10|
|FeS|2.578e-14|3.661e-14|1.427e-08|

BCP/DerBCP/Der2BCP 的直接组求和、native `additive=True`、native `additive=False` 及独立参考在误差范围内一致。两种 native 算法与直接积分的最大绝对差（两个模型、两个网格、全部核）为 3.2e-12。宽／窄 EF 窗口也一致。独立二阶有限差分并非机器精度参考：FeS 41² 的逐点绝对差最大 0.0135，但参考峰值约 9.47e5；积分绝对差最大 4.22e-5。另在一般 k 点将差分步长从 1e-3 降至 1e-5，Haldane 的尺度归一化误差从 1.31e-6 降至 1.35e-10，FeS 从 4.51e-5 降至 4.58e-9，符合中心差分收敛行为。

旧对角分母求导规则在 Haldane 两带测试中与新实现一致；在 FeS 采样子空间中，DerBCP trace 最大差为 0.278649，完整矩阵最大差为 0.264488。该对比保留同一连接导数，仅替换分母求导规则，定位的是遗漏的多带 Sylvester 项。

本地 1.7 reference、1.8 port-audit 和真实远端 1.7 环境分别通过全部 26 项回归。模型图同时显示独立 reference。这里验证同一网格下公式和算法的一致性，未宣称 21²／41² 已达到连续 BZ 积分收敛，也未将速度乘积表面核等同于完整电响应。

## Static calculator 文档格式

所有七个 static 类按照归档 v1.7 的 Sphinx raw docstring 格式组织：名称与单位、`| With Fermi ... integral`、`| Output: :math:...`、指标顺序与必要说明。输出公式按实际 `fder`、归一化和积分对象填写：sea 使用 f；surface 使用正的 −f′ 权重，BCP 用 `mathcal G`，Metric 用 G。
