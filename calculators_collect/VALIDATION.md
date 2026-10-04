# Metric / DerMetric 收录核验

日期：2026-10-05。修正静态积分语义并重新运行测试。

## 理论标准

对照当前 `notes/formulae_temp_clean.tex` 的以下标签（标签随排版稳定）：

- `eq:component-sums`：A 内和 B 内中间态的区分。
- `eq:Finterpolation-components`：`Fint=-DD`，`Fext=Fbar−Abar_AA Abar_AA+iD_AB Abar_BA+iAbar_AB D_BA`。
- `eq:projected-geometry-components`：`G=(F^(ab)+F^(ba))/2`。
- `eq:Ffirst-components`：内部 2 项、外部 7 项的显式求导。
- `eq:geometry-derivative-components`：求导之后做相同的 a,b 对称化。
- `eq:Dfirst`、`eq:interpolation-derivative-components`：复用的上游 DerDcov 和 Matrix_GenDer_ln。
- `eq:sea-surface`、`eq:sea-surface2`：积分分部、费米面权重和指标顺序。
- 子空间 metric 的讨论：`Tr_A G` 一般不等于各个孤立带 metric 的和；上游 `additive=False` 使用累积子空间 trace 差分，正确处理这种非可加性。

## 已执行的验证

在本机已有的完整 WannierBerri 1.7.0 参考目录，以及基于 WannierBerri 1.8.0 的 port-audit 目录运行同一测试套件。两者都通过实际包导入，不使用前一次报告的 AST/mock 基础类来代替积分器。1.8 目录含其他历史移植代码；本轮的 Metric 来自本目录，基础类、DerDcov、数据分组、积分／tabulation 则来自各自环境。具体路径、复用类的指纹、手册和汇总代码 SHA-256 记录在 `validation_manifest.json`。

| 验证 | 检查内容 |
|---|---|
| 解析两带模型 | kx=0.7 时 Gxx=0.49、∂xGxx=1.4；旧 DerMetric 的外部减项错误被排除 |
| 完整／空 active space | 完备物理模型中投影几何为零；矩阵形状和导数均正确 |
| 独立几何表示 | 直接从物理态连接 M=Abar+iD 构造 Q，再取 Cartesian 对称部分 |
| 上游 basic 参考 | 与 tildeFab、tildeFab_d 的对称化全矩阵比较 |
| 中心有限差分 | h=10⁻⁴，三方向、全部 a,b,d 分量；8 种开关、1/2 条 active bands；绝对容差 2×10⁻⁹ |
| A 内精确简并 | 与 B 保持能隙；随机组内 unitary 下矩阵协变，trace 导数与有限差分一致 |
| Hermiticity / axes | 能带厄米对称、a,b 对称、metric trace 半正定、3 种速度乘积矩阵顺序 |
| Fermi-sea 回归 | 每个 EF 下直接选择整个占据子空间作为参考；宽窄窗口一致；完整物理模型满带为零，且区别于非零的单带度规之和 |
| Tetra 调用路径 | 使用真实 StaticCalculator 的 tetra 分支，以受控权重比较两个单带区间、合并区间及满带区间的子空间结果 |
| Fermi surface | 对逐 EF 直接选取的占据子空间 VelDerMetric trace 做化学势中心差分，核对上游 fder=1、归一化、(c,a,b,d) 顺序 |
| Tabulator | 真正调用原 Tabulator，验证简并组 trace / 组大小 |
| 数据需求／绑定 | 缺 FF 报错、强制内部模式；三个静态入口直接绑定非可加公式，精确简并按上游分组处理 |

测试使用一致的移动 Wannier frame `S(k)=exp(−i f(k)T)`、二次 f 和二次 Hermitian H^W，而非互不相干的随机 AA/FF 数组。随机种子固定，数值模型无量纲。

## 明确边界

- 本地回归使用解析模型；此前另有用户提供的紧束缚模型测试，见下文。没有将这些测试等同于真实 Wannier90 材料收敛验证。
- tetra 测试验证非可加 trace 与上游分支的衔接，不独立重验上游 tetra 插值算法。
- 没有把 FF 的需求声明冒充文件构造实现。当前仅在已有一致 FF 的 data_K 上运行完整外部公式；内部模型可仅用 Ham。
- 原始 v1.7/1.8 归档不改动，历史 QMetric 名称不进入新 API。
- Metric 不拷贝原有 basic 定义；后续 BCP 的新 Sylvester 辅助函数记录在 `BCP_VALIDATION.md`。NeededData 原类同样不拷贝。

## 本次修正与既有模型证据

删除 `_BandGroupSum` / `_MetricStatic`，三个静态入口直接使用 `StaticCalculator`。此前通过的“满带仍应保持各固定组之和”测试目标错误，本次改为占据子空间参考；旧的通过记录不再作为新入口的验证依据。保留 Metric/DerMetric 及速度乘积的 `additive=False`，保留已经核对的 DerMetric 外部 AA-AA 减项修正。

此前在 SUSTech-Zhu 的 `/share/home/zhuhaiyuan/NLC/test_wb_calculators_v1.7` 完成的独立结果支持这一修正：

| 独立累积子空间积分对原生 False，81² | Metric 最大绝对差 | DerMetric 最大绝对差 |
|---|---:|---:|
| Haldane | 4.16×10⁻¹⁶ | 1.85×10⁻¹⁶ |
| FeS（避开零温占据边界的相同 EF 网格） | 1.22×10⁻¹⁵ | 1.68×10⁻¹⁴ |

Berry curvature 在 Haldane/FeS 的 41²、81²、161² 网格及宽窗、窄窗、满带窗口下，四种算法均一致；161² 的最大积分差分别为 1.24×10⁻¹⁵ 和 2.50×10⁻¹⁵。这对应曲率 trace 的可加性，不能据此要求非可加 Metric 的不同计算对象也一致。模型原始结果和完整代码在远端报告 `BERRY_CONTROL_REPORT.md`；本次修改没有覆盖那一轮用于比较四种算法的历史快照。

这些远端模型仅验证 internal terms。外部项依据本地一致的移动基模型、直接物理态几何、basic 参考式与有限差分验证。`MetricQuadrupole_FermiSurf` 的测试验证当前 VelDerMetric 核和 fder=1 的组合，不把单带积分分部关系自动提升为任意子空间二阶 k 导数的恒等式。

## BCP 增补

BCP 的 11 项新回归与上文 15 项 Metric 回归共同运行，当前共 26 项。数学定义、已修正的分母导数、加性、prefactor 约定和模型证据见 [BCP_VALIDATION.md](BCP_VALIDATION.md)。
