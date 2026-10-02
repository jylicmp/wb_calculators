# WannierBerri 自定义计算器归档

本仓库保存两个服务器环境中相对官方 WannierBerri 发布包发生变化的源码。源码按版本独立归档，没有把 1.7 的功能自动移植到 1.8，也没有修改原始算法。

详细的类、函数、物理含义和已知限制见 [CHANGE_REPORT.md](CHANGE_REPORT.md)。

> **张量指标提示：** 部分 FermiSurf 与 FermiSea 实现虽然描述同一物理响应，但原始数组的指标轴顺序不同。例如 `VelBCP` 把速度指标放在第一轴，`DerBCP` 把求导指标放在最后一轴。逐分量比较或相加前必须先转置到统一的物理指标约定，详见报告的“1.1 重要：FermiSurf 与 FermiSea 的张量指标顺序不一定一致”。

## 目录

```text
v1.7/
  wannierberri/   # SUSTech-Zhu 环境中不同于官方 1.7.0 的完整文件
  changes.patch   # 相对 PyPI 1.7.0 wheel 的补丁
  manifest.json   # 来源、基线、文件哈希和验证状态
v1.8/
  wannierberri/   # SUSTech-Phys 环境中不同于官方 1.8.0 的完整文件
  changes.patch   # 相对 PyPI 1.8.0 wheel 的补丁
  manifest.json   # 来源、基线、文件哈希和验证状态
```

## 官方基线

| 归档 | 主要基线 | GitHub 交叉核验 | 差异 |
|---|---|---|---|
| 1.7 | PyPI `wannierberri==1.7.0` wheel，SHA-256 `c210f1ce...3d247d` | 标签 `1.7.0`，提交 `134a94a3188787efd5f6e2fe3ec22fe7cb26c424` | 98 个包文件中 5 个被修改 |
| 1.8 | PyPI `wannierberri==1.8.0` wheel，SHA-256 `9c00e581...f595535` | 标签 `v1.7.0`，提交 `50265d5d1cef184f8377f87ed691c8d9a2f6cdbf` | 94 个包文件中 3 个被修改 |

官方仓库的标签命名存在容易混淆的情况：`v1.7.0` 对应提交中的包版本实际是 1.8.0，并且其 94 个 `wannierberri/` 文件与 PyPI 1.8.0 wheel 逐文件一致；真正的 1.7.0 对照标签是没有 `v` 前缀的 `1.7.0`。

每个版本的完整下载地址、哈希和逐文件哈希见对应 `manifest.json`。

## 建议的安装与应用方式

两个官方版本均要求 Python 3.11 或更高版本。建议在新环境中安装，避免覆盖原服务器环境：

```bash
conda create -n wberri_1.8_archive python=3.12 pip -y
conda activate wberri_1.8_archive
python -m pip install --index-url https://pypi.org/simple "wannierberri==1.8.0"
```

1.8.0 也可从清华或阿里云 PyPI 镜像安装；核查时两个镜像提供的 wheel 哈希均与 PyPI 官方文件一致。Conda-forge 当前没有 1.8.0，因此推荐用 Conda 创建 Python 环境，再用 pip 安装固定版本。

先定位当前环境的 `site-packages`，再应用相应补丁：

```bash
site_dir=$(python -c 'import pathlib, wannierberri; print(pathlib.Path(wannierberri.__file__).resolve().parent.parent)')
cd "$site_dir"
patch -p1 < /path/to/wb_calculators/v1.8/changes.patch
python -c 'import wannierberri; print(wannierberri.__version__, wannierberri.__file__)'
python -m pip check
```

1.7 使用同样流程，将安装版本和补丁路径分别改为 `1.7.0` 与 `v1.7/changes.patch`。补丁只适用于清单指定的官方基线。

## 验证边界

归档已执行完整目录哈希比较、源码语法检查、补丁重建和两个原服务器环境中的导入/符号检查。没有运行材料模型、k 点收敛或数值基准，因此本仓库不能替代对物理公式、符号、单位和数值收敛性的独立验证。

## 许可证

归档源码来自 WannierBerri，并保留原文件版权头。上游许可证全文见 [LICENSE](LICENSE)。
