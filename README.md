# Case10 补充标注前后指标对比

使用 `VesMamba_HiPaS_AirRC_metrics.zip` 中的指标代码，对 Case10 初始分割与补充标注进行逐体素、表面、骨架代理及连通块比较。

**按用户指定，将补标后版本作为本次评测的绝对真值（GT），以下是补标前结果的指标。** 动脉、静脉、气道沿用同方向的实算结果；肺已另行调用相同函数补齐。

| 补标前待评测结构 | Dice | clDice | NSD（1 mm） |
|---|---:|---:|---:|
| 肺动脉 | 98.65% | 93.53% | 97.15% |
| 肺静脉 | 99.33% | 98.18% | 98.67% |
| 气道 | 100.00% | 100.00% | 100.00% |
| 肺 | 100.00% | 100.00% | 100.00% |

[下载当前 GT 定义下的完整指标 Excel](Case10_以补标后为GT_补标前指标.xlsx) · [CSV](results/以补标后为GT_补标前指标.csv) · [JSON](results/user_designated_gt_metrics.json)

气道、肺两版逐体素相同，所以对该真值的得分为100%。肺的clDice按相同骨架公式补算，仅供统一表格比较；解释管状连通性时主要关注气道和血管。

**气道和肺掩码逐体素完全相同。修改集中在肺动脉、肺静脉：主块占比提高、非主块体素减少，同时产生新的跨类别重叠。**

## 查看结果

- [完整 Excel 指标表](Case10补标前后指标对比.xlsx)：9 个工作表，含结构变化、版本一致性、骨架代理、连通块去向及重叠区域定位。
- [中文结果解读](结果解读.txt)
- [完整机器可读结果](results/comparison_report.json)
- [逐结构原始指标和 CSV](results/)
- [原始工具包](source_archive/VesMamba_HiPaS_AirRC_metrics.zip)及[解压后的原始代码](tool/pulmonary_metrics/)

| 指标 | 肺动脉：补标前 → 后 | 肺静脉：补标前 → 后 |
|---|---:|---:|
| 体积（mL） | 312.007 → 314.785 | 359.932 → 361.034 |
| 连通块数（26 邻域） | 132 → 3 | 86 → 46 |
| 最大连通块占比 | 97.9097% → 99.9736% | 98.5980% → 99.9578% |
| 主块外体素数 | 13,569 → 173 | 10,499 → 317 |
| BC 分叉簇代理计数 | 740 → 865 | 958 → 988 |
| SL 骨架体素数 | 20,351 → 22,280 | 23,134 → 23,375 |

连通块减少包含连接和删除两种来源。动脉原非主块中 30 个的保留部分归入新主块、100 个全部删除；静脉对应为 18 个和 29 个。另有纯新增孤立块：动脉 1 个，静脉 6 个。

| 两版本一致性，以补标后为参考 | 肺动脉 | 肺静脉 |
|---|---:|---:|
| Dice | 98.65% | 99.33% |
| clDice | 93.53% | 98.18% |
| NSD（1 mm） | 97.15% | 98.67% |
| HD95（mm） | 0 | 0 |

补标后新增动静脉重叠 **836 体素 / 24 个区域**，气道与动脉重叠 **27 体素 / 11 个区域**。定位信息见 [overlap_regions.csv](results/overlap_regions.csv)。

## 解释边界

- 当前评测采用用户指定的补标后 GT，报告补标前的得分。该参考标注并非独立于两个版本的外部验证集；未将 GT 对自身的满分当作一次独立性能实验。原9工作表及历史JSON保留首次“版本一致性”的计算说明，其方向和数值与当前GT评测一致，肺的补算结果见新增GT表。
- BC / SL 是同一原生网格上的 Lee 骨架代理。SL 单位是体素，不是毫米；BC 不是经过人工确认的真实分支数。数量增加不能单独证明解剖正确性或分支召回提高。
- HD95 使用表面元面积加权的双向 95% 距离分位数，再取较大值。局部修改可以不改变该分位数，因此 HD95 为 0 不代表所有边界一致。
- NSD 的 1 mm 是本次探索性阈值；指标包为独立实现，未宣称与论文官方评测器完全等价。
- AUC 缺少概率图；BDR / DLR 缺少经过确认的分支 ID 中心线真值；MCS 因补标后的动静脉标签重叠而未计算。缺失值保持 NA。
- 所有主指标在相同原始网格计算：512 × 512 × 657，间距约 0.876953 × 0.876953 × 0.625 mm，无重采样、平滑或标签覆盖。

## 仓库内容

仓库包含本次分析代码、汇总结果、审计记录和原始指标包。**实际病例的 CT、分割影像、场景、截图及增删/重叠影像掩码未上传。** `tool/pulmonary_metrics/demo/` 内的小型影像是指标包自带的合成演示数据，可由 `make_demo.py` 生成。

发布副本中的本机绝对路径已替换为 `data/before/Label`、`data/after`、`inputs` 等通用引用；这些路径不表示实际影像已包含在仓库。计算数值和原始数据 SHA256 保留。路径替换与导出校验记录见 [publication_manifest.json](publication_manifest.json)。

## 本地复现本次 Case10

建议使用 Python 3.12。实际输入由使用者保存在本地；复现入口会核对四个结构的源文件 SHA256，确保对应本报告的两版 Case10。`overlap_audit.py` 和 `build_report.py` 含本病例的已核验数值与结论，不应直接用于其他病例或其他标注版本。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tool/pulmonary_metrics -p "test*.py"
```

本地输入文件：

```text
data/before/Label/
  airway.nii.gz
  lung.nii.gz
  pulmonary_artery.nii.gz
  pulmonary_vein.nii.gz
data/after/
  airway.seg.nrrd
  lung.seg.nrrd
  pulmonary_artery.seg.nrrd
  pulmonary_vein.seg.nrrd
```

先核验输入，再运行完整流程：

```powershell
.\.venv\Scripts\python.exe run_case10.py --before-dir data/before/Label --after-dir data/after --check-inputs
.\.venv\Scripts\python.exe run_case10.py --before-dir data/before/Label --after-dir data/after --output local_runs/case10
```

也可将 `--before-dir`、`--after-dir` 指向本机其他位置。输出目录必须尚不存在，避免覆盖已有分析。流程在新目录复制原始工具包，依次运行格式/网格核验、三套指标、连通块追踪、重叠定位和报表导出。该目录下才会生成实际的 `inputs/` 和 `changes/` 影像；`local_runs/` 已被 Git 忽略。

## 验证记录

原指标包的 26 项测试通过；四结构的 Dice / Precision / Recall 已按体素计数交叉核对。源文件和原始工具包的哈希一致，原始病例未修改。上传前再次检查了打包代码、原工具文件哈希和发布文件清单。
