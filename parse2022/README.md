# PARSE2022 三例 TotalSegmentator 肺动脉测评

2026-10-22 完成。三例 CT：PA000005、PA000016、PA000024；以各自原始 PARSE label 为参考真值，评价 `lung_arteries.nii.gz`。

| 病例 | Dice | clDice | NSD（1 mm） | HD95（mm） |
|---|---:|---:|---:|---:|
| PA000005 | 73.99% | 72.18% | 75.06% | 8.91 |
| PA000016 | 72.23% | 69.11% | 72.18% | 12.89 |
| PA000024 | 73.37% | 72.17% | 74.53% | 9.95 |
| 均值 ± 样本标准差 | 73.20 ± 0.89% | 71.16 ± 1.77% | 73.92 ± 1.53% | 10.58 ± 2.07 |

[完整 Excel](results/PARSE2022_TotalSegmentator_metrics.xlsx) · [逐例 CSV](results/metrics_per_case.csv) · [宏平均 CSV](results/metrics_macro.csv) · [完整 JSON](results/metrics_summary.json) · [中文结果解读](results/测评结果解读.txt)

## 测评方法

实际推理版本 TotalSegmentator 2.13.0，`lung_vessels` 任务，GPU；`fast=False`、`save_lowres=False`，无 ROI 子集。GT 前景值为 1。评估核对 CT、GT 和预测的形状、单位与 affine，不额外重采样或后处理。评估仅移除共同背景以降低内存，全部前景和原始间距保留。

指标沿用仓库中的 [pulmonary_metrics](../tool/pulmonary_metrics/) 独立实现。clDice 使用 Lee 三维骨架；NSD 为表面元面积加权、容差 1 mm；HD95 为两个方向各自面积加权 95% 分位距离的较大值。三例等权汇总，标准差分母为 n−1。完整协议、运行环境和输入哈希见逐例 JSON 与中文解读。

平均 Precision 为 60.01%，Recall 为 94.07%。相对所给标签，标签外阳性较多；具体解剖原因仍需叠加影像核查。只有肺动脉真值，静脉、气道、气道壁没有评分。仅三例描述性测评，与模型训练集的独立性尚未核实。

## 发布范围与来源

本目录发布已完成运行的 Excel、CSV、JSON 和结果解读；CT、GT、预测 NIfTI 仍保留在本地，不包含在此次结果归档中。原有 Case10 内容保留在仓库根目录。

发布副本中的本机和服务器绝对路径已替换为通用引用，主机名已隐藏。数值和原始证据 SHA256 保留；JSON 中原有哈希指向原始运行文件，不代表脱敏副本的哈希。原始文件与发布副本的哈希见 [publication_manifest.json](publication_manifest.json)。未重新运行分割，本次核对 Excel 单元格和逐例评分数值后归档。
