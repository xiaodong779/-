# 训练证据与日志索引

## 代码位置
- 训练仓库：`/hy-tmp/Vita-CLIP/`
- 训练脚本：`/hy-tmp/Vita-CLIP/train_scripts/run_ucf101_rap_cmsi_v4.sh`
- 日志目录：`/hy-tmp/Vita-CLIP/training/logs/`

## 关键日志
| 实验 | 日志文件 | 状态 |
|---|---|---|
| CMSI v4 (UCF101, bs16) | `ucf101_rap_bs16_cmsi_v4-20260821_083132.log` | 训练中 (2026-08-21 08:31 启动) |
| CMSI v3 (UCF101) | `ucf101_rap_bs16_cmsi_v3-*.log` | 完成, best 94.21 (用户报告) |

## 数据集
- UCF101 split1
- （历史：Diving48 数据已准备但论文不使用，见下方 RESOUND 条目说明）

## 已知数值 (截至 2026-08-21)
- RAP baseline (UCF101): **93.84%**
- CMSI v3 (UCF101): **94.21%**
- CMSI v4 (UCF101): **PENDING** (目标 ≥95%)

## 说明
- 论文中的最终数字必须与日志核对一致后再写入 manuscript。
- v4 完成前，正文中涉及主实验结果的句子用 PENDING 标记。

## 文献验证记录 (2026-08-21)
`main.bib` 已全面修复，以下条目经 web 检索核对：
- **Vita-CLIP** (wasim2023vitaclip): 标题为 "Vita-CLIP: Video and text adaptive CLIP via Multimodal Prompting"（非 Multiscale Temporal Modeling），作者 Wasim/Naseer/Khan/Khan/Shah，CVPR 2023。
- **RAP** (cao2024rap): 即 "RAP: Efficient Text-Video Retrieval with Sparse-and-Correlated Adapter"，Cao et al., ACL Findings 2024。**这是 MGTA 的直接 baseline，Related Work 中已明确点出。**
- **RESOUND/Diving48** (li2018resound): ECCV 2018，作者 Li, Li, Vasconcelos。**已从论文移除（用户确认不使用 Diving48）。**
- **VideoMamba** (liu2024video_mamba): 作者为 Kunchang Li 等，ECCV 2024（原条目作者为编造）。
- **X-CLIP** (ni2022xclip): 正确标题 "Expanding Language-Image Pretrained Models for General Video Recognition"，ECCV 2022。
- **ViT-Adapter** (chen2023vitadapter): 作者 Zhe Chen 等，CVPR 2023（原条目作者/年份错误）。
- **ST-Adapter** (pan2022st_adapter): NeurIPS 2022，替换不存在的 "Structure-preserving Video Learning"。
- **TimeSformer** (bertasius2021timesformer): ICML 2021，替换编造的 ARN 条目。
- **CLIP-ViP** (xue2023clipvip): ICLR 2023，替换不存在的 "CLIP It Yourself"。
- **CLIP2Video** (ju2021clip2video): ACM MM 2021。
- 已删除编造条目："CLIP It Yourself"、"ARN: Attention-based relation network"、"Structure-preserving Video Learning"。
