# MATE-CLIP 结果表 (待 v4 训练完成后填充)

## Table 1: 主实验 Top-1 Accuracy (%)
| Method | UCF101 |
|---|---|
| Frozen CLIP baseline | — |
| Vita-CLIP | — |
| + RAP baseline | 93.84 |
| + MGTA (Ours) | — |
| + MGTA + AMTF | — |
| + MGTA + AMTF + BFEM (MATE-CLIP, Ours) | 94.21 (v3) / **PENDING v4** |

## Table 2: 消融
### AMTF 策略
| 策略 | Acc |
|---|---|
| EOT only | |
| masked-mean only | |
| masked-attention only | |
| 三策略融合 | |

### MGTA 稀疏度
| top-k | Acc |
|---|---|
| 全覆盖 (k=all) | |
| k=1/4 | |
| k=1/2 | |

### BFEM 方向
| 方向 | Acc |
|---|---|
| V→T only | |
| T→V only | |
| 双向 | |

## Table 3: 鲁棒性
| Backbone | MATE-CLIP |
|---|---|
| ViT-B/16 | |
| ViT-L/14 | |
