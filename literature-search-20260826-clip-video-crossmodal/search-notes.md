# Search notes — CLIP video cross-modal mechanisms

日期：2026-08-26
模式：exploratory（方向调研，为 Vita-CLIP 借鉴机制）
目的：弄清别人的 CLIP 视频动作识别如何做 (a) 向量空间映射 (b) 跨模态注意力，用于改进用户框架。

## 使用的查询
1. "CLIP video action recognition cross-modal attention text-video alignment method X-CLIP ActionCLIP"
2. "frozen CLIP video action recognition temporal modeling prompt learning 2023 2024 methods"
3. "X-CLIP NeurIPS 2022 cross-frame attention video-guided text prompt text-to-video attention mechanism"
4. "video action recognition text-to-video video-to-text bidirectional cross-modal attention CLIP"
5. "Vita-CLIP ECCV 2024 video prompts learnable discriminative temporal modules"
6. "STAN CVPR 2023 spatio-temporal adapter frozen CLIP cross-modal"
7. "EVL ECCV 2022 frozen CLIP video transformer temporal modeling"

## 来源策略
- 主要用 arXiv 稳定链接 + CVF/Springer/ACM/Elsevier 会议页；网页仅用于发现候选，最终链接指向 arXiv/官方 proceedings。
- 排除：MDPI 来源（"Adapting CLIP for Action Recognition via Dual Semantic" 为 MDPI Electronics，政策排除，未纳入）。

## 候选筛选
screened ≈ 16，final = 10（papers.csv）。

## 机会点映射（exploratory）
- 对比对齐范式（范式1）：crowded，机制成熟，用户已有。
- V→T 视频引导文本（范式2）：X-CLIP 是代表性的 zero-shot 方案；在全监督框架里把"视频自适应文本原型"与已有的 prompt learner 结合，仍未成为标配 —— mechanism gap。
- T→V（范式3）：用户已有 BIE 双向插值，且发现 MIP-CLIP DFR 的硬加问题（被困在文本子空间）—— 这是用户框架的相对优势点。
- 深度融合（范式4）：计算代价高，用户 aggregation mixer 已有轻量替代。
- 建议主攻：video-specific text prompt（X-CLIP 思想）以轻量方式融入现有双文本分支。

## 质量评分说明
X-CLIP 评分最高（insight 5）：video-specific prompt 用 cross-attention 动态生成文本原型，是最具代表性的跨模态注意力机制，且实验证据充分（zero-shot 多个数据集）。

## 排除记录
- MDPI Electronics 2024 "Adapting CLIP for Action Recognition via Dual Semantic..."（policy exclusion）。
