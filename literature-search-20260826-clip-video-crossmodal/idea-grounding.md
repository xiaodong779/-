# Idea grounding — 用户框架（Vita-CLIP + RAP/BIE/SCR/Text_MSTFF）的跨模态借鉴

## 用户框架现状（source-supported）
- 文本侧：固定模板分支 + prompt learner 可学习分支，`text_fusion_logits` 软融合 → [C,D] 静态类原型。
- 视频侧：time_embed + RAP（自适应 top-k 时空注意力，层 3,5,7,9）+ aggregation mixer → [B,D]。
- 跨模态：BIE —— V→T 收集（`_CrossAttentionTextCollect`：video 为 query 从文本收集上下文）+ T→V 软插值（按预测概率 p 移动），零初始化门控恒等起点；最终 `logit_scale * V @ Q.T` + CE。
- loss：仅 CE（label smoothing 0.1），无辅助 loss。

## 证据卡
- [X-CLIP] 视频内容动态生成文本 prompt（video-specific prompt）→ 文本原型从静态变动态。用户文本侧目前是静态的 [C,D]，只被 prompt learner 全局微调，不感知具体视频。
- [EVL/STAN] 视频内时序建模放哪层、怎么融合 —— 用户已有 RAP + mixer 覆盖。
- [MIP-CLIP DFR vs BIE] 文本原型注入视频：硬加（DFR）会被困文本子空间；软插值（BIE）保持在段内。用户已站在正确一侧。
- [Text4Vis] 视频生成文本描述可增强动词类语义（UCF101/Diving48 类名短，语义弱）。

## 推断的缺口（inferred）
1. **视频自适应文本原型**（最高潜力）：现有双分支都不依赖视频内容。X-CLIP 式：`video_features` 作 query，对 `learnable text [C,D]` 做 cross-attention → 得到逐视频 [C,D] 动态文本原型，再与 fixed 分支融合。改动小、复用 `_CrossAttentionTextCollect` 结构、故事线顺（视频看到了才知道该类"像什么"）。
2. **文本信息在视频内的深注入**：BIE 的 V→T 只在最终特征上做一次收集；可在 RAP 层间让文本原型参与时空注意力（X-CLIP 的 text-to-video attention 变体）。
3. **反向描述增强**（Text4Vis 思想）：用视频生成类描述可考虑，但需要额外解码器，成本高、优先级低。

## 风险
- 动态文本原型会破坏 zero-shot 可解释性（文本不再固定）；若论文声称保留 zero-shot，需加"动态分支只在训练时使用 / 推理退化为静态"的协议。
- X-CLIP 本身是全监督不用视频微调的 zero-shot 协议；借鉴其机制时不要照搬其协议（用户是全监督）。

## 最小可行研究问题（MV-RQ）
在全监督 CLIP 视频识别中，视频内容条件化的文本原型（video-conditioned text prototype）相比静态 prompt 融合，能否在 UCF101/Diving48 上提升 top-1，同时不损害少样本泛化？

## 推荐下一步
1. 小实验：文本侧加 video-conditioned cross-attention 分支（零初始化恒等起点，与 BIE 一致的 warmup 风格），ablation：静态 vs 动态文本分支。
2. 基线：现有 fixed+learnable 融合（已有）；对照：X-CLIP 式 video-specific prompt。
3. 若效果正向，形成"视频自适应的文本语义"叙事 + 与 BIE 双向插值协同。
