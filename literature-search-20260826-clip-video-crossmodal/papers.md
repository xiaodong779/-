# CLIP 视频动作识别：跨模态映射与注意力机制文献调研

搜索目的：direction scouting —— 弄清"别人的 CLIP 视频识别怎么做跨模态（向量空间映射、注意力方向）"，为 Vita-CLIP 框架借鉴机制。

## Top papers（按范式聚类）

### 范式 1：对比对齐式（video-text contrastive alignment，跨模态只在最终相似度打分交互）
| 论文 | Venue | 视频侧 | 文本侧 | 跨模态 |
|---|---|---|---|---|
| ActionCLIP (2109.08472) | AAAI 2022 | CLIP 视觉塔 + 可选时序建模（frame/temporal transformer） | template prompt | InfoNCE video-text 对比 loss，无跨模态注意力 |
| CLIP4Video (2109.05210) | ACM MM 2022 | 帧特征 + temporal 建模 | 文本编码 | 对比对齐 + 动量蒸馏 |
| A5 (2401.11649) | AAAI 2024 | video adapter | prompt | 多任务：alignment + attraction + disentanglement |

### 范式 2：视频引导文本（V→T 跨模态注意力，video-specific prompt）
| 论文 | Venue | 核心机制 |
|---|---|---|
| X-CLIP (2207.07285) | NeurIPS 2022 | 视频侧 cross-frame attention（帧间注意力建模时间）；文本侧 **video-specific prompt**：视频 CLS token 与 text token 做 cross-attention，每个视频动态生成自己的类文本特征；再以 text-to-video 注意力聚合相似度。零样本/少样本，不微调 backbone |
| Text4Vis (2305.15377) | CVPR 2023 | 视频→文本：轻量解码器从视频生成每类文本 prompt（caption 化），增强文本侧语义 |

### 范式 3：文本引导视频（T→V，文本原型注入视频特征）
| 论文 | Venue | 核心机制 |
|---|---|---|
| MIP-CLIP (用户对照) | — | DFR：`V + A @ Q`，用分类概率把文本类原型硬加到视频特征上（用户已有对比实现） |
| BIE（用户已实现） | — | 双向软插值：`V + g_v * CrossAttn(V, Q)`（V→T 收集）+ `Q + g_t * p * (V - Q)`（T→V 插值），零初始化门控恒等起点 |

### 范式 4：深度交互 / 共享融合编码器
| 论文 | Venue | 核心机制 |
|---|---|---|
| EVL (2208.03550) | ECCV 2022 | frozen CLIP 帧特征 + 可训练轻量 temporal transformer decoder + temporal 位置编码，骨干冻结只训 fusion |
| STAN (2301.11116) | CVPR 2023 | 3D patch 化 + spatio-temporal adapter 注入 frozen CLIP，系统分析 temporal modeling 放哪层最有效 |
| TC-CLIP (2404.09490) | ECCV 2024 | 时序上下文化：temporal context 注入 CLIP token |

### 其他值得注意
| 论文 | Venue | 备注 |
|---|---|---|
| Vita-CLIP (2304.03307) | ECCV 2024 | 用户已用框架：视频侧 global/local/summary prompts，文本侧 prompt learning |
| Video Prompting (Ju et al.) | CVPR 2023 | 视觉 prompt 直接放在时间维度 |
| VLPA-CLIP (S0031320325004303) | Pattern Recognition 2025 | 语言 prompt + 视觉 adapter，保留 zero-shot 能力 |

## 图像 + 视频怎么做多模态（第二轮检索）

### 层面 1：预训练级 —— 图像-文本与视频-文本联合对比训练
| 论文 | Venue | 机制 |
|---|---|---|
| Frozen in Time (ICCV 2021) | ICCV | joint image-video encoder，在 image-text 对 + video-text 对上联合训练，让视频模型吃海量图像数据 |
| InternVideo (2212.03191) | ECCV 2024 | masked video modeling + video-language contrastive 联合预训练（视频基础模型） |

启示：图像提供静态语义、视频提供时序语义，联合预训练是"图像帮视频"的底层范式；但用户是全监督下游微调，这条只作背景理解。

### 层面 2：推理/架构级 —— 同一视觉塔出 image + video 双特征，分别与文本交互再融合
| 论文 | Venue | 机制 |
|---|---|---|
| X-CLIP (2207.07285) | NeurIPS 2022 | image 分支（每帧 CLS 特征）+ video 分支（cross-frame attention 后的视频 token）；文本分别做 **text-to-image attention** 和 **text-to-video attention**，两个相似度**加权融合** |
| BridgeFormer (2201.04850) | CVPR 2022 | image bridge（帧特征）+ video bridge（视频 token），文本与两个 bridge 交互，MCQ 多项选择预训练 |

启示：image 级相似度回答"哪一帧/静态外观最像"，video 级相似度回答"整个运动序列的语义"，两者互补——这是"图像和视频一起做"最直接的落点。

### 层面 3：特征聚合级 —— 帧(图像)特征如何聚合成视频特征
| 论文 | Venue | 机制 |
|---|---|---|
| CLIP4Clip (2104.08860) | ACM MM 2022 | 系统性比较三种相似度聚合：无参数（mean pooling / 平均相似度）、顺序型（时序 transformer 聚合帧）、紧密型（merge）；直接研究"image feature 够不够、怎么变 video feature" |
| EVL (2208.03550) | ECCV 2022 | frozen 帧特征 + 轻量 temporal transformer decoder |

启示：视频特征质量取决于帧→视频的聚合方式；用户已有 aggregation mixer，可对照三种聚合消融。

## 核心结论

1. **向量空间**：所有方法都把视频/文本映射到**同一个 CLIP joint embedding space**（ViT-B/16 为 512-D），打分统一是 `logit_scale * sim(V, T)`。没有方法用别的向量空间。
2. **注意力方向是主要分歧点**：
   - 只做视频内时序注意力（ActionCLIP/EVL/STAN）→ 跨模态靠最后的相似度打分；
   - V→T（X-CLIP/Text4Vis）→ 视频决定"该类的文本描述长什么样"；
   - T→V（MIP-CLIP DFR、BIE 的 text→video 插值）→ 文本原型决定"视频特征往哪挪"；
   - 共享融合编码器（EVL decoder/CLIP4Clip）→ 两者 token 级联合建模。
3. **X-CLIP 的 video-specific prompt 是文本侧目前最有代表性的"动态化"机制**：文本原型不是静态 [C,D]，而是逐视频由 cross-attention 生成的。
4. **图像 + 视频的经典做法是"双分支双相似度"**：同一视觉塔输出 image 级与 video 级两路特征，分别与文本交互后融合（X-CLIP 的 text-to-image + text-to-video 加权；BridgeFormer 的 image bridge + video bridge）。

## 扩展视野：完整技术版图（第三轮检索，不再局限于跨模态注意力）

### 路线 A：文本 prompt 学习派（图像 CLIP 分类发源，机制可迁移到视频文本侧）
| 论文 | Venue | 机制 |
|---|---|---|
| CoOp (2109.01134) | IJCV 2022 | 可学习连续 prompt 替代手工模板，最早 |
| CoCoOp (2203.05557) | CVPR 2022 | **条件 prompt**：meta-net 根据每个图像实例动态生成 prompt——与 X-CLIP 的 video-specific prompt 思想同源 |
| MaPLe (2303.03515) | CVPR 2023 | **多模态 prompt**：视觉与文本两侧同时加可学习 prompt，跨层耦合 |
| AAPL / CoCoLe / ATPrompt | 2024-2025 | 属性级 prompt / 概念级 prompt / 属性引导 prompt，强调 prompt 的可解释粒度 |

启示：用户的文本侧 prompt learner 属于 CoOp 派；CoCoOp 的条件化（按实例生成 prompt）和 MaPLe 的多模态耦合（视觉+文本 prompt 联合）是两条可直接借鉴的升级路径。

### 路线 B：adapter 派（frozen CLIP + 轻量特征适配）
| 论文 | Venue | 机制 |
|---|---|---|
| CLIP-Adapter (2110.04544) | ECCV 2022 | 视觉/文本特征后接两层 MLP adapter，残差融合 |
| Tip-Adapter (2202.03929) | NeurIPS 2022 | **cache model**：few-shot 样本特征缓存为"记忆"，无需训练直接推理，Tip-Adapter-F 微调增强 |
| AIM / ST-Adapter (2303.06943) | ICCV 2023 | 把 frozen 图像模型迁到视频：adapter 注入，空间+时间 |
| ViFi-CLIP (2301.01203) | ICCV 2023 | 极简：视频当帧序列过 image encoder + 文本 prompt + 轻量跨帧交互，被证明是强基线 |
| MIST (2303.08954) | AAAI 2023 | frozen CLIP + 视频侧 adapter，少参数高精度 |
| MA-FSAR / EV-CLIP | 2025-2026 | few-shot 视频动作识别：多模态 CLIP 适配 / 视觉 prompt 适配 |

启示：用户已有频率 adapter（visual）和 Text_MSTFF（文本），属此派；Tip-Adapter 的"缓存记忆"思想可作为 few-shot 增强。

### 路线 C：少样本视频（few-shot protocol，用户如果做 few-shot 必看）
| 论文 | Venue | 机制 |
|---|---|---|
| Exploring the Adaptation Strategy of CLIP for Few-Shot Action Recognition (3688863.3689571) | ACM MM 2024 | 系统对比 few-shot 下 CLIP 的适配策略（prompt vs adapter vs 全微调） |
| MA-FSAR | 2025 | 多模态 CLIP 适配 FSAR |
| ActionCLIP (2109.08472) | AAAI 2022 | 自带 zero/few-shot 协议（UCF101/HMDB51 1/2/4/8/16-shot） |
| X-CLIP (2207.07285) | NeurIPS 2022 | few-shot 视频协议最完整（多个数据集 k-shot 表） |

启示：做 few-shot 时直接用 ActionCLIP/X-CLIP 的 k-shot 划分协议（每类 1/2/4/8/16 条），少样本下 adapter 派 + prompt 派表现差异大，值得在用户框架里复现对比。

### 路线 D：视频语言多模态调优（video-language tuning）
| 论文 | Venue | 机制 |
|---|---|---|
| ViLT-CLIP (28347) | AAAI 2024 | Video and Language Tuning CLIP with Multimodal Prompt |
| M2-CLIP / A5 (2401.11649) | AAAI 2024 | 多模态多任务适配：alignment + attraction + disentanglement |
| M2-CLIP++ (S1077314226000755) | CVIU 2026 | dual-branch high-order vision-language adaptation |
| VLPA-CLIP (S0031320325004303) | Pattern Recognition 2025 | 语言 prompt + 视觉 adapter，显式保留 zero-shot |

### 路线 E：注意力池化/时序建模基础件
- RAP (Revisiting Attention Pooling, 2207.00261)：用可学习注意力池化替代全局平均池化，很多 CLIP 视频工作用它做帧/视频聚合。
- Video Prompting (CVPR 2023, Ju et al.)：visual prompt 直接落在时间维度。
