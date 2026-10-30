# 书法轨迹模型

当前主线是“有序笔画预测 → 单笔轨迹生成”。两层代码和串联推理适配器已实现，分别独立训练。
点层已完成job605训练及test675；笔画层job679已完成epoch10完整串联val，后续效果以对应run产物为准。
**训练、测试与完整推理统一使用说明：[TWO_STAGE_USAGE.md](TWO_STAGE_USAGE.md)。**
计划见 [TODO](TODO.md)，点层设计见 [SINGLE_STROKE](SINGLE_STROKE.md)。
`summary.json`各项指标的定义、单位、计算口径及示例见 [SUMMARY_METRICS](SUMMARY_METRICS.md)。

## 当前状态（2026-10-28）

- job605 已按用户要求彻底取消，checkpoint 和训练、验证产物保留。
- 选用 `runs/single_stroke_full/job_605/train/best.pt`：epoch 20、step 120100。
- 完整验证集83,119笔：自由生成轨迹 IoU **0.839904**、坐标 MAE **1.702859 px**、宽度 MAE **1.160251 px**、端点 L2 **4.136157 px**。
- test675已完成：68,413笔，点层条件自由生成IoU 0.839153、XY MAE 1.706272px；这是给定真实mask与首点的独立点层结果。
- 当前输入为真实笔画 mask 覆盖规范化整字原图，区域外填白，并提供真实首点 `(x,y,width)`；后续点自由生成。这是单笔条件评估。
- 笔画层新增独立 Dataset、模型、损失、可恢复训练、自由推理评估和Slurm入口；不覆盖job237或job605。

## 笔画层：按笔顺预测实例

输入为规范化整字灰度图 `[B,1,256,256]`，取值 `[0,1]`、黑0白1。
训练标签沿用现有整字级划分：单笔PNG的 `<128` 区域作为前景mask，轨迹首点提供XY与宽度直径。
框由mask前景像素的最小/最大坐标生成，采用包含边界像素中心的 `xyxy`，除以255；不依赖笔画类型标签。

全量配置的整字编码为随机初始化 **Swin-T → C5 Transformer Encoder（2层）→ FPN**。
P5/P4/P3（8×8、16×16、32×32）加二维位置编码后拼接为视觉memory；P2（64×64）用于mask预测。
历史的每笔二值mask由共享ResNet18编码，取C3/C4/C5三个尺度，各用可学习CLS查询跨注意力汇聚，再融合成一个笔画向量。
这里的“共享”指所有历史笔画共用参数，整字与历史编码器各自独立。
历史编码器采用GroupNorm，避免将不同时间位置一起编码时，BatchNorm统计使未来mask泄漏到前面的输出。

笔画向量加序号embedding，与BOS组成因果Transformer Decoder的输入；全量配置9层，dim64、4头。
每个输出位置预测：

| 输出 | 表示与用途 |
| --- | --- |
| mask | 每笔独立sigmoid，允许交叉重叠；64×64 logits双线性上采样到256×256 |
| 框 | 预测中心及尺寸，再转为有序的归一化xyxy并限制在画布内；用于定位监督 |
| 有效性 | 二分类logit；低于0.5时整字停止，不是24类笔画type分类 |
| 起点XY | sigmoid，分别乘255还原像素坐标 |
| 首点宽度 | softplus+1e-6，乘256还原像素直径；独立预测、独立监督 |

不设终点预测头或终点辅助损失。训练有K笔时，使用 `BOS, M1, …, MK` 依次监督
`第1笔, 第2笔, …, 第K笔, invalid`；每个位置只能读取之前的mask。
末尾invalid仅监督有效性，后续padding不参与注意力或损失。推理从BOS开始，将预测mask按0.5二值化后反馈，缓存整字特征与历史向量。
无效候选不保存；允许预测零笔；正常invalid停止与64笔上限截断分别记录。达到上限后仍检查下一次有效性，不能把截断当正常停止。

首版损失为：

```text
L_layer = 4*BCE(validity) + 5*BCE(mask) + 5*Dice(mask)
        + 5*L1(box) + 2*(1-GIoU(box)) + 5*L1(start_xy)
        + 1*Charbonnier(start_width)
L = L_final + auxiliary_weight * mean(L_intermediate_layers)
```

mask损失在上采样的256画布上计算；Dice平滑常数1。宽度使用归一化直径，Charbonnier的epsilon为
`0.01/256`，减去epsilon消除零误差偏置。几何与mask仅作用于真实有效笔画；每项先在每字的有效笔画内平均，再对字平均。
有效性包括K个正例与一个invalid，不额外改变正负权重。所有decoder中间层复用预测头并接受相同监督，`auxiliary_weight=1`；宽度不混入XY损失。

全量入口首版采用AdamW、恒定lr1e-4、batch2、梯度裁剪1、默认关闭增强，200epoch上限；GPU容量与收敛尚待实测。
每10epoch在完整val上自由解码，以 `free.ordered_mask_iou` 最大值保存best，不自动运行test或早停。
smoke配置使用ResNet18整字编码、dim32、2层decoder和固定8字子集，解码上限8笔；仅用于接入检查，不等同全量结构或完整验证。

结构来源为[两阶段有序笔画论文](https://arxiv.org/html/2609.02251v1)。Swin-T/ResNet18具体型号、C5编码器、
CLS跨注意力聚合、GroupNorm、FPN与多尺度memory安排、损失平均方式和优化配置属于本地实现选择。
移除终点监督、增加首点宽度是当前书法方案的明确改动，不表述为官方复现。

## 点层：给定首点生成单笔轨迹

现有job605模型为 **随机初始化ResNet18 → C5 Transformer（2层）→ FPN → 4层点decoder**。
每层先做因果自注意力，再串行跨注意力读取P5→P4→P3→P2；四个尺度分别为8、16、32、64的方形网格。
输入单笔灰度图及给定首点 `(x,y,width)`，历史点通过三维嵌入加序号表示。
训练使用真实历史XYW，从第二点开始监督；推理反馈预测历史，输出后续XY、宽度和点有效性。
点有效性低于0.5结束当前笔；上限400点与正常停止分开统计，invalid候选不写入轨迹。

损失为 `L1(XY) + 0.2*BCE(validity) + 1*Charbonnier(width)`，在笔内平均再对笔平均。
给定首点不计几何损失，最后一个invalid仅计算有效性损失。详细训练、评估口径见 [SINGLE_STROKE](SINGLE_STROKE.md)。

## 两层之间的固定协议

```text
规范化整字图 I ──→ 笔画层 ──→ 有序的 (mask_k, start_xyw_k)
       │                          │
       └──── mask内保留原图、外部填白 ──→ 单笔灰度图 I_k
                                           │
                   start_xyw_k ────────────┴──→ 点层 ──→ 该笔XYW轨迹
```

| 协议项 | 固定约定 |
| --- | --- |
| 坐标系 | 同一LMDB `image_key` 的256×256规范化整字画布；不使用未配准的 `original_image_key` |
| 点层视觉输入 | `I_k = where(mask_k >= 0.5, I, 1.0)`；保留原图灰度，不直接输入黑白mask |
| 空间变换 | 不按框裁剪、不居中、不另行缩放；框不传入点层 |
| 几何条件 | 仅起点 `(x/255, y/255, width/256)`；宽度是像素直径，不是压力或半径 |
| 第一轨迹点 | 直接保留笔画层的预测首点，点层从第二点生成，不重新预测第一点 |
| 其他信息 | 不传终点、隐藏z或笔画类别；点层不读取真实长度或后续真实点 |
| 停止 | 笔画有效性控制整字停止，点有效性控制单笔停止；分别记录截断 |
| 训练关系 | 两层独立训练；当前阈值mask提取与自由推理不做端到端梯度回传 |

训练点层使用真实mask和真实首点，串联推理改用预测mask和预测首点，存在条件误差传播；必须另做串联验收。
`networks/two_stage.py` 已实现上述适配，直接调用现有点层 `generate()`，不改变job605结构和权重格式。
论文正文写明第二阶段使用单笔图，但未明确“mask覆盖原图”的像素构造；本表记录的是当前job605的实际协议。

## 统一操作入口

[TWO_STAGE_USAGE](TWO_STAGE_USAGE.md)集中维护环境/数据准备、两层训练与恢复、
独立点层诊断、两阶段完整val/test、单图推理、可选可视化和NPZ字段。
指标逐项解释见 [SUMMARY_METRICS](SUMMARY_METRICS.md)。

历史CPU真实资产接入检查不等同GPU训练后的效果；当前结果以对应run的评估为准。

## CASIA-OLHWDB外部评估

新增 `scripts/download_casia_olhwdb.py` 下载官方POT测试轨迹，
`scripts/evaluate_casia_olhwdb.py` 将轨迹渲染为256×256图片，仅以图片输入两阶段模型，
计算逐字前景IoU、精确DTW和全量2048维FID，支持预热排除、断点续跑及样例可视化。
下载命令、依赖与指标定义见 [CASIA_EVALUATION](CASIA_EVALUATION.md)。
这是当前权重的外部数据集评估，未对齐论文的划分/采样/渲染时不直接横比论文分数。

## 数据与输出

固定划分：[full_character_v1.json](data_splits/full_character_v1.json)，说明见 [data_splits/README](data_splits/README.md)。

| 集合 | 字图 | 笔画 |
| --- | ---: | ---: |
| train | 56,622 | 576,465 |
| val | 8,121 | 83,119 |
| test | 6,714 | 68,413 |

全部集合共用 `../data/chtr_mmah_lmdb/`；先划分整字，再展开单笔。val 用于选择模型，test 不参与调参或早停。
数据、checkpoint 和运行产物被 Git 忽略，拉取代码后须单独准备。

## 现有产物

job605点层权重位于 `runs/single_stroke_full/job_605/train/`，test675结果位于同run的 `test_job_675/`。
job679笔画层及串联评估位于 `runs/ordered_stroke_full/job_679/train/`；
完整val输出两层指标，图库仅整字HTML与NPZ，无逐笔文件。训练与查看命令统一见上述使用说明。

## 代码组织

```text
datasets/single_stroke.py             当前单笔数据与批处理
datasets/ordered_stroke.py            整字级有序笔画数据、终止槽与批处理
datasets/dataset_split.py             固定划分及完整性校验
networks/single_stroke.py             四尺度单笔轨迹模型
networks/ordered_stroke.py            整字/历史mask编码与有序笔画解码
networks/two_stage.py                 预测mask提取原图并传首点XYW的串联适配
losses/single_stroke.py               坐标、宽度、有效性损失
losses/ordered_stroke.py              mask、框、有效性、首点XY及独立宽度损失
training/single_stroke*.py            checkpoint与完整集合评估
training/ordered_stroke*.py           笔画层checkpoint、自由推理评估及可视化
scripts/*single_stroke*               训练、独立评估、审计及Slurm入口
scripts/*ordered_stroke*              笔画层训练、独立评估及Slurm入口
scripts/infer_two_stage.py            纯图片两阶段串联推理
tests/                               当前模型、固定划分及基线回归测试
```

回归检查命令见 [统一使用说明](TWO_STAGE_USAGE.md#checks)。
CPU合成数据检查与实际GPU容量、真实效果验收分别记录。

## 历史基线和清理范围

保留 job237 的模型、推理与评估代码、配置及运行资产，供历史复查；其结构见 [ARCHITECTURE](ARCHITECTURE.md)，本地评估见 [LOCAL_EPOCH100](../reproduction/LOCAL_EPOCH100.md)。
job237 曾见完整训练库，其32图评估是训练集诊断，不能与 job605 的独立验证分数直接比较。
free rollout、旧三任务分层 AR、扩散 / Flow Matching 已放弃；对应专用文档、代码、配置和测试已清理。共享基线模块和数据生成管线保留。
