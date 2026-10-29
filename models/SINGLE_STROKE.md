# 两阶段书法迁移：单笔轨迹阶段

当前已实现独立第二阶段，完成CPU smoke、全量资产接入审核及服务器job605的完整训练/验证运行。
job605已取消，保留epoch20的best.pt；完整val轨迹IoU为0.839904。test675已完成全部68,413笔，条件自由生成IoU为0.839153；历史30字图库保留。
有序笔画层与串联推理已实现，job679已完成epoch10完整串联val；两层设计与协议见 [README](README.md)。
本阶段评估给定真实mask与首点 `(x,y,width)`，不是完整image-only推理。

## 模型与数据协议

- `datasets/single_stroke.py`：加载固定字符分层清单，校验完整候选元数据指纹、ID、组与集合覆盖；
  按 `(sample_id, stroke_index)` 展开，同一张整字的全部笔画只在一个集合。
- 使用 LMDB `image_key` 中规范化到256×256的原图灰度内容，以单笔 PNG 的 `<128` 区域为 mask，
  mask 外填255，保留整字坐标。原始 JPG 不再重新裁剪、居中或映射。
  已有 PNG 仅用于提取 mask，不直接把二值笔画图当视觉输入。
  `masked_stroke_image()` 也可接收第一阶段未来提供的预测概率 mask，阈值0.5。
- 输入灰度 `[0,1]`；XY除以255，宽度直径除以256。有效点全部保留，超限报错。
  仅 train 允许随机缩放/平移，同时变换图像、mask、XY及宽度；默认先关闭增强。
- 随机初始化 ResNet18 → 最粗 C5 上的 Transformer Encoder → FPN；
  点解码每块先因果自注意力，再依次跨注意力 P5→P4→P3→P2，共4块。
  256画布对应8×8、16×16、32×32、64×64视觉层，不做额外视觉token池化。
- 点历史包含真实XY和宽度，加入BOS与序号embedding。训练时输入 `p1…pL`，
  各位置预测 `p2…pL, invalid`；首点不重新预测，padding不参与注意力与损失。
  末尾invalid只算有效性损失，其XY和宽度不保存；单笔只有一点时也能学习invalid。
- 自由解码保留给定首点三元组，后续全部反馈预测历史；阈值0.5，区分正常invalid停止与点数上限截断。
  后续第一阶段须输出带宽度的 `u_k`；第二阶段接口不需要真实后续点或真实长度。

## 固定 train / val / test 划分

仓库已有 `models/data_splits/full_character_v1.json`，训练与 smoke 配置均读取该文件。
先按整字图片划分，再在各集合内展开单笔，所有集合共用同一份 LMDB；
不存在三份数据目录不代表未划分。核验命令及重建规则见 [划分说明](data_splits/README.md)。
清单以全部 ready 样本为候选，无额外 IoU 筛选，seed=42，目标比例80% / 10% / 10%；
字符分层、来源及重复组约束决定实际数量。验证集用于模型选择，测试集保留到最终验收。

2026-10-26 已使用 `third_party/StrokeExtraction/.venv/bin/python` 核验服务器 LMDB，
与现有 v1 清单的候选元数据指纹一致；ID互斥、覆盖、重复组隔离、字符分布及展开笔画数均通过。
直接复用此清单，未重新抽样或拆分LMDB。

| 集合 | 整字图片数 | 展开笔画数 |
| --- | ---: | ---: |
| train | 56,622 | 576,465 |
| val | 8,121 | 83,119 |
| test | 6,714 | 68,413 |

## 实现选择与书法扩展

论文未写明的 ResNet 型号、视觉Transformer放置位置、encoder层数、归一化与FFN细节，
这里分别选择 ResNet18、C5、默认2层、pre-norm与每串行块一个FFN；这些不是已核验的官方实现细节。
核心四尺度串行4层、teacher forcing、首点初始化与有效性停止保持论文方法。

宽度头及历史宽度属于书法扩展。XY为sigmoid；宽度为softplus+1e-6，保证正值。
损失按每笔先取均值，再对笔平均：

`L = L1(XY) + 0.2 * BCE(validity) + 1.0 * Charbonnier(width)`

XY与宽度只监督第二点之后的有效点；宽度误差采用归一化单位，Charbonnier的epsilon为
`0.01/256`（0.01像素），减去epsilon消除零误差偏置。有效性覆盖后续有效点及一个终止invalid位置，
不额外改变正负样本权重。每笔没有后续有效点时，几何损失为零。
另报像素XY MAE、宽度MAE/偏差、有效点召回与终止位置召回。
宽度损失权重属于首版实现选择，尚未用完整训练确定最优值；不得据此改写job237历史。

全量配置使用AdamW、lr=3e-4恒定学习率、batch96、dim64、最大400点；
全库实测最长393点，没有静默截断。job605已使用batch96完成至少20轮训练；个人电脑容量另行实测。
smoke配置使用dim32、视觉encoder1层，但仍保留ResNet18和四尺度4层点解码。

## 操作入口与验证边界

点层训练、恢复、条件val/test，以及两阶段完整训练/测试/推理的全部命令统一见
[TWO_STAGE_USAGE](TWO_STAGE_USAGE.md)。本文件仅保留点层结构、loss、数据与历史验证依据。
机器环境见 [ENVIRONMENT](ENVIRONMENT.md)，两阶段指标定义见 [SUMMARY_METRICS](SUMMARY_METRICS.md)。

历史CPU合成夹具覆盖因果性、padding、停止/截断、保存恢复及导出。
job605已完成至少20轮batch96训练及完整val，test675完成68,413笔条件评估。
真实mask/首点条件下的点层结果，不能直接代表预测mask/首点驱动的完整串联效果。
