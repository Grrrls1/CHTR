# 两阶段书法迁移：单笔轨迹阶段

当前已实现独立第二阶段及本机 CPU smoke 验证。有序笔画阶段尚未实现；本阶段评估给定库内单笔 PNG
与真实首点 `(x,y,width)`，不能当作完整 image-only 推理或训练收敛结论。

## 模型与数据协议

- `datasets/single_stroke.py`：加载固定字符分层清单，校验完整候选元数据指纹、ID、组与集合覆盖；
  按 `(sample_id, stroke_index)` 展开，同一张整字的全部笔画只在一个集合。
- 训练、验证和给定真实笔画的条件评估直接读取 `masks_key` NPZ 内的 `strokeXXX.png`，
  解码后仅除以255；不开增强时逐像素保留。不用它重新截取整字图，不阈值化重建输入，也不单独裁剪、居中。
  这些PNG来自清理后的笔画mask，以白底黑笔画形式导出，并非mask区域内的原图灰度截取。
- 完整推理时才用第一阶段预测mask提取规范化整字原图内容，区域外填白，保持256×256画布。
  `masked_stroke_image()`保留为此推理接口，阈值0.5；它不参与直接PNG训练。
  这条完整串联尚待第一阶段实现，需在P3检查预测mask及灰度输入差异的影响。
- 输入灰度 `[0,1]`；XY除以255，宽度直径除以256。有效点全部保留，超限报错。
  仅train允许随机缩放/平移，同步变换单笔PNG、XY和宽度；默认先关闭增强。
- 当前输入协议为 `stored_stroke_png_v2`，写入配置、数据使用记录、checkpoint及训练签名。
  旧版mask截原图输入的checkpoint拒绝续训/条件评估，不将旧结果冒充本版本结果；网络结构版本不变。
- 随机初始化 ResNet18 → 最粗 C5 上的 Transformer Encoder → FPN；
  点解码每块先因果自注意力，再依次跨注意力 P5→P4→P3→P2，共4块。
  256画布对应8×8、16×16、32×32、64×64视觉层，不做额外视觉token池化。
- 点历史包含真实XY和宽度，加入BOS与序号embedding。训练时输入 `p1…pL`，
  各位置预测 `p2…pL, invalid`；首点不重新预测，padding不参与注意力与损失。
  末尾invalid只算有效性损失，其XY和宽度不保存；单笔只有一点时也能学习invalid。
- 自由解码保留给定首点三元组，后续全部反馈预测历史；阈值0.5，区分正常invalid停止与点数上限截断。
  后续第一阶段须输出带宽度的 `u_k`；第二阶段接口不需要真实后续点或真实长度。

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

优化器Adam，默认lr=1e-4、余弦周期991000步。模板配置仅供启动，不代表容量或收敛参数已验收。
批量1、dim64、最大4096点；全库实测最长393点，没有静默截断。
smoke配置使用dim32、视觉encoder1层，但仍保留ResNet18和四尺度4层点解码。

## 本机运行

以下命令从仓库根目录执行。本机可用解释器为 `/home/lkh/anaconda3/envs/test/bin/python`，
Python3.8、torch1.13.0、torchvision0.14.0；CUDA不可用，本次未完成GPU容量验证。
依赖见 `requirements-single-stroke.txt`；torchvision须与torch版本匹配。
所有新run要求空目录，resume要求原目录且架构、数据、split和训练设置一致。
路径相对仓库根目录解析。权重文件仅应加载可信本地产物。

```bash
# 单元及端到端恢复/导出检查
/home/lkh/anaconda3/envs/test/bin/python -m unittest models.tests.test_dataset_split models.tests.test_single_stroke -v

# 实际数据短跑；若目录已存在，换一个output-dir
/home/lkh/anaconda3/envs/test/bin/python models/scripts/train_single_stroke.py \
  --config models/configs/single_stroke_smoke.json \
  --output-dir models/runs/single_stroke_smoke_new

# 在原目录恢复，延长到8步
/home/lkh/anaconda3/envs/test/bin/python models/scripts/train_single_stroke.py \
  --config models/configs/single_stroke_smoke.json \
  --output-dir models/runs/single_stroke_smoke_new \
  --resume models/runs/single_stroke_smoke_new/last.pt --max-steps 8

# 留出验证集的条件诊断，不使用test选模型
/home/lkh/anaconda3/envs/test/bin/python models/scripts/evaluate_single_stroke.py \
  --checkpoint models/runs/single_stroke_smoke_new/last.pt \
  --output models/runs/single_stroke_smoke_new/eval_val \
  --split val --limit 3 --decode-max-points 64

# 全库接口审核：解码所有图/单笔PNG和轨迹，不进行模型训练或测试集调参
/home/lkh/anaconda3/envs/test/bin/python models/scripts/audit_single_stroke_data.py \
  --dataset data/chtr_mmah_lmdb \
  --split-file models/data_splits/full_character_v1.json \
  --output models/runs/single_stroke_audit_new.json
```

训练输出：`config.json`、`data_usage.json`、`metrics.jsonl`、`last.pt`、按验证loss选择的`best.pt`、
`training_status.json`。保存优化器、调度器、epoch/cursor和随机状态；CPU测试覆盖启用增强与dropout的中途恢复，
结果与连续训练逐参数一致。未实现多进程加载、混合精度或GPU上的确定性承诺。

`train_char_limit` 从训练集合按笔画数覆盖选字；`train_stroke_limit` 从所选字中按轨迹长度覆盖取笔，
只用于显式小集诊断；为1时选中位长度，0为全部。实际使用key均保存，不修改split或截断轨迹。
`eval_strokes` 是验证子集大小，不是完整验证集指标。评估脚本的 `--split train --training-subset`
可只检查run实际使用的训练笔画。评估导出PNG、逐笔NPZ、宽度曲线HTML与JSON，明确标注库内单笔PNG和真实首点条件。

## 本次验证边界

当前PNG协议的短跑及像素校验见 [本机验证记录](reports/single_stroke_png_smoke_20261023.json)。
此前全量资产审核核对71,457张图与727,997笔、中心点均在对应mask内，资产及划分未改变；
该历史记录中的训练输入为旧协议，旧拟合指标不适用于当前PNG输入。
本次在train/val/test各64张、合计2,011笔上重新校验Dataset输入与存储PNG逐像素一致。
这验证接口与几何对应，不等同人工标签质量审核或近重复完备检测。
单笔拟合与短跑验证可训练性；固定32字的充分拟合、留出自由解码质量、GPU显存容量仍需后续验收。
job237资产与旧训练入口未修改；没有生成或提交Slurm。
