# 两阶段书法迁移：单笔轨迹阶段

当前已实现独立第二阶段；迁移前已完成 CPU smoke 验证。有序笔画阶段尚未实现；本阶段评估给定真实
mask 与首点 `(x,y,width)`，不能当作完整 image-only 推理或训练收敛结论。

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

2026-10-26 已使用 `third_party/StrokeExtraction/.venv/bin/python` 核验本机 LMDB，
与现有 v1 清单的候选元数据指纹一致；ID互斥、覆盖、重复组隔离、字符分布及展开笔画数均通过。
直接复用此清单，未重新抽样或拆分 LMDB。结果见
[本机核验记录](reports/local_dataset_split_20261026.json)。

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
全库实测最长393点，没有静默截断。batch96尚未做GPU容量实测，可按实际峰值调整。
smoke配置使用dim32、视觉encoder1层，但仍保留ResNet18和四尺度4层点解码。

## 本机运行

本机仓库为 `/home/lkh/workspace/CHTR`，解释器为
`/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python`（沿用 StrokeExtraction 环境）。
2026-10-26 实测：Python 3.10.19、torch 2.12.1+cu130、torchvision 0.27.1+cu130、
NumPy 1.23.5、Pillow 9.5.0、opencv-python-headless 3.4.18.65、lmdb 3.0.0。
工具沙箱内CUDA不可见；Slurm分配GPU后已确认RTX5090和CUDA13.0前向/反向可用。
系统默认 Python 位于 miniconda3，未安装训练依赖，请使用上述独立环境。
该环境已含本阶段所需依赖，无需另外安装整套 requirements 或升级现有 torch / OpenCV。
`requirements-single-stroke.txt` 是通用安装参考；本机现存 OpenCV 版本低于其声明的4.5，
是否可用以本机测试与数据接入核验为准。

以下命令从仓库根目录执行。计算作业遵循
[workspace Slurm skill](/home/lkh/workspace/slurm-jobs/SKILL.md)。用户要求手动提交，本次没有提交或取消作业。
Slurm脚本保留CUDA_VISIBLE_DEVICES，并设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 读取可信本地训练状态。

## 全量训练与提交

```bash
cd /home/lkh/workspace/CHTR
mkdir -p /home/lkh/workspace/logs
gpu-sbatch --user_name "$(id -un)" models/scripts/train_single_stroke_full.slurm
```

配置为 `models/configs/single_stroke.json`：batch96、lr3e-4、AdamW、weight_decay=0、
grad_clip=1、seed42、4线程、augment=false、dropout=0、最多200个epoch，无step上限。
保持ResNet18、2层视觉encoder、4层四尺度串行点decoder，width_weight=1、valid_weight=0.2。
每个epoch遍历576,465笔，batch96时为6,005步；每100步和每个epoch边界保存last.pt。
单卡、4 CPU、32G内存、3天时限。脚本先检查CUDA再直接训练，不进行GPU smoke。
输出为 `models/runs/single_stroke_full/job_<ID>/train/`，新run要求独立空目录。

旧job604的batch48及1000步/16笔验证配置已经被加载，不会随源码配置修改自动改变。
需要新run应用这次变更；batch和评估协议改变后，旧checkpoint不能直接resume到新配置。
是否停止旧job604由用户手动决定，先通过 `squeue -u "$(id -un)"` 检查状态，若仍在运行可执行 `scancel 604`。
job237和已有训练/评估产物保持保留。

## 每10个epoch的完整验证

`evaluation_protocol=full_split_free_v2`：仅在完整的第10、20、30…epoch结束后验证。
没有step中途验证，也不会因max_steps到达而补做部分epoch验证。
使用全部8,121张验证图片的83,119笔；`eval_batch_size=32` 独立于训练batch。
每笔只给真实mask提取的原图和真实首点XY/宽度，后续完全反馈预测点；解码上限400、有效性阈值0.5。
另外在同一完整验证集报告teacher-forced指标作为辅助诊断，JSON中两者分开命名。
批量自由解码对已停止样本移出后续计算，保持与单笔解码一致的停止语义。

自由解码报告：轨迹重绘IoU、等序列进度重采样后的XY/宽度MAE与宽度偏差、
末点欧氏距离、点数MAE、点数完全一致率、正常停止/上限截断数量与比例。
重绘IoU使用相同的XY/直径渲染规则处理真实与预测轨迹；不是第一阶段mask IoU或论文指标。
重采样仅用于评估不同长度的序列，不修改训练标签；几何误差不含给定首点，且不是DTW对齐。
TF辅助报告总loss、XY/宽度/validity损失、像素XY/宽度MAE、宽度偏差、全局有效点/终止位置召回率。
loss与几何指标先每笔平均再对全库平均；召回率按全库点级计数汇总。

按完整验证集的 `free_trajectory_iou` 最大值选择best.pt。
early_stopping仍为patience=10、min_delta=0.001、min_epochs=20，但patience现在计10次验证轮次，
即连续10次（约100个epoch）无超过0.001的IoU改善，且达到最少20个epoch时停止。
该监控项与job391使用的训练图自由解码IoU的数据集和模型条件不同，不能直接比较分数。

## 固定30字可视化

运行初始化时从val按笔画数跨度确定性选择30个不同字符的样本，冻结实际sample_id到
`visualization_selection.json` 和 `data_usage.json`，resume时校验不变。
每次完整验证仍评估全部笔画，只对固定30字的全部笔画保存图像和NPZ。

```text
train/
  config.json / data_usage.json / visualization_selection.json
  metrics.jsonl / last.pt / best.pt / training_status.json
  eval_total_epoch_010/
    summary.json      # 完整验证集汇总，含实际epoch、step
    strokes.jsonl     # 全部验证笔画的自由解码指标
    index.html        # 固定30字总览：原图/真实轨迹/预测轨迹
    <sample_id>/
      index.html     # 该字全部笔画及真实/预测宽度曲线
      original.png / gt.png / pred.png
      stroke000_input.png / stroke000_gt.png / stroke000_pred.png
      stroke000.npz  # 像素坐标及宽度的真实/预测序列
  eval_total_epoch_020/ ...
  test_best/
```

## 最终test与独立评估

训练正常达到epoch上限或early stopping后，自动加载val选出的best.pt，
在全部6,714张测试图片、68,413笔上做一次相同的完整评估，结果写入 `test_best/`。
测试集不用于调参、阈值、checkpoint选择或早停。作业被取消/超时，或仅因max_steps结束时不自动test。
若需要独立评估可信本地checkpoint，可手动通过Slurm提交，例如（替换JOB_ID）：

```bash
gpu-srun --user_name "$(id -un)" --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  /home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python models/scripts/evaluate_single_stroke.py \
  --checkpoint models/runs/single_stroke_full/job_JOB_ID/train/best.pt \
  --output models/runs/single_stroke_full/job_JOB_ID/eval_val_manual \
  --split val --device cuda:0 --batch-size 32 --visualize-characters 30
```

评估默认完整集合；显式 `--limit N` 才是部分诊断，summary标记full_split=false。
训练仍使用teacher forcing，单笔条件评估并不是完整image-only两阶段推理。

## 验证边界

本次使用CPU合成夹具验证16项测试：完整划分覆盖、只按epoch调度、固定字图库、批量/单笔
解码一致、混合停止长度、最终完整test、因果性、padding、loss梯度、早停及中途恢复逐参数一致。
没有启动GPU smoke或新训练；batch96显存容量与全库评估耗时仍待用户提交后确认。
历史全量资产审核见 [2026-10-23记录](reports/single_stroke_smoke_20261023.json)，
本机元数据指纹核验见 [2026-10-26记录](reports/local_dataset_split_20261026.json)。
