# 两阶段训练、测试与完整推理统一使用说明

更新：2026-10-29。本文件是当前模型的统一操作入口。
结构与loss见 [模型说明](README.md)、[点层设计](SINGLE_STROKE.md)；
硬件与Python版本见 [ENVIRONMENT](ENVIRONMENT.md)；指标逐项解释见 [SUMMARY_METRICS](SUMMARY_METRICS.md)。

导航：[环境与数据](#environment) · [点层训练](#point-training) · [点层测试](#point-test) · [笔画层训练](#stroke-training) · [全流程评估](#full-evaluation) · [完整推理](#inference)

## 1. 先选对入口

| 任务 | 入口 | 输入与行为 |
| --- | --- | --- |
| 训练点层 | `scripts/train_single_stroke.py` | 真实笔画mask提取的原图、真实历史XYW；只更新点层 |
| 点层独立val/test | `scripts/evaluate_single_stroke.py` | 真实mask、真实首点；后续点自由生成，属于条件诊断 |
| 训练笔画层 | `scripts/train_ordered_stroke.py` | 整字图、真实历史mask；只更新笔画层，点层提前加载并冻结 |
| 两阶段完整val/test | `scripts/evaluate_ordered_stroke.py` | 整字图→预测mask/首点→点层轨迹；同时报告两层指标 |
| 单图完整推理 | `scripts/infer_two_stage.py` | 仅图像及两层权重，不需要LMDB或GT；输出NPZ，可选HTML |

两层独立训练。通常先取得可用点层权重，再训练笔画层；当前直接复用job605点层。
笔画层训练期间的val也执行两阶段完整推理，不会只评估mask。

<a id="environment"></a>

## 2. 工作目录、环境与文件准备

以下相对路径和命令均以 **CHTR仓库根目录** 为工作目录。

服务器：

```bash
cd /home/lkh/workspace/CHTR
CHTR_RUN_PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
```

个人电脑：

```bash
cd /home/lkh/vscode/CHTR
CHTR_RUN_PY=/home/lkh/anaconda3/envs/test/bin/python
```

设置可信本地checkpoint加载与日志环境：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
```

后文直接Python命令供个人电脑或已获分配GPU的作业内部执行。
服务器新计算使用Slurm，不能用未分配GPU的进程检测结果判断设备能力。
Slurm脚本已配置解释器与环境变量；无需另外激活虚拟环境，也不手工覆盖 `CUDA_VISIBLE_DEVICES`。

训练／集合评估需要：

```text
CHTR/
  models/                                   模型代码
    data_splits/full_character_v1.json       固定整字划分
    runs/single_stroke_full/job_605/train/best.pt
    runs/ordered_stroke_full/job_679/train/best.pt
  data/chtr_mmah_lmdb/
    data.mdb                                约3.6 GiB，完整原始LMDB
    complete.json
    summary.json
    stroke_type_map.json
```

只做单图完整推理时只需代码、两层权重和输入图片，不需要上述LMDB与划分清单。
数据、权重和运行产物被Git忽略，复制仓库或git pull不会自动获得它们。
不要复制服务器虚拟环境到个人电脑；保留数据与checkpoint内约定的相对路径。
LMDB只读无锁加载，不需要复制运行中的 `lock.mdb`；即使部分诊断也会读取完整候选元数据校验指纹，不能只复制test记录。

在同一终端设置后续示例的权重：

```bash
POINT_CKPT=models/runs/single_stroke_full/job_605/train/best.pt
STROKE_CKPT=models/runs/ordered_stroke_full/job_679/train/best.pt
```

job605点层best为epoch20、step120100；笔画层使用你选定run的best.pt，上述job679仅为当前示例。
新训练尚未到第一次验证时可能没有best.pt，last.pt则按checkpoint间隔保存。

## 3. 配置与batch的单位

| 项目 | 点层全量 | 笔画层探索 | 笔画层全量 |
| --- | --- | --- | --- |
| 配置文件 | `configs/single_stroke.json` | `configs/ordered_stroke_explore.json` | `configs/ordered_stroke.json` |
| 结构 | ResNet18、dim64、4层点decoder | Swin-T、dim64、9层笔画decoder | 与探索相同 |
| train字图／笔画 | 56,622字／576,465笔 | 固定32字图 | 56,622字图 |
| val | 8,121字／83,119笔 | 固定32字图，标记部分诊断 | 完整8,121字图 |
| 训练batch | 96条笔画 | 2个整字 | **8个整字** |
| 验证batch | 32条笔画 | 2个整字 | **2个整字** |
| 串联点层batch | 不适用 | 8条笔画 | **16条笔画** |
| 优化器／学习率 | AdamW／恒定3e-4 | AdamW／恒定1e-4 | 相同 |
| epoch上限 | 200 | 10 | 200 |
| 每隔几epoch验证 | 10 | 1 | 10 |
| 可视化字图 | 30 | 8 | 30 |

新训练使用新的空输出目录。独立点层评估要求输出路径尚不存在；两阶段评估及纯推理允许空目录，但不覆盖非空目录。
所有val/test默认完整集合，探索run的训练内val除外；独立评估即使加载探索checkpoint，也默认评估完整集合。
`--limit N`才显式取部分诊断。固定划分详情见 [data_splits/README](data_splits/README.md)。

`batch_size`用于训练，`eval_batch_size`用于对应层的集合评估；
笔画层配置中的 `point_batch_size`只控制冻结点层每批处理多少条预测笔画。
例如笔画层2字共预测20笔、点层batch16，会分16笔和4笔处理，再归回原字。

全量笔画层每epoch为 `ceil(56622/8)=7078 step`；点层为 `ceil(576465/96)=6005 step`。
epoch用时应按实际字图／秒或笔画／秒估算；复杂字、共享GPU负载与完整串联验证都会改变耗时。
`ordered_stroke_smoke.json`是ResNet18/dim32/2层的轻量接入配置，不代替正式结构的探索。

当前两层有效性阈值0.5、mask阈值0.5；每字最多64笔、每笔最多400点。
预测mask保留原始概率，不做连通性清理，不按连通域拆成多条轨迹。
修改配置文件只影响后续新训练，已运行进程和已保存checkpoint不会自动同步修改。

<a id="point-training"></a>

## 4. 点层训练

服务器手动提交全量训练（会新建run，不会覆盖job605）：

```bash
gpu-sbatch --user_name lkh models/scripts/train_single_stroke_full.slurm
```

个人电脑先用轻量配置检查接入：

```bash
"$CHTR_RUN_PY" -u models/scripts/train_single_stroke.py \
  --config models/configs/single_stroke_smoke.json \
  --output-dir models/runs/point_smoke_new --device cuda:0
```

本机全量训练将配置改为 `models/configs/single_stroke.json`，输出换为新的空目录；
8 GiB个人电脑的batch应先按容量调整独立配置，不照搬服务器batch96。

点层按完整val的自由轨迹IoU选best，训练使用teacher forcing。
配置的最终test会在正常达到epoch上限或早停后加载best执行；取消、超时或max_steps提前结束不会自动补test。

<a id="point-test"></a>

## 5. 点层独立验证／测试（真实mask和首点条件）

个人电脑先用64笔检查环境，使用较小batch：

```bash
"$CHTR_RUN_PY" -u models/scripts/evaluate_single_stroke.py \
  --checkpoint "$POINT_CKPT" --output models/runs/point_test_check \
  --split test --limit 64 --batch-size 4 --visualize-characters 5 --device cuda:0
```

完整test：

```bash
"$CHTR_RUN_PY" -u models/scripts/evaluate_single_stroke.py \
  --checkpoint "$POINT_CKPT" --output models/runs/point_test_full \
  --split test --limit 0 --batch-size 4 --visualize-characters 30 \
  --threshold 0.5 --decode-max-points 400 --device cuda:0
```

完整val将 `--split test`改为 `--split val`，并换输出目录。
个人电脑batch4是起始值，显存不足降到2或1；服务器已有job605专用脚本使用batch32：

```bash
gpu-sbatch --user_name lkh models/scripts/test_single_stroke_job605.slurm
```

该专用脚本固定job605权重。其他点层checkpoint可用 `gpu-srun` 包装上面的Python命令。
独立点层入口启动打印加载阶段，每15秒输出心跳，逐批打印比例、速度和ETA；心跳不保证I/O已推进。

<a id="stroke-training"></a>

## 6. 笔画层训练：探索与全量

先探索正式结构的小集：

```bash
gpu-sbatch --user_name lkh models/scripts/train_ordered_stroke.slurm models/configs/ordered_stroke_explore.json
```

全量训练：

```bash
gpu-sbatch --user_name lkh models/scripts/train_ordered_stroke.slurm models/configs/ordered_stroke.json
```

服务器训练脚本均申请1 GPU、4 CPU、32 GiB内存、3天时限。
笔画层脚本的首个参数是配置文件路径；默认也是全量配置。

个人电脑的探索入口：

```bash
"$CHTR_RUN_PY" -u models/scripts/train_ordered_stroke.py \
  --config models/configs/ordered_stroke_explore.json \
  --output-dir models/runs/stroke_explore_new --device cuda:0
```

笔画层配置必须指定 `point_checkpoint`，当前为job605。
启动时加载点层一次并冻结，训练内val复用当前笔画层和该点层，不重复读取权重。
原图/mask/轨迹按批从LMDB读取，只有元数据、划分及模型常驻，不将全库图像一次性搬到GPU。

val始终执行预测mask／首点驱动的点层推理；best仍按笔画层 `free.ordered_mask_iou`选择。
点层指标同时记录，但点层参数不更新；笔画层训练结束不会自动使用test。
改变batch、结构、loss、数据或点层设置需新建run；不要把探索checkpoint直接resume到全量配置。

<a id="full-evaluation"></a>

## 7. 两阶段完整val／test

完整val：

```bash
"$CHTR_RUN_PY" -u models/scripts/evaluate_ordered_stroke.py \
  --checkpoint "$STROKE_CKPT" --point-checkpoint "$POINT_CKPT" \
  --output models/runs/two_stage_val --split val \
  --batch-size 2 --point-batch-size 16 --device cuda:0
```

完整test：

```bash
"$CHTR_RUN_PY" -u models/scripts/evaluate_ordered_stroke.py \
  --checkpoint "$STROKE_CKPT" --point-checkpoint "$POINT_CKPT" \
  --output models/runs/two_stage_test --split test \
  --batch-size 2 --point-batch-size 16 --device cuda:0
```

省略 `--point-checkpoint`时读取笔画层checkpoint配置中的点层路径。
没有点层配置时明确报错，不会静默退回mask-only评估。
`--limit 32`可做32字图诊断；默认limit0为完整集合。减少点层batch不会改变笔画层batch。
可视化数量读取笔画层checkpoint配置中的 `visualize_characters`。

服务器通过Slurm执行同一入口，例如：

```bash
gpu-srun --user_name lkh --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1-00:00:00 \
  "$CHTR_RUN_PY" -u models/scripts/evaluate_ordered_stroke.py \
  --checkpoint "$STROKE_CKPT" --point-checkpoint "$POINT_CKPT" \
  --output models/runs/two_stage_test --split test \
  --batch-size 2 --point-batch-size 16 --device cuda:0
```

输出中 `stroke_layer`报告笔画层TF及自由推理指标，`point_layer`报告预测条件下的串联轨迹指标。
顶层 `teacher_forced/free`是笔画层兼容别名；[全部指标定义](SUMMARY_METRICS.md)包含平均方式和局限。
首点误差及漏笔后的序号错配会影响点层几何分数；不能直接横比独立点层的真实条件分数。
test仅用于验收，阈值、配置及checkpoint选择使用val。

<a id="inference"></a>

## 8. 单图完整推理：图像 → NPZ，可选可视化

输入须已规范化到训练一致的 **256×256整字画布**；脚本转为灰度，但不会自动裁剪、缩放或居中。
推理不要求GT、LMDB、数据划分、笔画数或真实首点；加载两层checkpoint后自主生成。

```bash
"$CHTR_RUN_PY" -u models/scripts/infer_two_stage.py \
  --stroke-checkpoint "$STROKE_CKPT" --point-checkpoint "$POINT_CKPT" \
  --image /path/to/image_256.png --output models/runs/inference/example \
  --device cuda:0 --point-batch-size 16
```

默认输出 `prediction.npz`和 `summary.json`；需要整字可视化，在同一命令末尾加 `--visualize`。
它额外写 `index.html`，包含整字覆盖、走向、起终点、箭头及笔顺播放，不输出逐笔文件。
无GT时不计算精度指标；summary只记录权重、预测笔数和停止/截断原因。

服务器上，同样将该Python命令接在 `gpu-srun --user_name lkh --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G`之后。
没有CUDA的个人电脑可用 `--device cpu`；自由生成通常更慢。

## 9. 输出文件与NPZ字段

服务器训练目录：

```text
models/runs/single_stroke_full/job_<ID>/train/     点层
models/runs/ordered_stroke_full/job_<ID>/train/    笔画层
  config.json / data_usage.json
  metrics.jsonl / last.pt / best.pt
  eval_total_epoch_010/                          点层验证目录命名
  eval_epoch_010/                                笔画层完整串联验证目录命名
```

训练正常结束时还会写 `training_status.json`；运行中或被取消时可能不存在。
新评估图库每个选中字图只写一个HTML和一个整字NPZ，图片/SVG内嵌；
30字图库加根目录summary、JSONL、总览共63个文件，不再写逐笔图或逐笔NPZ。历史结果不自动重建。

| 文件／字段 | 含义 |
| --- | --- |
| 集合评估 `summary.json` | 全集指标、权重、阈值及评估范围 |
| `characters.jsonl` | 两阶段评估的逐字指标及逐笔点层停止原因 |
| `strokes.jsonl` | 独立点层评估的逐笔指标 |
| 图库 `character_trajectory.npz` | 整字GT和预测轨迹，以及可视化所需原图和中间产物 |
| 推理 `prediction.npz` | 纯预测数据，无GT |
| 两种NPZ的 `points[N,3]` | 像素XYW；width为直径，不是深度、压力或半径 |
| `stroke_offsets[K+1]` | 预测笔画左闭右开边界；第k笔为 `points[offsets[k]:offsets[k+1]]` |
| 推理 `masks[K,256,256]` | 笔画层原始前景概率，0～1，尚未清理；点层使用≥0.5区域提取原图 |
| 图库 `predicted_masks` | 与推理masks同义，仅完整串联评估保存；独立点层图库没有此项 |
| 推理 `starts`／图库 `predicted_starts_xyw_px` | 像素首点XY及像素宽度直径 |
| 推理 `boxes`／图库 `predicted_boxes_xyxy_px` | 像素xyxy框，框不用于裁剪点层图像 |
| `image_size` | 256 |
| 图库 `ground_truth_xyw_px`／`ground_truth_stroke_offsets` | GT整字轨迹及分笔边界 |
| 图库 `complete_character` | 输入是否覆盖完整字；不表示预测笔数或轨迹正确 |

拼接点数组不意味着笔间画连线。读取示例：

```python
import numpy as np
with np.load('prediction.npz', allow_pickle=False) as data:
    points = data['points']
    offsets = data['stroke_offsets']
    strokes = [points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
```

一个预测mask即使有多个连通域，也只对应一条点层轨迹；不在单笔内部自动抬笔。

## 10. 恢复训练与监控

仅在原训练进程已停止时恢复同一个run。先将下面变量替换成实际run目录：

```bash
POINT_RUN=PATH_TO_POINT_RUN
STROKE_RUN=PATH_TO_ORDERED_STROKE_RUN
```

点层恢复：

```bash
"$CHTR_RUN_PY" -u models/scripts/train_single_stroke.py \
  --config "$POINT_RUN/config.json" --output-dir "$POINT_RUN" \
  --resume "$POINT_RUN/last.pt" --device cuda:0
```

笔画层恢复：

```bash
"$CHTR_RUN_PY" -u models/scripts/train_ordered_stroke.py \
  --config "$STROKE_RUN/config.json" --output-dir "$STROKE_RUN" \
  --resume "$STROKE_RUN/last.pt" --device cuda:0
```

服务器恢复也需 `gpu-srun`分配资源。已有sbatch脚本默认按新作业ID创建run，不直接用于上述同目录恢复。
`--epochs`、`--max-steps`是累计上限，不是额外增加数量；恢复不会自动将上限翻倍。
保持模型、数据、batch、loss及点层配置一致；不要改签名绕过恢复检查。

服务器查看状态与日志：

```bash
squeue -u lkh
tail -f /home/lkh/workspace/logs/chtr-ordered-stroke-679.out
```

点层训练日志名 `chtr-stroke-full-<ID>.out`，笔画层为 `chtr-ordered-stroke-<ID>.out`；
将679替换为实际作业ID。`.err`中看异常堆栈，不能仅凭`.out`暂时为空认定失败。
独立评估不覆盖非空输出目录；重复运行换目录名，不删除已有报告来绕过检查。

## 11. 常见区别与问题

- **纯推理与评估**：纯推理仅图像→预测，无GT精度；集合评估需要LMDB与GT来算指标。
- **训练与验证加载**：笔画层训练启动已加载冻结点层；训练中val复用权重，独立val/test是新进程，会重新加载。
- **mask覆盖图与轨迹覆盖图**：前者是区域预测，后者按预测XYW重绘；形状接近不代表笔顺和方向正确。
- **阈值、mask碎片**：当前没有形态学或连通域清理；不要仅按最大连通域假定其他部分无效。
- **显存不足**：按阶段调整训练batch、eval_batch_size或point_batch_size；显存瞬时占用不等于复杂多笔字的峰值。
- **加载缓慢**：完整元数据校验在初始化阶段进行；优先本地SSD，检查I/O等待与路径，不通过跳过指纹校验解决。
- **数据不匹配**：检查原始完整LMDB、固定划分及checkpoint；不要重新划分或修改checkpoint配置冒充兼容。

本次只统一文档，不提交新训练、评估或推理任务，不重建历史图库。

<a id="checks"></a>

## 12. 代码回归检查

使用CPU合成数据，不读取完整训练库，不代表真实效果或GPU容量验收：

```bash
"$CHTR_RUN_PY" -m unittest discover -s models/tests -v
```
