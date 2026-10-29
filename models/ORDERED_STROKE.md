# 笔画层探索、全量训练及两阶段评估

所有命令在 `CHTR` 根目录执行。点层固定使用job605的epoch20 `best.pt`，不参与笔画层优化。
训练期间val、独立val/test均执行：整字图 → 笔画层预测mask/首点 → 点层预测轨迹。
标签只用于teacher-forced辅助诊断、计算指标及GT展示；自由推理不读取真实mask、首点、长度或笔数。

## 配置

| 项目 | 探索 `configs/ordered_stroke_explore.json` | 全量 `configs/ordered_stroke.json` |
| --- | --- | --- |
| 结构 | Swin-T、dim64、9层decoder | 相同 |
| train / val字图 | 固定32 / 32字图，跨笔画数选择 | 56,622 / 8,121字图 |
| epoch上限 | 10 | 200 |
| 每隔几epoch验证 | 1 | 10 |
| batch / eval batch | 2 / 2 | 8 / 2 |
| 优化器 / 学习率 | AdamW / 恒定1e-4 | 相同 |
| 笔画解码上限 | 64 | 64 |
| 点层batch | 8 | 16 |
| 点层解码上限 | 400 | 400 |
| 两层有效性阈值 | 0.5 / 0.5 | 相同 |
| 可视化字图 | 8 | 30 |

`ordered_stroke_smoke.json`另保留轻量ResNet18/dim32/2层decoder配置，用于代码接入检查，
不代替上述正式结构的探索。其笔画解码上限8，模型位置容量64；推理按解码配置执行。

两个配置都包含以下可调整字段：

```json
{
  "point_checkpoint": "models/runs/single_stroke_full/job_605/train/best.pt",
  "point_batch_size": 16,
  "point_decode_max_points": 400,
  "point_validity_threshold": 0.5
}
```

探索时优先调整batch、history_chunk_size、学习率、train_char_limit和验证频率。
显存同时容纳笔画层与冻结点层；本次未实测正式结构的GPU训练容量。
数据、结构、loss、batch与点层设置参与恢复签名，改变配置应新建run；探索run不直接resume到全量配置。
修改JSON影响后续新训练；已保存checkpoint的独立评估默认读取checkpoint中的配置。

## 服务器手动启动

先探索正式结构的小集（命令会提交作业）：

```bash
cd /home/lkh/workspace/CHTR
gpu-sbatch --user_name lkh models/scripts/train_ordered_stroke.slurm models/configs/ordered_stroke_explore.json
```

手动启动全量训练：

```bash
cd /home/lkh/workspace/CHTR
gpu-sbatch --user_name lkh models/scripts/train_ordered_stroke.slurm models/configs/ordered_stroke.json
```

脚本申请1 GPU、4 CPU、32 GiB内存、3天时限；checkpoint为
`models/runs/ordered_stroke_full/job_<作业ID>/train/`，日志为
`/home/lkh/workspace/logs/chtr-ordered-stroke-<作业ID>.out`。
`last.pt`包含可恢复训练状态，`best.pt`仍按完整val的笔画层 `free.ordered_mask_iou` 选择；
点层指标同时报告，不改变选模标准。探索配置的val是部分诊断，会标记 `full_split=false`。
训练不自动使用test，test保持独立验收。评估开销包含两层推理，并非只计算笔画mask。

个人电脑可用已配置的Python运行同一训练入口，参考 [ENVIRONMENT](ENVIRONMENT.md)：

```bash
python -u models/scripts/train_ordered_stroke.py --config models/configs/ordered_stroke_explore.json --device cuda:0
```

## 独立val / test

已有笔画层checkpoint后，在个人电脑已配置的环境运行（把路径替换为实际checkpoint；输出目录须为空）：

```bash
python -u models/scripts/evaluate_ordered_stroke.py --checkpoint PATH_TO_ORDERED_BEST.pt --point-checkpoint models/runs/single_stroke_full/job_605/train/best.pt --output models/runs/ordered_eval_val --split val --device cuda:0
python -u models/scripts/evaluate_ordered_stroke.py --checkpoint PATH_TO_ORDERED_BEST.pt --point-checkpoint models/runs/single_stroke_full/job_605/train/best.pt --output models/runs/ordered_eval_test --split test --device cuda:0
```

服务器上通过Slurm运行相同入口，例如：

```bash
gpu-srun --user_name lkh --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1-00:00:00 /home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python -u models/scripts/evaluate_ordered_stroke.py --checkpoint PATH_TO_ORDERED_BEST.pt --output models/runs/ordered_eval_test --split test --device cuda:0
```

未指定 `--limit` 时评估完整集合，即使checkpoint来自32字探索run；`--limit 32`才是部分诊断。
`--batch-size`覆盖笔画层评估batch，`--point-batch-size`覆盖点层batch；后者不改变笔画层batch。
未提供或未配置点层checkpoint时明确报错，避免静默退回仅笔画层评估。

## 两层指标

全部字段、单位、平均方式和epoch10示例见 [SUMMARY_METRICS](SUMMARY_METRICS.md)。

`summary.json`中的：

- `stroke_layer.teacher_forced`：笔画层loss、mask IoU、有效笔画及终止召回率。
- `stroke_layer.free`：按序mask IoU、严格序列mask@0.5、笔数误差、首点XY/宽度误差、空mask、停止与截断。
- `point_layer`：完整串联后的整字轨迹IoU、按序轨迹IoU、已对齐笔画的XY/宽度/端点/点数误差及停止/截断率。
- `point_checkpoint`：实际使用的点层权重路径、epoch和step；另记录两层解码阈值和上限。

顶层 `teacher_forced` / `free`为笔画层兼容字段，与 `stroke_layer`对应内容一致。
逐字 `characters.jsonl`同时保存该字的两层指标与每笔点层停止原因。

整字轨迹IoU比较GT与预测所有笔画重绘的并集；按序轨迹IoU将漏笔/多笔补零，分母取两者笔数较大值。
`aligned_*`误差仅比较共同序号，漏笔后可能错配，必须同时查看missing_strokes/extra_strokes。
几何误差使用等点序进度重采样，包含预测首点误差，不是DTW。没有共同笔画时为null；没有预测笔画时停止率为null。
这些是串联指标，不能直接当成job605在真实mask/首点条件下的独立点层分数。

## 精简后的可视化与文件

每个选中字图仅新增一个HTML和一个NPZ，整图展示：

- 原图、GT/预测整字轨迹覆盖。
- GT/预测笔画mask覆盖。
- GT/预测整字走向、起点/终点/方向箭头、笔顺播放。

图片和SVG嵌入HTML，不生成逐笔图片、逐笔NPZ、逐笔方向页或单独SVG文件。
30字图库对应60个逐字文件，加根目录summary、JSONL和总览，共63个文件。
单笔独立评估也使用该精简格式。本次不重建或删除旧评估结果。

每字 `character_trajectory.npz`保存预测 `points`（像素XYW）、`stroke_offsets`、
GT轨迹与offsets、原图，以及预测mask、像素首点/框、置信度和逐笔停止原因。
数组拼接不表示笔间连线，渲染必须使用offsets保留抬笔边界；终止不等于终止位置正确。
