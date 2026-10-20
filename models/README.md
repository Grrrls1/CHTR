# 书法图像到书写轨迹模型

本目录实现 LMDB 数据适配、图像编码、有序笔画解码、逐点轨迹生成与多任务训练。
输入为单张灰度书法图像，输出按书写顺序排列的笔画轨迹。模型结构见
[ARCHITECTURE.md](ARCHITECTURE.md)，分支监督与停止规则见
[BRANCH_SEPARATION.md](BRANCH_SEPARATION.md)。

## 数据接口

数据来源为 `../data/chtr_mmah_lmdb/`，读取器位于 `datasets/`。
训练使用有完整合法轨迹的 ready 样本，pending 样本不参与轨迹训练。
每个 LMDB 目录必须包含匹配的 `stroke_type_map.json`，类别 ID 为 0–23。

| 字段 | 含义 |
| --- | --- |
| `image` | 灰度图像，默认 256×256 |
| `points[T,6]` | `[x,y,width,continue,stroke_end,drawing_end]` |
| `stroke_offsets[K+1]` | 语义笔画边界，左闭右开 |
| `subpath_offsets` | 数据中须等于 `stroke_offsets`，一笔对应一条连续路径 |
| `stroke_types[K]` | 24 类笔画监督 |
| `stroke_masks[K,H,W]` | 逐笔前景掩码 |

原始坐标和宽度直径采用像素单位；模型内部将 XY 除以 `size-1`、宽度除以 `size`。
PNG 掩码采用白底黑笔画，读取时以 `<128` 转成前景。
批处理使用有效笔画与有效点掩码，padding 不参与注意力或损失；超过序列上限的标签会报错。

训练加载器支持 `--min-trajectory-iou` 与 `--min-mask-iou`，两项按 AND 组合，默认不设阈值。
mask 默认使用 `mask_cleanup_whole`，也可用 `--mask-iou-stage mask_extraction_whole` 切换。
筛选不删除 LMDB 记录，质量指标不会作为自由生成的输入。

## 网络与训练方式

默认模型为无预训练 CNN/FPN，隐藏维度 64，笔画与逐点解码器各 2 层。
笔画模块预测 mask、类别、最后一笔及辅助起终点；逐点模块接收图像特征、笔画条件、
软 mask 注意力偏置和点历史，预测坐标、宽度与三类点事件。

训练采用 teacher forcing：点历史来自真实标签，`condition_mode=predicted` 使用预测的 mask 和类别。
`model.separate_branches=true` 时，视觉和逐点参数组分别接受各自监督并分别裁剪梯度。
默认几何损失采用像素尺度 Charbonnier，另含首点与笔内位移监督；其他损失包含宽度、
点状态、mask BCE/Dice、类别、最后一笔和辅助起终点。
可微重绘损失与自喂历史 rollout 训练尚未实现。

## 当前全量配置

配置文件：[configs/full_separated_10gb.json](configs/full_separated_10gb.json)。

| 参数 | 设置 |
| --- | --- |
| batch / workers | 48 / 2 |
| 学习率 / weight decay | 0.0003 / 0 |
| 数据增强 / dropout | 关闭 / 0 |
| 最大笔画数 / 每笔点数 | 64 / 1024 |
| 最大轮数 | 100 |
| checkpoint 间隔 | 每 100 步及轮末 |
| 早停 | patience=10，min_delta=0.001，min_epochs=20 |

配置中的相对 dataset 和 output_dir 路径以 `CHTR/models/` 为基准。
`minimal.json` 和 `gpu.json` 默认使用保留的 1915 数据集；全量配置使用完整数据集。

## 训练与恢复

在工作区根目录通过 Slurm 运行：

```bash
cd /home/lkh/workspace
mkdir -p logs
gpu-sbatch --user_name lkh chtr_train_10gb.slurm
```

其他账户提交时替换 `--user_name`。已分配 GPU 的作业中，也可使用模型入口：

```bash
bash CHTR/models/scripts/train_gpu.sh \
  --config /home/lkh/workspace/CHTR/models/configs/full_separated_10gb.json \
  --output-dir /absolute/path/to/new-run
```

解释器优先使用 `CHTR_PYTHON` 或 `models/.venv`，否则使用
`../third_party/StrokeExtraction/.venv/bin/python`。保留调度器分配的 `CUDA_VISIBLE_DEVICES`。

正式权重保留在：

- `runs/full_separated_10gb/job_237/best.pt`、`last.pt`。
- `runs/full_separated/job_231/last.pt`：此前停止的正式训练断点。

恢复同配置训练：

```bash
gpu-sbatch --user_name lkh chtr_train_10gb.slurm \
  --resume /home/lkh/workspace/CHTR/models/runs/full_separated_10gb/job_237/last.pt
```

`--resume` 恢复模型、优化器、随机状态和数据位置；`--init-weights` 只加载模型权重，创建新的训练状态。
两者互斥。`--epochs` 和 `--max-steps` 是累计上限，继续已满轮数的 checkpoint 时需相应提高上限。
数据、模型、采样及停止设置须满足恢复兼容检查；更改分支或损失配置应使用新训练目录。

训练输出包括最新 `last.pt`、有效改善时的 `best.pt`、逐步及逐轮指标、实际配置、类别表、
数据筛选统计和结束状态。这些正式训练记录与权重一起保留。

## 自由生成与评估入口

`networks/autoregressive.py` 提供 image-only greedy 解码，仅输入图像，不使用真实轨迹、
mask、类别、笔画数或点数。点事件控制换笔和整字结束；达到上限时标记 truncated，不伪造结束事件。

`scripts/evaluate_memorization.py` 提供训练图抽样的 teacher-forced / image-only 对比。
在 Slurm 分配的环境中传入可信 checkpoint 与新的 `--output` 目录运行；
抽样拟合指标不能视为泛化指标。它与 `scripts/` 中其他工具均为可重新执行的源码，历史评估结果已清理。

## 源码目录

```text
configs/         网络、数据和训练配置
datasets/        LMDB 适配、变长批处理和同步增强
networks/        编码器、笔画与逐点解码、自由生成
losses/          多任务与像素几何损失
training/        早停状态管理
scripts/         训练、评估、可视化和环境工具
visualization/   笔画类别与 mask 展示
tests/           模型与数据行为测试源码
```
