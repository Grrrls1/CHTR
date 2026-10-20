# 全库 free rollout 训练

统一入口：`scripts/train_free_rollout.py`；配置：`configs/free_rollout.json`。
保留 teacher-forced 代码作为原有实现；本说明只描述 free rollout 全库训练。

## 默认设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | `../data/chtr_mmah_lmdb` 的全部 ready 样本，pending 不参与 |
| 初始化权重 | `runs/full_separated_10gb/job_237/last.pt` |
| 训练 | 100 轮，batch=1，梯度累积=1，学习率 1e-5 |
| 展开 | 最多 64 笔、每笔 1024 点，完整历史、完整 BPTT |
| 宽度 | 与 XY 一样使用像素 Charbonnier，权重 1 |
| 评估 | 初始数值基线；第 20/40/60/80/100 轮评估并生成可视化 |
| 固定评估图 | 从训练库按笔画数分位抽取 32 张，仍全部参与训练 |
| 保存 | 每 200 步及轮末保存 last.pt，周期保存 epoch_020.pt 等 |

`train_limit=0`、`require_full_dataset=true` 确保使用全部 ready 样本。
`max_steps=0` 表示按轮数运行，无早停。固定图评估是训练图拟合检查，不代表泛化性能。
完整反传的显存随实际展开长度增长；没有实现截断 BPTT 或激活重计算。

## 服务器运行

Git 同步代码后，在已有 Slurm 分配的单 GPU 作业中执行下面的命令。
不提供或生成 Slurm 文件；沿用你自己的资源申请与日志设置。
保留调度器设置的 `CUDA_VISIBLE_DEVICES`，作业内使用 `cuda:0`。

解释器需要 PyTorch >= 2.0、numpy、opencv-python、lmdb、matplotlib。
已有 CUDA 兼容的 PyTorch 环境可安装 `models/requirements-free-rollout.txt` 中的其他依赖；
PyTorch 的 CUDA 构建按服务器环境选择。
Git 不包含 LMDB 与训练权重，服务器需已具备上述资产，或通过参数指定实际路径。

在仓库根目录：

```bash
# CHTR_PYTHON 使用你在服务器准备的 Python 解释器。
"$CHTR_PYTHON" -u models/scripts/train_free_rollout.py \
  --config models/configs/free_rollout.json \
  --dataset /absolute/path/to/chtr_mmah_lmdb \
  --init-weights /absolute/path/to/job_237/last.pt \
  --output-dir "/absolute/path/to/runs/free_rollout_full/job_${SLURM_JOB_ID}" \
  --device cuda:0
```

配置内 dataset/initial_weights/output_dir 的相对路径以 `models/` 为基准。
命令行 `--config` 相对当前工作目录解析；建议跨机器调用时显式使用绝对路径。
每个新作业使用独立输出目录；非空目录会被拒绝。

## 恢复训练

在新的 Slurm 分配中，沿用原配置、数据路径和原输出目录：

```bash
"$CHTR_PYTHON" -u models/scripts/train_free_rollout.py \
  --config models/configs/free_rollout.json \
  --dataset /absolute/path/to/chtr_mmah_lmdb \
  --output-dir /absolute/path/to/original-run \
  --resume /absolute/path/to/original-run/last.pt \
  --device cuda:0
```

`--init-weights` 仅加载权重并创建新优化器，`--resume` 恢复完整训练状态，两者互斥。
恢复要求数据路径、样本、模型及训练设置相同，允许改变 device 或增加 epochs/max_steps。
保存包含优化器、样本游标、随机状态和兼容签名；CUDA 默认不承诺逐位数值复现。
接收到 SIGUSR1/SIGTERM 时尝试在下一次优化器更新边界保存并退出。
作业若被强制终止，只能恢复到最近成功写入的断点。此入口不会自行提交或重排 Slurm 作业。

## 生成与梯度

- `networks/free_rollout.py::rollout()` 只接收图像和解码上限，完全不接收监督标签。
- 每笔从 BOS 开始，之后输入自己的坐标、直径及 argmax 事件 one-hot。
- 预测事件决定换笔、结束整字；上限是保护措施，不制造结束标签。
- 连续反馈和注意力缓存完整保留计算图，离散 argmax 与实际展开长度不可微。
- `separate_branches=true` 保留视觉条件与点分支的梯度隔离。
- 复用原有 encode_image/next_stroke/point_step；要求 dropout=0，保持训练和 greedy 推理公式一致。
- 单图顺序处理，无 batch padding 或已结束样本继续展开的问题。
- `last_head` 和 `endpoint_head` 仍仅作辅助监督，尚未更改实际停止策略。

## 首版损失及局限

标签只进入损失和固定样本划分，不进入轨迹生成。

1. 按笔画序号配对，对各笔预测和标签分别按均匀点索引重采样到 64 点，
   计算像素 Charbonnier XY、宽度、起点、局部位移损失，逐笔等权。
   宽度乘 image_size 还原为直径，XY 乘 image_size-1。
   这不是弧长对齐；短笔和长笔的监督不依赖完全相同点数。
2. 点事件按原标签的点时间索引监督：真实终点之前继续，之后收笔或结束。
   即使轨迹被重采样，事件目标也不随模型自己的终止位置重采样。
3. 缺失笔画也计入几何目标：使用最后预测点作为退化路径，与全部缺失标签比较。
   终止 logits 同时接受继续/换笔惩罚，产生可训练的梯度；不是只加不可微笔数差。
4. 多余笔画对真实最后一笔的终止 logits 增加结束监督。
5. 长度项用已访问 logits 的继续概率累积乘积构造生存长度代理，并延伸到配置上限。
   该代理对过早停止提供梯度，但不等于未访问 decoder 状态上的真实期望长度。
6. 单独预测全部真实视觉槽位，计算 mask BCE/Dice、类别、last、辅助起终点损失。
   监督槽位数不改变轨迹 rollout。这一额外视觉前向可以覆盖预测提前结束后的视觉监督。

以上惩罚能为缺失/额外笔画提供梯度，但不保证模型无法找到新的投机解；
小集拟合、终止率与图像 IoU 必须共同验证。没有实现可微重绘、mask 外惩罚、
soft-DTW 或语义笔画匹配；发生漏笔后按序号配对也可能错配，这是首版的重要限制。

## 输出与评估

- `metrics.jsonl`：每步总损失、分项损失、梯度、展开长度、笔数、耗时和显存。
- `validation.jsonl`：初始及每 20 轮固定训练图的 free IoU 和完成/截断结果。
  文件名沿用 validation，但这些图片实际来自训练集。
- `last.pt`：最近断点；`best.pt`：固定图片平均 free IoU 最高的权重，含训练前基线。
- `epoch_020.pt` 至 `epoch_100.pt`：周期权重。
- `eval_epoch_020/` 等：index.html、summary.json、teacher/free 轨迹、mask/type 图和预测 NPZ。
- `config.json`、`split.json`、`training_status.json`：实际设置、训练/评估 ID 和运行状态。

周期可视化复用 `scripts/evaluate_memorization.py`。截断轨迹显示时会临时补边界，
实际完成/截断状态保持原样。需要额外评估时可单独调用该脚本。

## 验证与限制

`tests/test_free_rollout.py` 检查贪心推理一致性、真实标签不进入生成、跨步反传、
分支隔离、缺失/额外笔画的状态梯度和有限损失。
`profile_free_rollout.py` 可用于独立检查实际样本的反传及显存，不更新权重。

变长对齐和停止惩罚是首版近似；尚未实现可微重绘、mask 外惩罚或更好的语义笔画匹配。
最佳 free IoU 不保证宽度拟合也最佳，需要共同检查宽度曲线、笔画数和完成率。
独立数据集泛化和更大规模训练的实际改善仍需验证。
