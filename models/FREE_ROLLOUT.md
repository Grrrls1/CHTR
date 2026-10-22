# 1915 数据集从头初始化的 free rollout 训练

统一入口：`scripts/train_free_rollout.py`；配置：`configs/free_rollout.json`。
保留 teacher-forced 代码作为原有实现；本说明描述 1915 全集上的独立 free rollout 训练。

## 默认设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | `../data/chtr_mmah_lmdb_1915` 的全部 1915 条样本 |
| 初始化 | 所有模型参数随机初始化，seed=42，ResNet-18 无预训练；默认不加载任何 checkpoint |
| 模型 | 完整 ResNet-18（layer1–layer4）+ FPN，dim=128，heads=4，笔画/逐点解码器各 2 层，dropout=0 |
| 参数量 | 13,162,979，约 1316 万；其中编码器 11,890,240 |
| 训练 | 最大 500 轮，batch=24，梯度累积=1，学习率 1e-5 |
| 早停 | 轮均训练损失，patience=10、min_delta=0.001、min_epochs=20，与 job237 相同 |
| 展开 | 最多 64 笔、每笔 1024 点，完整历史、完整 BPTT |
| 宽度 | 与 XY 一样使用像素 Charbonnier，权重 1 |
| 评估 | 随机初始化基线；每 20 轮、达到最大轮数或早停时评估并生成可视化 |
| 固定评估图 | 从训练库按笔画数分位抽取 32 张，仍全部参与训练 |
| 保存 | 每 200 步及轮末保存 last.pt，周期保存 epoch_020.pt 等 |

`train_limit=0`、`require_full_dataset=true` 确保使用所选 1915 库的全部样本，
不表示切回 73,606 条记录的完整库。`max_steps=0` 仅禁用步数上限，早停仍有效。
500 轮是上限，不保证一定执行到该轮数或已经收敛。固定图评估是训练图拟合检查，不代表泛化性能。
完整反传的显存随实际展开长度增长；没有实现截断 BPTT 或激活重计算。

## 服务器运行

Git 同步代码后，使用 `scripts/train_free_rollout.slurm` 申请单 GPU、4 CPU 和 32G 内存。
可以从仓库根目录或其工作区父目录提交，先创建相应的 `logs/` 目录。
保留调度器设置的 `CUDA_VISIBLE_DEVICES`，作业内使用 `cuda:0`。

解释器需要兼容的 PyTorch >= 2.0 / torchvision >= 0.15，以及 numpy、opencv-python、lmdb、matplotlib。
已有 CUDA 兼容的 PyTorch 环境可安装 `models/requirements-free-rollout.txt` 中的其他依赖；
PyTorch 的 CUDA 构建按服务器环境选择。
Git 不包含 LMDB，服务器需已具备 1915 库。默认从头训练不需要 job237 或其他旧权重。

在仓库根目录：

```bash
mkdir -p logs
gpu-sbatch --user_name lkh models/scripts/train_free_rollout.slurm
```

默认输出为 `models/runs/free_rollout_1915_resnet18_dim128_batch24/job_<作业号>/`。
提交入口不传 `--init-weights` 或 `--resume`，训练脚本也不再自动读取配置里的旧权重路径。
模型结构完全由配置中的 `model` 决定，dim=128 不从旧 checkpoint 的 dim=64 覆盖回来。
其他账户提交时替换 `--user_name`；可通过 `CHTR_PYTHON` 指定已有环境。

配置内 dataset/output_dir 的相对路径以 `models/` 为基准。
命令行 `--config` 相对当前工作目录解析；建议跨机器调用时显式使用绝对路径。
每个新作业使用独立输出目录；非空目录会被拒绝。

## 恢复训练

在新的 Slurm 分配中，沿用原配置、数据路径和原输出目录：

```bash
"$CHTR_PYTHON" -u models/scripts/train_free_rollout.py \
  --config models/configs/free_rollout.json \
  --dataset /absolute/path/to/chtr_mmah_lmdb_1915 \
  --output-dir /absolute/path/to/original-run \
  --resume /absolute/path/to/original-run/last.pt \
  --device cuda:0
```

恢复是明确传入 `--resume` 的操作，默认新运行不会自动执行。
`--init-weights` 仍保留为显式实验选项，两者互斥，且都要求 checkpoint 的模型结构与配置完全一致。
旧 CNN backbone 的 job333 和旧 dim=64 的 job237 均不能作为当前 ResNet-18 运行的精确恢复权重。
恢复要求数据路径、样本、模型及训练设置相同，允许改变 device 或增加 epochs/max_steps。batch_size 参与兼容签名，不能用旧 batch=1 断点直接恢复 batch=24。
保存包含优化器、样本游标、随机状态、轮内损失累计值、早停状态和兼容签名。
当前 free-rollout checkpoint 版本为 2，缺少早停状态的旧版断点不能直接精确恢复。
CUDA 默认不承诺逐位数值复现。
接收到 SIGUSR1/SIGTERM 时尝试在下一次优化器更新边界保存并退出。
作业若被强制终止，只能恢复到最近成功写入的断点。此入口不会自行提交或重排 Slurm 作业。

## 早停与权重选择

每个优化器步的总训练损失按实际处理图片数量累计，轮末得到样本加权均值。
比上次有效最佳值下降超过 0.001 才算改善，否则累计无改善轮数；
至少完成 20 轮后，连续 10 轮没有有效改善时停止。早停实现复用
`training/early_stopping.py`，规则及阈值与 job237 保持一致。
轮内中断会保存已累计损失和样本数，恢复后继续计算同一轮均值。
自由生成 IoU 用于选择 `best.pt`，不参与早停；最低有效训练损失另外保存为 `best_loss.pt`。

## 生成与梯度

- `networks/free_rollout.py::rollout()` 只接收图像和解码上限，完全不接收监督标签。
- 每笔从 BOS 开始，之后输入自己的坐标、直径及 argmax 事件 one-hot。
- 预测事件决定换笔、结束整字；上限是保护措施，不制造结束标签。
- 连续反馈和注意力缓存完整保留计算图，离散 argmax 与实际展开长度不可微。
- `separate_branches=true` 保留视觉条件与点分支的梯度隔离。
- 复用原有 encode_image/next_stroke/point_step；要求 dropout=0，保持训练和 greedy 推理公式一致。
- 训练使用 `rollout_batch()` 真正并行生成 24 张图；每张图独立结束，逐点压缩活跃样本及注意力缓存，保留完整梯度。
- 各图按笔画序号同步，短笔等待同批长笔；已结束的笔画不继续生成，整字结束后不参与后续笔画。
- `batch_size` 是一次并行前向的图片数；`gradient_accumulation` 是每次优化器更新的批次数，尾批按实际图片数归一化。
- 每轮 1915 张图约 80 次更新（79 批×24 张，末批 19 张）；step 表示优化器更新，不能与旧 batch=1 的 step 直接比较。
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
- `last.pt`：最近断点；`best.pt`：固定图片平均 free IoU 最高的权重，含随机初始化基线。
- `best_loss.pt`：轮均训练损失有效改善时保存的断点。
- `epoch_metrics.jsonl`：轮均训练损失、有效最佳损失和无改善计数。
- `epoch_020.pt` 等：周期及训练结束评估时的权重，最多到 `epoch_500.pt`。
- `eval_epoch_020/` 等：index.html、summary.json、teacher/free 轨迹、mask/type 图和预测 NPZ。
- `config.json`、`split.json`、`training_status.json`：实际设置、训练/评估 ID 和运行状态。

周期可视化复用 `scripts/evaluate_memorization.py`。截断轨迹显示时会临时补边界，
实际完成/截断状态保持原样。需要额外评估时可单独调用该脚本。

## 验证与限制

`tests/test_free_rollout.py` 检查贪心推理一致性、真实标签不进入生成、跨步反传、
分支隔离、缺失/额外笔画的状态梯度和有限损失。
`profile_free_rollout.py` 可用于独立检查实际样本的反传及显存，不更新权重。
该工具同样默认随机初始化，并从配置读取 dim=128；仅显式传 `--checkpoint` 才加载匹配权重。
`tests/test_free_rollout_training.py` 覆盖不读取旧权重、模型维度、训练损失早停及轮内恢复。

变长对齐和停止惩罚是首版近似；尚未实现可微重绘、mask 外惩罚或更好的语义笔画匹配。
最佳 free IoU 不保证宽度拟合也最佳，需要共同检查宽度曲线、笔画数和完成率。
独立数据集泛化和更大规模训练的实际改善仍需验证。

批量回归测试：`tests/test_free_rollout_batch.py` 检查不同结束时间、两种截断上限、batch=24、变长视觉槽位，以及批量/逐图损失与完整梯度的一致性。

## 2026-10-21 batch24 GPU 验证

以下是旧 CNN/FPN backbone 的历史测量，不代表新 ResNet-18 的显存容量。
RTX 5090，dim=128、FP32、完整 BPTT：Slurm 验证任务 326 通过。
24 张真实训练图片的三步随机初始化训练无 OOM，峰值约 1.17 GB（初期每图只生成 1 点，不能代表后期速度或显存）。
独立的受控容量测试强制生成指定长度，只用于验证完整反传和显存，不是训练效果评估：

| 每图生成长度 | 峰值 allocated 显存 |
| --- | --- |
| 4 笔 × 128 点 | 2694 MB |
| 1 笔 × 1024 点 | 26224 MB |

两项均得到有限梯度；后者已接近单卡容量，更多长笔组合仍可能超出显存，不能由此保证所有 64×1024 上限组合都可训练。
验证记录保存在本地 `runs/free_rollout_1915_dim128_batch24/validation_20261021/benchmark.json`，不提交训练产物到 Git。

## 2026-10-22 ResNet-18 Backbone 切换

默认 free rollout 改为 `backbone=resnet18`、`pretrained=false`，其余损失和三分类停止机制沿用原实现。
灰度输入复制成三通道并使用固定 ImageNet 均值 / 标准差变换；不下载预训练权重。
使用全部四个残差阶段并构建四级 FPN：256 像素输入对应 64/32/16/8 的空间网格，通道统一到128。
最深两个尺度池化为 8×8 和4×4，保持80个视觉 memory tokens；mask 分支仍从64×64特征上采样。
保留 ResNet 原生 BatchNorm：训练和 eval 分别使用批统计与运行统计，因此批量训练和单图评估不要求逐位一致。
CPU 上49项测试通过，包含最深层梯度和批量完整训练反传；ResNet 的 batch24 GPU 容量仍待实测。
后续需提交独立新作业才会使用该结构；修改配置不代表已有 CNN 作业已切换 Backbone。
