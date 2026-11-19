# CASIA 点层迁移到书法

准备日期：2026-11-13。只准备配置和提交入口；正式 GPU 训练由以下命令启动。

```bash
cd /home/lkh/workspace/CHTR
gpu-sbatch --user_name lkh models/scripts/finetune_calligraphy_point.slurm
```

配置为 `models/configs/calligraphy_single_stroke_finetune.json`。初始化来自已冻结的
`models/runs/casia_olhwdb11/frozen_points/job1136_epoch004_step1120224.pt`，
即 CASIA 点层 epoch 4、step 1120224，SHA256 为
`e0cd0b92536b4213073f1886f9553534244bba0d7e83be14d3baad34bbae7043`。
只继承模型权重，优化器、scheduler、step、epoch、数据 cursor、早停和 best 均重新开始。
微调默认更新全部点层参数，不冻结骨干，不继承 CASIA 的优化器动量。

数据采用 `data/chtr_mmah_lmdb` 和现有 `models/data_splits/full_character_v1.json`，
不重划分、不截断、不筛除长笔画，不修改原始图像、XYW、笔画边界和顺序。
原划分记录为 train 56,622 字 / 576,465 笔，val 8,121 字 / 83,119 笔，
test 6,714 字 / 68,413 笔；本次训练只使用 train，每轮评估使用完整 val。
训练输入为真实笔画掩膜覆盖的原图和真实历史 XYW；验证给定真实掩膜及真实首点，后续自由生成。
这一步仅微调第二阶段点层，不训练第一阶段，也不表示完整两阶段端到端成绩。

## 与 CASIA 当前配置对齐

以 `models/configs/casia_single_stroke.json` 为基准，使用当前设置，而非 job1136 原始 batch16：

| 设置 | 微调配置 |
|---|---|
| 训练 / 验证 batch | 256 笔 / 128 笔 |
| 优化器、学习率、weight decay | AdamW、3e-4、0 |
| scheduler、梯度裁剪 | 恒定学习率、1.0 |
| seed、增强 | 42、关闭 |
| width / validity loss 权重 | 1.0 / 0.2 |
| 训练上限、早停 | 200 epochs；至少20轮；patience10；min_delta0.001 |
| checkpoint / 验证 | 每100步保存；每轮全量 val |
| 验证协议 | `full_split_free_v2`，真实掩膜和真实首点，逐笔等权 |
| 指标及选优 | AIoU / DTW / LDTW 及原有 XYW 指标；以 val AIoU 选 best |
| 指标实现 | `pen_net_equations_local256_v1`，未声明官方代码等价 |
| 可视化、阈值 | 固定30个字、0.5 |
| 自动 final test | 关闭，与 CASIA 当前配置一致；test 不用于选优 |

数据相关的必要差异：使用书法原始宽度标签，不沿用 CASIA 的合成随机渲染宽度。
用户已确认允许将模型容量和解码上限从256点扩为400点，以容纳书法原始轨迹。
400点沿用已完成的书法全量点层训练配置；数据加载器仍逐笔检查，超限时报错而不截断。
位置嵌入复制原有257行（含额外槽），新增144行保留新模型的默认随机初始化，其余参数严格加载。
除数据、输出、初始化元数据及容量/解码上限外，其余配置保持 CASIA 当前值。

## 产物与恢复

结果位于 `models/runs/casia_to_calligraphy/single_stroke/job_<作业ID>/train/`，
包含 `fine_tuning.json`（父权重SHA256与扩容记录）、`config.json`、`data_usage.json`、
`last.pt`、`best.pt`、`metrics.jsonl`、每轮 `eval_total_epoch_XXX/summary.json`、逐笔明细与画廊。
Slurm 日志位于 `/home/lkh/workspace/logs/chtr-calli-point-ft-<作业ID>.out` 和 `.err`。

同一微调 run 的中断恢复仍使用原有 `--resume`，必须使用同一配置与输出目录，并通过 Slurm 分配GPU：

```bash
# 在已分配GPU的Slurm作业中执行，将<ID>替换为原微调作业ID：
third_party/StrokeExtraction/.venv/bin/python models/scripts/train_single_stroke.py \
  --config models/runs/casia_to_calligraphy/single_stroke/job_<ID>/train/config.json \
  --output-dir models/runs/casia_to_calligraphy/single_stroke/job_<ID>/train \
  --resume models/runs/casia_to_calligraphy/single_stroke/job_<ID>/train/last.pt --device cuda:0
```

父权重已在 CPU 上实际加载并完成 256→400 点位置嵌入迁移，SHA256 核验通过。
`models/tests/test_point_finetuning.py` 的3项CPU测试通过，覆盖权重与新增位置保留、
SHA256/非法架构变更拒绝，以及合成书法LMDB上的单步训练、全量小验证集AIoU选优和checkpoint保存。
Slurm脚本通过 `bash -n`，配置与Python语法检查通过；未执行正式书法全量训练或GPU容量测试。
GPU 容量和实际训练效果需要提交后核验；当前没有 batch256 的容量实测结论。


## 2026-11-19：点层结果与笔画层微调

点层 `job_3215` 已在 epoch23 / step51796 正常早停，连续10次验证未达到早停改善条件。
按 val AIoU 选出的 best 为 epoch13 / step29276，不使用最新的 last.pt 替代 best。

| 指标（完整 val，83,119笔，GT mask/首点） | epoch13 best | epoch23 last |
|---|---:|---:|
| AIoU | 0.726612 | 0.725402 |
| 轨迹 IoU | 0.904611 | 0.896873 |
| XY MAE（px） | 2.619541 | 2.452760 |
| 宽度 MAE（px） | 1.542042 | 1.506022 |
| 终点 L2（px） | 5.940496 | 5.623285 |
| LDTW（px） | 1.561414 | 1.475799 |
| DTW | 179.701747 | 130.111627 |
| 截断笔数 | 125 | 38 |

不同指标的最佳轮不同；下一阶段统一冻结 AIoU 选定的 epoch13，不拼接不同权重的成绩。
上述是点层诊断，不是纯图像完整两阶段成绩。尚未做本轮权重的书法 test 评估。
来源：[best轮 summary](runs/casia_to_calligraphy/single_stroke/job_3215/train/eval_total_epoch_013/summary.json)、
[last轮 summary](runs/casia_to_calligraphy/single_stroke/job_3215/train/eval_total_epoch_023/summary.json)、
[结束状态](runs/casia_to_calligraphy/single_stroke/job_3215/train/training_status.json)。

笔画层微调配置：`models/configs/calligraphy_ordered_stroke_finetune.json`。
从 CASIA `job1597` 的 epoch6 / step151632 笔画层 best 初始化，仅继承模型权重；
笔画位置容量由32扩为64，复制原33行位置嵌入，新增32行按模型默认方式初始化。
冻结书法 `job3215` 的 epoch13点层，点层容量保持400；两份权重保存到
`models/runs/casia_to_calligraphy/frozen_weights/` 并校验SHA256。

训练 batch32字、val batch16字、级联点层 batch128笔，AdamW lr1e-4、weight_decay0，
每100步保存checkpoint；每epoch完整val，纯图像串联两个阶段，按整字AIoU选best。
同时保存笔画层数量/掩膜/首点指标、两阶段CharIoU/OrdIoU和XYW误差，以及整字DTW/LDTW。
指标实现仍为本地256px协议，不声明官方等价。最多200轮；当前笔画层入口没有自动早停，
单次Slurm时限3天，时限结束需从同一run的last.pt恢复。

```bash
cd /home/lkh/workspace/CHTR
gpu-sbatch --user_name lkh models/scripts/finetune_calligraphy_stroke.slurm
```

产物：`models/runs/casia_to_calligraphy/ordered_stroke/job_<ID>/train/`；
日志：`/home/lkh/workspace/logs/chtr-calli-stroke-ft-<ID>.out` 和 `.err`。
本次准备不提交正式GPU作业。


2026-11-19准备检查：两份真实权重已在CPU加载成功，笔画位置嵌入完成32→64扩容，
点层确认 `requires_grad=False`；两份快照SHA256与配置一致。
`models.tests.test_ordered_finetuning` 与 `models.tests.test_point_finetuning` 共4项CPU测试通过，
覆盖两个阶段的权重迁移、独立优化器/进度、单步训练、级联AIoU评估及best保存；
Slurm脚本通过 `bash -n`。未执行新笔画层GPU容量检查或提交正式训练。
