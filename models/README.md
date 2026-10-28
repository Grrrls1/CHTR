# 书法轨迹模型

当前主线是“有序笔画预测 → 单笔轨迹生成”。单笔轨迹阶段已实现并完成 job605 全量训练的前20轮验证；有序笔画阶段和串联尚未实现。
计划见 [TODO](TODO.md)，接口和评估协议见 [SINGLE_STROKE](SINGLE_STROKE.md)。

## 当前状态（2026-10-28）

- job605 已按用户要求彻底取消，checkpoint 和训练、验证产物保留。
- 选用 `runs/single_stroke_full/job_605/train/best.pt`：epoch 20、step 120100。
- 完整验证集83,119笔：自由生成轨迹 IoU **0.839904**、坐标 MAE **1.702859 px**、宽度 MAE **1.160251 px**、端点 L2 **4.136157 px**。
- test675 已提交并开始逐批推理：完整68,413笔，batch32，30个字符可视化。最终结果以输出目录中的 `summary.json` 为准，不把运行状态当作已完成结论。
- 当前输入为真实笔画 mask 覆盖规范化整字原图，区域外填白，并提供真实首点 `(x,y,width)`；后续点自由生成。这是单笔条件评估。

## 数据与输出

固定划分：[full_character_v1.json](data_splits/full_character_v1.json)，说明见 [data_splits/README](data_splits/README.md)。

| 集合 | 字图 | 笔画 |
| --- | ---: | ---: |
| train | 56,622 | 576,465 |
| val | 8,121 | 83,119 |
| test | 6,714 | 68,413 |

全部集合共用 `../data/chtr_mmah_lmdb/`；先划分整字，再展开单笔。val 用于选择模型，test 不参与调参或早停。
数据、checkpoint 和运行产物被 Git 忽略，拉取代码后须单独准备。

## 运行入口

服务器与个人电脑环境：[ENVIRONMENT](ENVIRONMENT.md)。个人电脑安装、文件准备、CPU/GPU测试命令：[TEST_README](TEST_README.md)。

服务器独立 test（执行会提交新作业）：

```bash
cd /home/lkh/workspace/CHTR
gpu-sbatch --user_name lkh models/scripts/test_single_stroke_job605.slurm
```

当前测试结果：`runs/single_stroke_full/job_605/test_job_675/`，包括 `summary.json`、逐笔 `strokes.jsonl`、30个字符的 `index.html` 和轨迹NPZ。
实时日志：

```bash
tail -f /home/lkh/workspace/logs/chtr-stroke-test605-675.out
```

独立评估入口输出初始化阶段、每15秒状态提示，以及每批完成数、比例、平均速度和预计剩余时间。心跳不代表磁盘读取一定在推进。
后续若重训，使用 `scripts/train_single_stroke.py` 与 `configs/single_stroke.json`，服务器入口为 `scripts/train_single_stroke_full.slurm`；当前没有继续训练的待办。

## 代码组织

```text
datasets/single_stroke.py             当前单笔数据与批处理
datasets/dataset_split.py             固定划分及完整性校验
networks/single_stroke.py             四尺度单笔轨迹模型
losses/single_stroke.py               坐标、宽度、有效性损失
training/single_stroke*.py            checkpoint与完整集合评估
scripts/*single_stroke*               训练、独立评估、审计及Slurm入口
tests/                               当前模型、固定划分及基线回归测试
```

回归检查（CPU合成数据，不读取完整训练库）：

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 third_party/StrokeExtraction/.venv/bin/python -m unittest discover -s models/tests -v
```

## 历史基线和清理范围

保留 job237 的模型、推理与评估代码、配置及运行资产，供历史复查；其结构见 [ARCHITECTURE](ARCHITECTURE.md)，本地评估见 [LOCAL_EPOCH100](../reproduction/LOCAL_EPOCH100.md)。
job237 曾见完整训练库，其32图评估是训练集诊断，不能与 job605 的独立验证分数直接比较。
free rollout、旧三任务分层 AR、扩散 / Flow Matching 已放弃；对应专用文档、代码、配置和测试已清理。共享基线模块和数据生成管线保留。
