# CASIA-OLHWDB1.1 两阶段训练

本设置独立训练点层和有序笔画层。网络随机初始化，不加载书法权重；原有书法配置、run和产物保留。
数据准备及CPU回归不使用GPU。所有CUDA工作通过 `gpu-sbatch --user_name lkh` 提交。

## 数据准备

官方数据来源：[CASIA下载页](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)。
使用原始 `Pot1.1Train.zip` / `Pot1.1Test.zip`，不是512维特征包或离线GNT图像。

在仓库根目录执行（已下载时核验后跳过相同文件）：

```bash
PY=third_party/StrokeExtraction/.venv/bin/python
$PY models/scripts/download_casia_olhwdb.py --split train --download-tool aria2c
$PY models/scripts/download_casia_olhwdb.py --split test --download-tool aria2c
$PY models/scripts/prepare_casia_training.py
```

默认生成 `data/CASIA-OLHWDB/training_v1/{index.sqlite,split.json,audit.json}`。
准备目录必须不存在，失败时保留未完成目录用于排错；重新构建使用新的 `--output` 并同步修改配置。
下载保留原始ZIP、续传文件及带SHA256的下载报告，解压核验CRC、路径和已有文件内容。

字符范围固定为解码后的U+3400～U+9FFF，与现有本地外部评估入口一致；不宣称严格等于论文类别集合。
官方训练集按POT文件名识别书写者，用NumPy default_rng、seed42无放回选24位作为val，其余train；
官方test不参与训练、参数选择或模型选择。审核包括test资产完整性和长度统计，不依据test调整笔宽或模型。
拒绝重复writer文件、跨官方划分writer重合及无法解码的标签。
所有合格中文记录都保留，重复点、单点笔画、笔顺和笔画分隔不变；不平滑、简化、重采样或截断。
`audit.json`记录分集样本/笔画/点数、长度直方图、最大长度、重复点、预设上限覆盖。

SQLite只存索引和长度，按字随机读取POT；不为百万字逐笔保存256像素mask。
启动时核验manifest、索引和所有POT的SHA256，以及样本和writer互斥，拒绝数据修改后继续旧run。

## 渲染协议 v1

- 所有笔画共享整字等比例变换：256×256、边距8、最大跨度239像素、包围框中心127.5；XY归一化除255。
- 首版训练每字每epoch一个整数直径，均匀采样 `[2,12]` 像素，全字笔画共用。
  通过 `SHA256(seed:epoch:sample_id)` 的前8字节初始化NumPy default_rng；相同epoch和样本在两阶段得到相同宽度，恢复不依赖取样顺序。
- 验证/测试固定直径8像素，不刷新宽度、不挑选最有利输入。训练集诊断评估也默认固定宽度。
- OpenCV LINE_8、4倍超采样、圆形端帽、INTER_AREA降采样；每笔coverage阈值128生成二值mask，整字coverage取各笔最大值后反色。
  图像、mask、框、起点和点序列在同一次取样内生成。
- W是256画布内的合成渲染直径，训练除256；不代表真实笔宽或压力。主任务是有序XY恢复，W指标仅辅助。
- 固定宽度对照配置只覆盖训练宽度范围为 `[8,8]`，其他数据和渲染设置保持一致。两个阶段应配对使用同一宽度设置的checkpoint。

这些是本地首版协议，宽度范围及模型训练超参数尚未经train/val充分拟合验收，不是论文官方设定。

## 训练及GPU接入检查

先执行真实数据的有限步接入检查：

```bash
gpu-sbatch --user_name lkh models/scripts/casia_gpu_smoke.slurm
```

该作业在真实train/val各抽少量样本，两阶段分别3个优化步骤（包含断点恢复），
使用完整Swin-T/9层笔画网络，验证冻结点层及独立串联val入口。
同一脚本可在 `CUDA_VISIBLE_DEVICES=''` 下加 `--device cpu --output <新目录>` 执行真实数据CPU接入检查。
自由解码只允许8点/2笔，属于明确截断的小集诊断；不读取官方test，不是充分拟合、效果或全量容量验收。
报告位于 `models/runs/casia_olhwdb11/smoke/job_<id>/check/smoke_report.json`。

全量训练入口（本次准备工作不自动启动）：

```bash
gpu-sbatch --user_name lkh models/scripts/train_casia.slurm single_stroke
# 点层训练完成后，复制有序笔画配置，将point_checkpoint设为该CASIA run的train/best.pt，再提交：
gpu-sbatch --user_name lkh models/scripts/train_casia.slurm ordered_stroke /path/to/ordered_config.json
```

随机宽度配置为 `models/configs/casia_{single_stroke,ordered_stroke}.json`；
固定宽度对照为 `casia_{single_stroke,ordered_stroke}_fixed_width.json`。
有序笔画配置中的point_checkpoint是待替换路径，必须指向实际CASIA点层best；
加载时核验CASIA后端、同一数据版本/划分/渲染及点层checkpoint签名，拒绝书法权重。
所有run使用独立Slurm job目录。根据train/val审核的最大单笔192/116点、最大26/25笔，首版点层上限冻结为256点、笔画层32笔；
这两个上限依据train/val确定，test仅核验覆盖。超过上限直接报错，不能静默丢弃。
2026-11-06按用户要求，CASIA点层正式配置（随机/固定宽度两组）使用训练batch256笔、验证及测试batch128笔。
训练入口的val与可选final test共用 `eval_batch_size=128`；独立 `evaluate_single_stroke.py` 对CASIA checkpoint默认128，
可用 `--batch-size` 显式覆盖。笔画层的整字batch设置独立保留。
本次未做容量测试，未重启作业；job1136已有run配置和checkpoint仍记录原batch16。
已有checkpoint的训练恢复签名包含batch，改为256后不能直接使用原 `--resume`；变更batch使用下面的 `--continue-from` 新run入口。

两个阶段都以**val AIoU**选best。点层是带真实mask和起点的逐笔诊断，按笔等权平均；
笔画层串联冻结的CASIA点层，以纯图片输入的逐字AIoU等权平均选best。
`best_selection_score`记录所选分数；旧书法IoU字段在CASIA run中置null，避免混淆指标。
`final_test=false`，正式模型与协议冻结后再单独运行test，不自动在每次调参后测test。

## 指标口径

依据[PEN-Net论文第3节](https://arxiv.org/html/2210.15879#S3)，本地实现
`pen_net_equations_local256_v1`：输入图OTSU前景，与预测XY的1像素中心线比较，
采用3×3方形核重复膨胀，取直到画布饱和的全部候选中的最大IoU。
使用棋盘距离直方图一次计算各次膨胀的交并集，并与显式OpenCV逐次膨胀做数值一致性测试。
不在不同笔画之间画连接线，单点笔画渲染一个像素，空预测AIoU=0；W不参与中心线生成。
这不是固定宽度IoU，也不是通过改变val/test输入笔宽选分。

官方仓库公开的train入口引用未提供的test_core，未找到可直接核验的官方指标实现；
本地实现只声明论文公式层面的复现，不声明与官方代码或论文数值一致。
坐标分辨率、舍入、端帽、阈值、膨胀穷举和聚合方式均显式记录，不能直接横比论文分数。

同时输出精确欧氏DTW最小累积代价和 `LDTW=DTW/所选最优路径长度`，
坐标单位为256画布像素，路径代价相同时优先对角、竖直、水平；按笔顺拼接XY，但抬笔间不增加连接点。
保留旧外部评估字段名 `dtw_path_mean_px`，新CASIA训练评估显式另报 `ldtw_px`。
空序列DTW/LDTW记null并单列数量，不按0混入均值。另保留漏笔/多笔、笔数、停止和截断指标。
普通IoU仍保留为不同指标；现有宽度IoU和W误差只作辅助。FID在本次数据/训练适配中不加入每轮val；
如最终评估需要FID，使用独立外部评估入口及其冻结的渲染/特征协议，另行报告。

服务器已有uv管理的环境可执行 `uv pip install --python third_party/StrokeExtraction/.venv/bin/python "numba>=0.57,<0.59"`。
全量DTW需Numba加速，依赖见 `models/requirements-casia-eval.txt`；缺Numba时有精确Python实现，但不适合全量评估。

## CPU回归

```bash
CUDA_VISIBLE_DEVICES='' TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 third_party/StrokeExtraction/.venv/bin/python -m unittest discover -s models/tests
```

新检查覆盖真实格式POT的保序/重复点/单点、writer互斥、数据篡改拒绝、训练宽度复现、
两阶段资产一致、停止标签、上限拒绝和AIoU与显式膨胀的一致性。合成数据检查不替代真实POT与GPU接入结果。

## 本次准备结果（2026-11-03）

| 划分 | 书写者 | 字 | 笔画 |
| --- | ---: | ---: | ---: |
| train | 216 | 808,696 | 4,480,881 |
| val | 24 | 89,877 | 519,116 |
| official test | 60 | 224,559 | 1,251,111 |

原始ZIP、POT及准备索引合计约720MB。完整79项CPU回归及4项针对性复验通过；
真实数据两阶段各3步CPU训练/恢复、独立串联val通过，使用完整Swin-T/9层，解码限制8点/2笔。
GPU作业1135已按lkh提交但仍在排队，不能据此宣称GPU验证通过。未启动正式全量训练。

详细结果见 [准备报告](reports/casia_training_preparation_20261103.json)，
[渲染预览](reports/casia_training_preview_20261103.png)每行同一字，三列分别为2/8/12px。

## 从job1136继续训练：train256、val/test128

先等待epoch4完整验证结束且checkpoint保存完成，再停止旧作业，避免两份训练同时运行。
仅有eval004的summary还不能保证last.pt已提交验证状态；新入口会核验 `last_evaluated_epoch>=4`，
若不满足会报错，不会跳过这轮未提交的验证。最好在日志已经出现验证后的下一条train记录时再停止。

```bash
cd /home/lkh/workspace/CHTR
# 确认第4轮验证和checkpoint保存完成后，由用户执行：
scancel 1136
# 确认squeue中旧作业已退出，再提交：
gpu-sbatch --user_name lkh models/scripts/train_casia.slurm single_stroke \
  models/configs/casia_single_stroke.json \
  --continue-from models/runs/casia_olhwdb11/single_stroke/job_1136/train/last.pt \
  --require-evaluated-epoch 4
```

产物写入新作业的 `models/runs/casia_olhwdb11/single_stroke/job_<新ID>/train/`，
保留旧job1136目录。新目录保存 `parent_last.pt` 快照、`continuation.json`（来源SHA256与参数变化）、
继承的best，以及新的last/best、配置、metrics和后续评估。继承的best若尚未被新阶段超过，仍保留父checkpoint的原始配置。

该入口恢复模型、AdamW动量、scheduler、RNG、累计step、epoch、已消费样本cursor、早停和历史best。
只允许训练/评估batch及输出、设备、线程、总epoch、step限制改变，其余训练/数据/渲染/模型设置必须相同。
原 `--resume` 继续保留同目录且签名完全一致的严格恢复语义。若停机时第5轮已开始，会从最近保存的第5轮cursor继续，而不是重跑第4轮。
200 epochs仍表示历史累计总轮数，不是额外再训练200轮。训练batch256、val/test128、学习率3e-4。
本入口按用户要求未运行测试；batch256的GPU容量也尚未验证。本次没有取消或提交任何作业。
