# 独立三任务分层自回归基线

实现日期：2026-10-22。对应 [主 TODO](TODO.md)。本模式新增文件实现，不修改旧模型、旧损失或旧训练 / 评估入口。
可复用原有 ResNet/FPN、注意力缓存、渲染及早停工具；不会构造旧六头模型。

## 模型与三个任务

- 模型：[networks/hierarchical_ar.py](networks/hierarchical_ar.py)，类 `HierarchicalAR`。
- 几何头：每点预测归一化 x/y/width。XY sigmoid，width softplus，3维自预测几何反馈。
- 点结束头：每点一个 logit，决定当前笔是否结束。
- 笔画结束头：每笔一个 logit，当前笔完整生成后决定整字是否结束。
- 无 mask、type、辅助 endpoints、数量头或旧三分类状态输入；几何梯度可回传全部上游。
- Backbone 为完整 ResNet-18 + 四级 FPN，随机初始化，dim=128。
  四级特征按细/粗分组池化融合成8×8和4×4网格，共80个视觉tokens；
  无mask分支时仍让所有FPN层参与监督，避免保留未使用的高分辨率分支参数。
- 训练和推理均逐点自回归，保留完整历史梯度；换笔重置点缓存，历史笔画 token 保留。

训练使用真实有效笔数和每笔点数组织计算，但不输入真实坐标历史、不把真实数量嵌入条件。
预测的两级结束不执行 break，所以全部原始点得到监督；不重采样、不把缺失点补零。
这属于“自预测几何历史、标签长度展开”，不是完全按预测停止的 free rollout。
推理只接收图像，点结束和笔画结束独立阈值默认0.5；上限384点/笔、16笔/图。
达到上限但没有预测结束时标记截断，不制造正常完成。

## 损失与归约

独立实现：[losses/hierarchical_ar_loss.py](losses/hierarchical_ar_loss.py)。

```text
L = 1 * geo + 1 * point_end + 1 * stroke_end
geo = 每图均值(每笔均值(每有效点的三个坐标绝对误差均值))
```

标签坐标 x/(W-1)、y/(H-1)、width/S；方形图 S=H=W。
点结束 BCE 的标签为最后一点1、此前0；每笔正负组均值各占一半，再按笔画及图片平均。
笔画结束 BCE 对最后一笔为1、此前0，每图正负组均值各半，再按图片平均。
只有正组的单点 / 单笔样本只计算正组，不乘0.5。
没有 start、delta、length、missing、extra、mask、type 或 endpoint 辅助损失。

动态压缩训练活跃批次，不生成 padding 点。batch_size 表示并行图片数，gradient_accumulation
表示每次更新累积的微批次数；尾批按实际图片数计权。ResNet 使用原生 BatchNorm，
因此小微批累积与同等图片数的大并行批次可能因批统计不同而有数值差异；仅损失权重归约一致。

## 数据与固定划分

新数据读取器：[datasets/hierarchical_ar_dataset.py](datasets/hierarchical_ar_dataset.py)，只读取原图和轨迹 NPZ。
不要求 mask PNG、笔画类别文件或 stroke_type_map.json；原始轨迹中的旧事件不作为新模型输入。
训练启动时扫描实际 offsets 校验上限，并对图像 / 轨迹内容计算摘要用于严格恢复。

配置：[configs/hierarchical_ar.json](configs/hierarchical_ar.json)。1915图默认按固定seed=42按图留出10%：

| 划分 | 数量 |
| --- | ---: |
| 训练 | 1723 |
| 留出验证 | 192 |
| 固定训练图评估 | 32 |
| 固定留出图评估 | 32 |

这是样本级划分，不保证汉字类别或书写风格独立。旧 job333 使用全部1915训练，比较时须注明差异。
需要全1915拟合实验时，在新配置副本设置 validation_fraction=0；此时只能报告训练图拟合。
train_limit 仅限制训练子集，不会将留出图加入训练。实际ID写入每个run的 split.json。

## 运行与恢复

以下仅提供命令，本次未提交正式训练或GPU验证任务。GPU计算应通过项目 Slurm 队列。

```bash
cd /home/lkh/workspace
mkdir -p logs
gpu-sbatch --user_name lkh CHTR/models/scripts/train_hierarchical_ar.slurm
```

新模式脚本申请1张GPU、4 CPU、32G系统内存（不是显存限额）。日志为
`logs/chtr-ar3-r18-b24-<jobid>.out/.err`，输出为
`models/runs/hierarchical_ar_three_task/job_<jobid>/`。batch24 GPU容量仍需实测。

短程GPU检查可显式限制更新和评估数：

```bash
gpu-sbatch --user_name lkh --job-name=chtr-ar3-check \
  CHTR/models/scripts/train_hierarchical_ar.slurm --max-steps 2 --eval-limit 1
```

新Slurm分配恢复到原目录，保留原批次 / 数据 / 划分 / 归约等设置：

```bash
gpu-sbatch --user_name lkh CHTR/models/scripts/train_hierarchical_ar.slurm \
  --output-dir /absolute/path/to/original-run \
  --resume /absolute/path/to/original-run/last.pt
```

调度器以同一job ID重新排队时，脚本发现其专属 last.pt 会恢复该新模式断点。
新run必须空目录，输出目录有进程锁，避免两个trainer同时写同一run。
checkpoint以 `model_mode=hierarchical_ar_three_task`、`model_version=1` 标识；旧模型断点会被拒绝。
恢复严格校验模型配置、输入内容摘要、划分、批次、损失和阈值等；允许改变device、epochs和max_steps。
保存模型、优化器、游标、CPU/CUDA/Python/NumPy随机状态与早停累计；SIGUSR1/SIGTERM在更新边界保存。
`max_steps`、`step`均表示优化器更新数；训练中断只能恢复到最近写入成功的断点。

## 评估与产物

周期默认每20轮，并在最后一轮、早停或到达max_steps时评估。

- `metrics.jsonl`：三项原始loss、加权总loss、原始点像素距离、宽度MAE、首尾误差、梯度、时间和GPU allocated/reserved峰值。
- `epoch_metrics.jsonl`：按实际训练图片数加权的轮均loss及早停状态。
- `validation.jsonl`：分开记录固定训练图与留出图的完整自主停止结果。
- `last.pt`：最近断点；`best_loss.pt`：轮均训练loss有效改善的断点。
- `best.pt`：有留出集时按固定留出图的image-only IoU选取，否则按固定训练图；包含随机初始化基线。
- `eval_step_*/training_images/`、`heldout_images/`：summary.json、index.html、comparison_sheet.png及逐图input/label/free/overlay。
- `prediction.npz`：六维像素点、实际offsets、两级结束概率及完成/截断状态；没有虚构的mask/type字段。

六维事件由新推理边界派生，仅用于输出协议；截断的末笔可保留continue，不能误写drawing_end。
渲染时在显示副本上补端点，不修改NPZ。错笔数样本仍计入IoU、笔数误差、总点数误差；
逐笔点数MAE按序号配对，缺失/额外笔画以0点计入，明确不等同于语义最佳匹配。
另列 `label_length_geometry` / `label_length_losses` 诊断：它们用了真实长度，不能当作自主生成质量。

独立评估只接受本模式断点，默认选择保存的留出ID：

```bash
CHTR/third_party/StrokeExtraction/.venv/bin/python \
  CHTR/models/scripts/evaluate_hierarchical_ar.py \
  --checkpoint /absolute/path/to/run/last.pt \
  --output /absolute/path/to/new-eval-directory --device cuda:0 --limit 32
```

GPU命令放进已分配的Slurm作业内执行；CPU可显式 `--device cpu`。
`--split training_images` 查看训练图拟合；`--split all` 表示混合输入，不宣称独立验证。

## 已做验证与尚未完成

- 全套65项CPU测试通过，其中新增16项覆盖模型与真实LMDB格式的训练/评估流程。
- 新测试覆盖变长压缩、提前停止不影响训练覆盖、跨点完整反传、几何梯度到达ResNet和笔画模块、
  两级自主停止、截断事件、batch24维度、尾批归约、微批计权、精确CPU断点恢复和离线评估导出。
- 实际1915数据扫描：8990笔，最多13笔/图、338点/笔，全部在新上限内。
- GPU batch24容量、长程小集收敛、固定32图拟合及正式训练尚未执行。
  单元测试和一次反传不代表几何已拟合，也不证明新结构优于旧模型。
