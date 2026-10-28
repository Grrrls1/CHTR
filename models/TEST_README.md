# job605：个人电脑上的独立 test

独立入口是 `models/scripts/evaluate_single_stroke.py`，不需要启动训练或安装 Slurm。
以下命令均在个人电脑的 **CHTR 仓库根目录**执行。

个人电脑与服务器的硬件、驱动及Python环境记录见 [ENVIRONMENT](ENVIRONMENT.md)。
个人电脑已确认具有RTX 4060 GPU；某个解释器的CUDA检测结果不能代替整机配置。

## 评估内容

- 使用 job605 的 `train/best.pt`：epoch 20、step 120100，验证集轨迹 IoU 为 0.8399036936。
- test 集共 6,714 个字图样本、4,993 个不同汉字、68,413 条笔画。
- 默认命令评估全部 test 笔画，可视化 30 个不同汉字各一个字图及其全部笔画。
- 输入包含真实笔画 mask 和真实首点 `(x, y, width)`；后续点自由生成。
  这是单笔轨迹阶段的评估，不是仅输入整字图片的端到端预测。

## 需要复制的文件

从服务器复制 `CHTR/models/` 的代码、划分文件和下列 checkpoint，以及完整 LMDB。
最省事的方式是复制 `CHTR/models/`，排除 `runs/` 后单独复制 `best.pt`。
不需要复制第三方项目虚拟环境、训练日志、`last.pt` 或历史可视化。

个人电脑上保持以下结构：

```text
CHTR/
├── data/
│   └── chtr_mmah_lmdb/
│       ├── data.mdb
│       ├── complete.json
│       ├── summary.json
│       └── stroke_type_map.json
└── models/
    ├── datasets/
    ├── losses/
    ├── networks/
    ├── training/
    ├── scripts/evaluate_single_stroke.py
    ├── data_splits/full_character_v1.json
    └── runs/single_stroke_full/job_605/train/best.pt
```

服务器源路径：`/home/lkh/workspace/CHTR/`。
`data.mdb` 约 3.6 GiB，`best.pt` 约 138 MiB。数据应放在本地 SSD，避免网络盘或同步盘。
读取器使用只读、无锁模式，不需要复制运行中的 `lock.mdb`。
只复制 test 对应记录不够：启动会校验完整候选元数据和划分指纹，需要原始完整 LMDB。
保留上述相对目录，不要修改 checkpoint 内的配置或重新生成划分文件。

## 环境选择

个人电脑使用已核验的Conda `test`：

```text
/home/lkh/anaconda3/envs/test/bin/python
```

2026-10-28实测torch1.13.0、torchvision0.14.0、CUDA构建11.7；沙箱外RTX4060
CUDA前向/反向测试通过，且可加载job605的best.pt。无需新建环境或重复安装依赖。
下文直接调用该解释器，无需激活环境。
`third_party/StrokeExtraction/.venv/bin/python`属于服务器环境，不是本机10月23日CPU smoke的解释器。
详细记录见 [ENVIRONMENT](ENVIRONMENT.md)。

checkpoint 包含模型和训练状态。当前加载器需要以下环境变量，且只应加载自己复制的可信 checkpoint。
Linux / macOS：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
```

Windows PowerShell：

```powershell
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
$env:PYTHONUNBUFFERED = "1"
```

## 运行命令

以下使用本机Linux终端命令。
先进入本机 `CHTR` 目录：

```bash
cd /home/lkh/vscode/CHTR
```

先用 64 条笔画检查环境和数据链路（部分诊断，不是完整 test 指标）：

```bash
/home/lkh/anaconda3/envs/test/bin/python -u models/scripts/evaluate_single_stroke.py --checkpoint models/runs/single_stroke_full/job_605/train/best.pt --output models/runs/single_stroke_full/job_605/test_local_check --split test --limit 64 --visualize-characters 5 --batch-size 4 --device cuda:0 --threads 4
```

这个检查仍会读取并校验完整数据集元数据；部分采样也不保证覆盖所选字图的全部笔画。
没有 CUDA 时，将 `--device cuda:0` 改成 `--device cpu`。

完整 test，30 个字符可视化，GPU：

```bash
/home/lkh/anaconda3/envs/test/bin/python -u models/scripts/evaluate_single_stroke.py --checkpoint models/runs/single_stroke_full/job_605/train/best.pt --output models/runs/single_stroke_full/job_605/test_local --split test --limit 0 --visualize-characters 30 --batch-size 4 --threshold 0.5 --decode-max-points 400 --device cuda:0 --threads 4
```

完整 test，CPU（预计比 GPU 慢）：

```bash
/home/lkh/anaconda3/envs/test/bin/python -u models/scripts/evaluate_single_stroke.py --checkpoint models/runs/single_stroke_full/job_605/train/best.pt --output models/runs/single_stroke_full/job_605/test_local_cpu --split test --limit 0 --visualize-characters 30 --batch-size 4 --threshold 0.5 --decode-max-points 400 --device cpu --threads 4
```

`--limit 0` 表示全量评估；`--visualize-characters 30` 仅限制可视化数量。
本机8GiB显存先用batch4；显存不足时降到2或1，显存充足后可实测增大。
这是单 GPU 批量推理；点序列仍逐点自回归生成，数据读取没有多进程预取。
为保持与 epoch 20 验证的评估口径一致，请保留阈值 0.5 和解码上限 400。

## 输出与常见问题

输出目录中的文件：

- `summary.json`：整体指标、checkpoint epoch/step、是否完整评估、实际可视化数量。
- `strokes.jsonl`：每条笔画的自由生成指标。
- `index.html`：可视化总览，用浏览器打开。
- 各样本子目录：原图、GT/预测叠加图、逐笔图、轨迹 NPZ 和详情页。

必须使用尚不存在的 `--output` 目录；重复运行时换一个新目录名。
checkpoint 和配置内的数据路径相对 `CHTR` 根目录解析；`--output` 相对当前工作目录解析。

启动会先加载 checkpoint、读取 LMDB 元数据并校验划分，这段过程没有进度打印。
开始评估后，每完成 1,024 条笔画打印一次进度，结束后打印 JSON 汇总。
`-u` 能避免 stdout 缓冲，但不会给初始化阶段增加日志。
如果启动长时间没有输出，先检查数据是否完整、是否放在本地 SSD，以及磁盘读取是否受阻。

遇到 `checkpoint data/split signature mismatch`，检查是否使用原始完整数据、原划分和原 checkpoint；
不要通过修改或跳过指纹校验来继续运行。
已在个人电脑核验Conda环境CUDA前向/反向与job605 checkpoint加载；完整模型GPU评估、速度及显存容量仍需运行下述测试确认。
