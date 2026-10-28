# 个人电脑与服务器环境记录

本文件集中保存机器配置，供后续训练、评估和排错使用。个人电脑与服务器分别记录，
不要把一台机器或一个 Python 解释器的检测结果套用到另一环境。

## 后续使用约定

- 讨论 CUDA、显存或运行命令前先读本文件，不重复询问已经记录的硬件配置。
- 个人电脑已确认具有 NVIDIA GPU，驱动可正常识别；不能因某个解释器返回
  `torch.cuda.is_available() == False` 就断言本机没有 GPU 或只能使用 CPU。
- 分别记录 GPU / 驱动、实际 Python 环境、PyTorch 的 CUDA 构建及进程可见性。
  `nvidia-smi` 中的 CUDA Version 与 `torch.version.cuda` 分开记录，不要求数值相同。
- 仅当环境发生变化、切换机器 / 解释器，或运行出现相关错误时补充针对性检查。
  不需要用户每次重新粘贴 CUDA 配置。
- 服务器配置依据服务器实测和已保存的 Slurm 作业内检测记录，不根据个人电脑配置推断。

## 个人电脑

记录时间：2026-10-28 19:12:37（Asia/Shanghai）。
硬件依据：用户提供的本机 `nvidia-smi` 输出。

| 项目 | 配置 / 状态 |
| --- | --- |
| 用途 | 本地开发、单笔轨迹模型测试集评估与可视化 |
| 仓库路径 | `/home/lkh/vscode/CHTR` |
| 系统 | Linux 桌面环境；发行版、版本待补充 |
| GPU 数量 | 当前输出显示1张，编号0 |
| GPU 型号 | NVIDIA GeForce RTX 4060；输出末尾被截断，完整型号后缀未确认 |
| GPU 总显存 | 8,188 MiB，约8 GiB |
| NVIDIA 驱动 | 595.99.02 |
| NVIDIA-SMI 版本 | 595.99.02 |
| NVIDIA-SMI 显示 CUDA Version | 13.2 |
| GPU 状态 | 驱动识别正常，桌面正在使用 GPU |
| 测量时显存占用 | 898 MiB / 8,188 MiB；仅为当时快照，不是固定可用显存 |
| 桌面进程 | Xorg、gnome-shell、VS Code、Firefox 等 |
| 实际评估 Python 环境 | 待确认并补充；与下列历史解释器记录区分 |
| 实际评估环境的 torch / torchvision | 待补充 |
| 实际评估环境的 `torch.version.cuda` | 待补充 |
| 实际评估环境的 CUDA 可用性 | 以该环境在本机终端的实际检测为准 |

### 已有解释器检测记录

本会话工具曾使用 `/home/lkh/anaconda3/envs/test/bin/python`，得到：

| 项目 | 检测值 |
| --- | --- |
| PyTorch | 1.13.0 |
| torchvision | 0.14.0 |
| `torch.cuda.is_available()` | False |

此结果仅描述该解释器在当时会话执行环境中的检测情况，不能覆盖上面的本机 GPU / 驱动事实，
也不表示这是最终选择的测试环境。此前 CPU smoke 结果仍是 CPU 实测，不改写为 GPU 实测。

如需补全实际评估环境，在用户选定的环境中运行一次并将输出记录到上表即可：

```bash
python -c "import sys, torch, torchvision; print('python:', sys.executable); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not visible in this process')"
```

job605 的本机测试操作见 [TEST_README](TEST_README.md)。使用 GPU 评估时指定
`--device cuda:0`，batch 大小按本机实测显存调整；尚无本机完整测试的显存容量验收结论。

## 服务器

记录时间：2026-10-28（Asia/Shanghai）。系统、依赖、GPU 与 Slurm 信息在服务器只读核实；
作业内 CUDA 运行结果引用 job605 启动时保存的
[environment.json](runs/single_stroke_full/job_605/environment.json)，本次未新增 GPU 检测作业。

| 项目 | 配置 / 状态 |
| --- | --- |
| 主机名 / Slurm 节点 | `amax` / `server1` |
| 系统 | Ubuntu 22.04.5 LTS，x86_64 |
| 内核 | `6.8.0-86-generic` |
| CPU | 2 × Intel Xeon Platinum 8468；共96物理核、192逻辑 CPU |
| 内存 / Swap | 约503 GiB / 15 GiB；整机容量，不是单作业额度 |
| 工作区 / 仓库 | `/home/lkh/workspace` / `/home/lkh/workspace/CHTR` |
| 工作区文件系统 | `/dev/sdb1`，ext4，挂载于 `/home/lkh/workspace`，总容量约15 TiB |
| 训练 / 评估解释器 | `/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python` |
| Python | `3.10.19` |
| PyTorch | `2.12.1+cu130` |
| torchvision | `0.27.1+cu130` |
| NumPy / Pillow | `1.23.5` / `9.5.0` |
| opencv-python-headless / lmdb | `3.4.18.65` / `3.0.0` |
| GPU | 4 × NVIDIA GeForce RTX 5090，物理编号0–3 |
| 单卡显存 | 32,607 MiB，约31.84 GiB（nvidia-smi 总量） |
| NVIDIA 驱动 / NVIDIA-SMI | `580.142` / `580.142` |
| NVIDIA-SMI 显示 CUDA Version | `13.0`，驱动支持信息 |
| `torch.version.cuda` | `13.0`，当前 PyTorch 构建使用的 CUDA 版本 |
| 系统 CUDA Toolkit / nvcc | `/usr/local/cuda/bin/nvcc`；CUDA `12.8`，`V12.8.93` |
| cuDNN | `torch.backends.cudnn.version()` 返回 `92000` |
| 作业内 CUDA 验证 | job605：`available=true`、可见1张 RTX 5090、`cuda:0`，CUDA kernel 和反向检查通过 |
| GPU 计算能力 | job605 检测为 `12.0`；该 PyTorch 构建包含 `sm_120` |
| Slurm | `26.05.4` |
| 默认分区 / 节点资源 | `foreground` / `server1`；GRES 显示 `gpu:5090:4` |
| 单笔训练脚本申请资源 | 1 GPU、4 CPU、32 GiB 内存、3天时限；属于脚本配置 |

### 解释器与 CUDA 使用

直接使用上表的独立解释器，无需激活环境。不要将系统 `python3` 或其他 Conda 环境
视为已安装相同依赖；当前 shell 的 PATH 没有裸 `python` 命令。
`requirements-env.txt` 是另一套环境创建时使用的依赖约束，不等于上表已实测的现存训练环境。

```bash
cd /home/lkh/workspace/CHTR
third_party/StrokeExtraction/.venv/bin/python -c "import sys, torch, torchvision; print(sys.executable); print(torch.__version__, torchvision.__version__); print('torch CUDA:', torch.version.cuda)"
```

驱动支持版本、系统 `nvcc` 和 PyTorch CUDA 构建是三个独立字段。
这里分别为13.0、12.8和13.0；运行现有 PyTorch wheel 不以系统 `nvcc` 版本判断 CUDA 可用性。
只有编译自定义 CUDA 扩展时，才需要进一步核对工具链兼容性。
沙箱或未分配 GPU 的进程可能无法访问设备，不能据此否定已保存的作业内检测结果。

### Slurm 提交与运行约定

服务器提供 `/usr/local/bin/gpu-sbatch`、`gpu-srun`、`gpu-salloc` 包装器，
要求首个参数为 `--user_name lkh`，由包装器把用户元数据写入原生 Slurm 提交参数。
作业内部保留 Slurm 设置的 `CUDA_VISIBLE_DEVICES`，用 `cuda:0` 访问获配 GPU；
这个编号是进程内编号，不一定对应物理 GPU 0。

现有训练提交示例（会提交新的训练，按实际任务需要执行）：

```bash
cd /home/lkh/workspace/CHTR
gpu-sbatch --user_name lkh models/scripts/train_single_stroke_full.slurm
```

脚本设置 `PYTHONUNBUFFERED=1`、`PYTHONNOUSERSITE=1`、`MPLBACKEND=Agg`，
以及 `OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`、`MKL_NUM_THREADS=4`。
读取可信训练 checkpoint 时设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。
日志目录为 `/home/lkh/workspace/logs/`，模型产物位于 `models/runs/`。

### 数据与存储

单笔训练 / 测试共用 `data/chtr_mmah_lmdb/`，其中 `data.mdb` 约3.6 GiB；
划分文件是 `models/data_splits/full_character_v1.json`。
2026-10-28 记录时，工作区分区使用率95%，可用约827 GiB；这是当时快照，应以 `df` 为准。
测试 job672 曾出现 `D` 状态及 `folio_wait_bit_common` 等待，已按用户要求取消；
这是观察到的文件页面 I/O 等待，未确认底层故障原因，不能归因于 GPU 推理。

后续环境变化时更新对应字段，保留个人电脑配置，不混合两边的 CUDA 结论。
