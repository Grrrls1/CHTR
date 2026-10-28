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
- 服务器配置由用户后续补充；未填写的字段保持待补充，不根据个人电脑配置推断。

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

## 服务器（待用户补充）

| 项目 | 配置 / 状态 |
| --- | --- |
| 服务器标识 / 节点 | 待补充 |
| 系统版本 | 待补充 |
| 仓库路径 | 现有测试说明记录 `/home/lkh/workspace/CHTR/`，后续在服务器确认 |
| Python 解释器 / 环境名 | 待补充 |
| Python 版本 | 待补充 |
| PyTorch / torchvision | 待补充 |
| GPU 型号、数量、单卡显存 | 待补充；区分节点硬件与作业实际获配资源 |
| NVIDIA 驱动 | 待补充 |
| NVIDIA-SMI 显示 CUDA Version | 待补充 |
| `torch.version.cuda` | 待补充 |
| 作业内 CUDA 可用性 | 待补充；登录节点检测不能代替计算作业检测 |
| Slurm 分区 / GPU 资源参数 | 待补充；遵循现有 HANDOFF 与用户的提交指令 |

后续用户提供服务器环境后直接更新此节，保留个人电脑配置，不混合两边的 CUDA 结论。
