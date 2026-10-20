# uv 环境与服务器训练

## 环境选择

本配置以 RTX 5090 训练为目标，同时尽量保留原项目的旧版科学计算依赖。

| 组件 | 固定版本 |
| --- | --- |
| Python | 3.10 系列（固定 3.10.19） |
| PyTorch / torchvision | 2.12.1+cu130 / 0.27.1+cu130 |
| NumPy | 1.23.5 |
| OpenCV | opencv-python-headless 3.4.18.65 |
| SciPy / pandas | 1.9.3 / 1.5.3 |
| matplotlib / Pillow | 3.5.3 / 9.5.0 |

源码中的 `np.float` 和 `np.int` 已替换为行为相同的 Python 内置类型，避免 NumPy 1.20 起的弃用警告。NumPy 保留为支持 Python 3.10 且已经过本项目验证的 1.23.5；OpenCV 保留为 `opencv-python-headless` 3.4.18.65，仍提供项目所用的 `cv2.resize/cvtColor/putText/imwrite`。不要同时安装 opencv-python，它们会覆盖同一个 cv2 模块。

配置面向 Linux x86_64 和 RTX 5090。PyTorch 使用官方 CUDA 13.0 wheel；服务器 NVIDIA 580.142 驱动高于 CUDA 13.0 wheel 所需的 Linux 最低驱动 580.65.06。检查脚本会打印设备计算能力、wheel 架构列表和兼容结论。PyTorch wheel 自带 CUDA 运行时，不要求系统另装同版本 CUDA Toolkit。

本地验证（2026-09-10 复核）：uv 0.12.10 已创建 Python 3.10.19 环境，44 个安装包通过 `uv pip check`；训练模块导入、VGG/ContentNet 预训练权重加载、NumPy/OpenCV 操作、CPU 反向传播及 torchvision NMS 均通过。三阶段训练已在服务器完成，最终 checkpoint 已返回当前工作区。当前执行节点的 `nvidia-smi` 仍无法连接 NVIDIA 驱动，所以本机不适合重跑 CUDA 训练或单字推理；服务器上仍应使用下文检查命令。

依据：
- https://numpy.org/doc/1.24/release/1.24.0-notes.html
- https://pytorch.org/get-started/previous-versions/
- 本仓库 ReadMe.md 记录的 Python 3.8 / PyTorch 1.9 是上游历史要求；本配置为支持 RTX 5090 所做的迁移。

## 本地与服务器安装

在项目目录运行。本地 uv 已安装到 `.tools`；服务器可使用已有 uv，也可同样安装项目内 uv：

```bash
python3 -m pip install --target .tools uv==0.12.10
bash scripts/uv.sh sync --locked
MPLBACKEND=Agg bash scripts/uv.sh run --locked python scripts/check_environment.py
```

`scripts/uv.sh` 将 Python、缓存、虚拟环境保留在项目的 `.python`、`.uv-cache`、`.venv` 中。`pyproject.toml` 固定直接依赖，`uv.lock` 固定解析结果。日常使用 `--locked`，防止服务器无意更新锁文件。首次安装需要联网下载约 2.7 GB，并为环境和下载缓存预留约 10 GB 空间。

另外提供从 uv.lock 导出的 `requirements.lock.txt`（含文件哈希），用于检查或兼容其他部署工具；环境的主要来源仍是 uv.lock。

## 数据与训练权重

当前代码实际读取下列路径（注意 VGG 的扩展名是 `.th`，与上游 README 的 `.pth` 不一致）：

```text
dataset/RHSEDB/train/*.npz
dataset/RHSEDB/test/*.npz
char_recognise/out_vgg_bn/model/model.th
content_net_model/out/model_content.pth
```

已检测到你放入的 `dataset/RHSEDB.zip`、`char_recognise/model.pth`、`content_net_model/model_content.pth`。如果在服务器仍使用这些位置，可执行：

```bash
unzip -n dataset/RHSEDB.zip -d dataset
mkdir -p char_recognise/out_vgg_bn/model content_net_model/out
cp -n char_recognise/model.pth char_recognise/out_vgg_bn/model/model.th
cp -n content_net_model/model_content.pth content_net_model/out/model_content.pth
```

当前 RHSEDB 已完整展开，约 54 GiB，包含 25,272 个训练 NPZ 和 2,808 个测试 NPZ；`dataset_forSegNet_ExtractNet_RHSEDB` 中间数据约 66 GiB。VGG、ContentNet 和三个最终推理 checkpoint 也均已就位，且 `results/1613` 证明三阶段推理曾成功运行。重建或搬运环境时仍要先执行检查：原代码缺少预训练权重时，某些位置只打印失败后继续运行。

## 服务器检查与训练

项目源码中的无参数 `.cuda()` 和 `.to('cuda')` 会使用进程的默认逻辑 GPU。要独占使用物理 GPU 3，在启动进程前设置 `CUDA_VISIBLE_DEVICES=3`；该 GPU 在进程内部会映射为 `cuda:0`，这是正常行为。不要再把源码设备写成 `cuda:3`，因为设置可见设备后进程内只有 `cuda:0`。

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=2 MPLBACKEND=Agg bash scripts/uv.sh run --locked python scripts/check_environment.py --training
CUDA_VISIBLE_DEVICES=2 MPLBACKEND=Agg bash scripts/uv.sh run --locked python main_train.py
```

检查脚本先执行依赖导入、NumPy/OpenCV 图像操作、CPU 反向传播和 torchvision NMS；`--training` 额外检查数据/权重路径与 CUDA 卷积反向传播。输出中的 `physical device 3 -> logical cuda:0` 表示选择正确。它不等于完整模型或数据集训练验证。

训练顺序由原 `main_train.py` 控制：SDNet → 中间先验数据 → SegNet → ExtractNet。请在项目根目录启动，原代码使用相对路径。

### RTX 5090 训练参数

`main_train.py` 的默认参数以 32 GB RTX 5090 为起点：

| 阶段 | epoch | batch size | 学习率衰减周期 |
| --- | ---: | ---: | ---: |
| SDNet | 80 | 16 | 20 epoch |
| SegNet | 20 | 32 | 4 epoch |
| ExtractNet | 40 | 32 | 10 epoch |

epoch 相比原入口的 40/10/20 加倍，学习率衰减周期也同比加倍，初始学习率仍为 `1e-4`。默认启用 4 个 DataLoader worker、pinned memory、异步 GPU 传输和 cuDNN benchmark。batch size 根据训练进程仅占约 1.8 GB、整卡总占用约 5 GB 的实测恢复到 16/32/32；首次训练建议另开终端观察：

```bash
watch -n 1 nvidia-smi
```

如果出现 CUDA OOM，优先将对应阶段 batch size 减半；例如：

```bash
CUDA_VISIBLE_DEVICES=3 MPLBACKEND=Agg bash scripts/uv.sh run --locked python main_train.py \
  --sdnet-batch-size 4 --segnet-batch-size 8 --extractnet-batch-size 8
```

如果 OOM 信息显示其他 PID 占用了大部分显存，应先确认该进程是否属于自己；batch size 无法弥补几乎被占满的 GPU。不要终止不属于自己的进程。

如果峰值显存长期低于约 24 GB，可逐阶段增加 batch，而不是一次同时增加三个阶段。全部参数可通过 `python main_train.py --help` 查看。

当前项目没有 TensorBoard 依赖，也没有 `SummaryWriter` 或 event 文件输出。训练脚本仅在各阶段的 `out/.../loss/` 目录用 matplotlib 写入 loss 曲线 PNG，并在终端打印指标。

## 单字笔画提取

使用训练完成的三个 RHSEDB checkpoint 对一个测试样本推理。物理 GPU 编号按实际空闲设备设置，例如 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 MPLBACKEND=Agg bash scripts/uv.sh run --locked python \
  extraction_stroke_application_for_single_character_.py \
  --input dataset/RHSEDB/test/1613.npz \
  --results-dir results
```

脚本默认从 `out/SDNet_RHSEDB/model/`、`out/SegNet_RHSEDB/model/` 和 `out/ExtractNet_RHSEDB/model/` 加载模型。可分别用 `--sdnet-model`、`--segnet-model`、`--extractnet-model` 覆盖。输入必须是包含 `target_image`、`reference_single_image`、`stroke_label` 和 `name` 的 RHSEDB `.npz` 文件。`reference_single_centroid` 由推理脚本根据参考单笔画现场计算，不要求输入 NPZ 预先提供。

每个样本保存到 `results/<输入文件名>/`。`all_stages.npz` 汇总全部数组；各阶段另有独立 `.npy` 和预览 PNG。主要接口包括 SDNet 的形变目标、形变场、彩色参考和逐笔画结果，SegNet 的 7 通道概率与参考先验，以及 ExtractNet 的逐笔画二值 mask。

## 打包传输

```bash
bash scripts/pack_server.sh
```

产物在 `dist/`，含代码、uv.lock、配置与检查脚本，并生成 SHA256 校验文件。服务器解压后重新运行 `uv sync --locked`，不要直接搬运 `.venv`，其解释器链接和路径依赖本地位置。

代码包排除数据、权重、训练输出、缓存和 Python 环境。请另外传输 RHSEDB.zip 和两份权重，按上述路径放置。该包是联网重建环境包，不是离线安装包。

### 搬运训练完成的模型

完整训练会产生以下三个 checkpoint：

```text
out/SDNet_RHSEDB/model/sdnet_model.pth
out/SegNet_RHSEDB/model/model.pth
out/ExtractNet_RHSEDB/model/model_extract.pth
```

三阶段全部完成后运行：

```bash
bash scripts/pack_trained_models.sh RHSEDB
```

脚本会生成 `dist/StrokeExtraction-models-RHSEDB-时间戳.tar.gz` 及对应 `.sha256`，并整理成推理代码期望的目录：

```text
model/
├── sdnet_model.pth
├── model.pth
└── model_extract.pth
```

把最新服务器代码包、模型包及两个 SHA256 文件传到新电脑。新电脑验证并解压：

```bash
sha256sum -c StrokeExtraction-server-*.tar.gz.sha256
sha256sum -c StrokeExtraction-models-RHSEDB-*.tar.gz.sha256
tar -xzf StrokeExtraction-server-最新时间戳.tar.gz
cd StrokeExtraction
tar -xzf ../StrokeExtraction-models-RHSEDB-最新时间戳.tar.gz
bash scripts/uv.sh sync --locked
```

只做推理不需要搬运 RHSEDB、`dataset_forSegNet_ExtractNet_RHSEDB`、训练图片或 optimizer 状态。当前 checkpoint 保存的是模型权重和 epoch 编号，不包含 optimizer/scheduler 状态，因此适合推理搬运，但不能无损续训。
