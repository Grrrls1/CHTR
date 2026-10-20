# Linux / NVIDIA 本地查看 epoch 100

迁移包解压后直接得到 `CHTR/`，保留服务器上项目的相对目录结构。
包内包含全部项目及第三方源码，以及以下运行资产：

- `models/runs/full_separated_10gb/job_237/`：第 100 轮的 best/last 权重、配置、类别表及训练指标。
- `data/chtr_mmah_lmdb/`：完整库，73,606 条记录，71,457 ready、2,149 pending。
- `data/chtr_mmah_lmdb_1915/`：冻结的 1915 数据集。
- `third_party/StrokeExtraction/out/*_MMAH_CCSEDB_v4/model/`：当前使用的三阶段提取权重。
- `third_party/virtual_sketching/outputs/snapshot/pretrain_clean_line_drawings/`：VS 推理 checkpoint 和配置。

未打包虚拟环境、缓存、Git 元数据、历史实验、第三方训练数据、VGG/ContentNet 训练用权重，
也未重复携带旧 RHSEDB 提取权重。现有三阶段和 VS 权重可用于对应推理，无需重新训练这两个项目。
LMDB 的 `lock.mdb` 不迁移；数据内容与类别表均包含在包内。

## 解压

将压缩包放在希望保存项目的父目录，运行：

```bash
sha256sum -c CHTR_epoch100_linux.tar.gz.sha256
tar -xzf CHTR_epoch100_linux.tar.gz
cd CHTR
```

不要再在已有 `CHTR/` 内直接解压，否则会出现 `CHTR/CHTR/`。
校验文件和压缩包需处于同一目录。包内源码的 `reproduction/LOCAL_EPOCH100.md` 即为本说明。

## 准备 CHTR 评估环境

使用与本机 NVIDIA 驱动兼容的 PyTorch 环境，需要 torch、numpy、opencv-python、lmdb、matplotlib。
CNN 配置不要求 torchvision，运行 CHTR 模型本身也不需要 TensorFlow 1 或第三方训练环境。
模型的通用依赖见 `models/requirements.txt`，损失曲线绘制另需 matplotlib。
服务器虚拟环境没有复制到电脑，应在本机准备相应环境。

## 评估完整库上的 32 张训练图

在 `CHTR/` 根目录、已配置好的评估环境运行：

```bash
python models/scripts/evaluate_memorization.py \
  --checkpoint "$PWD/models/runs/full_separated_10gb/job_237/last.pt" \
  --dataset "$PWD/data/chtr_mmah_lmdb" \
  --device cuda:0 --limit 32 --max-strokes 64 --max-points 1024 \
  --output "$PWD/models/runs/full_separated_10gb/job_237/eval_epoch_100"
```

`--dataset` 覆盖 checkpoint 中的服务器绝对路径，不需要修改权重文件。
以上上限覆盖模型配置的 64 笔、每笔 1024 点。若要对齐早期 1915 实验的旧解码上限，
可改为 `--max-strokes 16 --max-points 512`，但全量样本可能超过这些上限并被标记截断。
每次使用新的输出目录，避免混入之前的结果。

结果包含 `index.html`、`summary.json`、输入/标签/teacher-forced/自由重绘 PNG、
逐笔 mask/type 可视化、预测 NPZ 和训练损失曲线。直接用浏览器打开 `index.html` 查看。
这是训练图拟合检查，teacher-forced 与纯图片自由生成分别报告，不代表泛化评估。

## 使用 1915 库

使用同一个 job237 checkpoint，将上面的数据和输出目录改为：

```text
--dataset "$PWD/data/chtr_mmah_lmdb_1915"
--output "$PWD/models/runs/full_separated_10gb/job_237/eval_epoch_100_1915"
```

这表示在 1915 库上查看 job237 的效果，不是恢复已经清理的旧 1915 拟合实验权重。
单独查看 CHTR 预测只需模型和 LMDB，第三方推理权重不会在此评估过程中加载。

## 在服务器重新打包

在工作区根目录运行只读计划：

```bash
python3 CHTR/reproduction/package_assets.py --datasets both --dry-run
```

实际打包按服务器约定使用 CPU 作业，不申请 GPU：

```bash
mkdir -p logs
gpu-sbatch --user_name lkh CHTR/reproduction/package_assets.slurm \
  --datasets both --output /absolute/path/to/new-package-directory
```

`--datasets full` 或 `--datasets 1915` 可选择只携带一套库；默认 `both`。
包内代码清单由 CHTR 的 Git 跟踪和未忽略源码组成；运行资产使用明确白名单加入。
打包会验证每个归档成员的大小和 SHA-256，并提供归档校验文件及外部 `manifest.json`。
