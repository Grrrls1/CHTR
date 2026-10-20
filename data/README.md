# CHTR 数据目录与读取工具

数据构建流程、原理、Slurm 命令和中间产物说明统一维护在
[reproduction/README_zh.md](../reproduction/README_zh.md)。本目录保存正式数据集和数据工具源码。

## 保留的数据集

| 目录 | 内容 | 数量 |
| --- | --- | ---: |
| `chtr_mmah_lmdb/` | 完整 LMDB，保留全部清理合格图像 | 73,606 条，其中 71,457 ready、2,149 pending |
| `chtr_mmah_lmdb_1915/` | 冻结的严格质量筛选数据集 | 1,915 条，全部 ready |

1915 数据集是独立完整的 LMDB，不依赖旧备份或 smoke 库。
两个目录均保留 `data.mdb`、`lock.mdb`、`summary.json`、`complete.json` 和 `stroke_type_map.json`。
数据发布使用的 `.chtr_mmah_lmdb.lock` 也保留。

`__ids__` 保存全部记录；全量库的 `__ids_ready__` 列出具有合法完整轨迹的记录。
ready / pending 表示轨迹是否完整合法，质量分数及通过标记另行保存，不能混为同一条件。
1915 库的冻结信息和来源摘要保存在自身 `summary.json` 中。

## 数据接口

图像默认为灰度 256×256。六维轨迹为
`[x,y,width,continue,stroke_end,drawing_end]`，坐标及宽度直径采用像素单位。
`stroke_offsets` 标识语义笔画边界，`subpath_offsets` 与之相等，一笔对应一条连续路径。
类别 ID 为 0–23，必须与本数据集旁置的 `stroke_type_map.json` 匹配。

LMDB 内保存规范化图像、原图、逐笔 mask、轨迹和元数据；pending 样本缺少完整轨迹，
不能供给完整轨迹监督。通用读取器默认跳过 pending，传入 `--allow-pending` 可读取这些记录。

## 读取数据

从工作区根目录运行，使用现有 PyTorch/LMDB 环境：

```bash
CHTR/third_party/StrokeExtraction/.venv/bin/python CHTR/data/chtr_lmdb_dataloader.py \
  --lmdb CHTR/data/chtr_mmah_lmdb --limit 3

CHTR/third_party/StrokeExtraction/.venv/bin/python CHTR/data/chtr_lmdb_dataloader.py \
  --lmdb CHTR/data/chtr_mmah_lmdb_1915 --limit 3
```

读取器支持 `--min-iou`、`--require-quality-pass` 和 `--allow-pending`；默认不按 IoU 筛选。
通用读取器保留变长数组。模型专用的 padding、同步增强、字段合同及两个整字 IoU 筛选条件
由 `../models/datasets/` 实现，使用方法见 [模型说明](../models/README.md)。

## 源码入口

| 文件 | 用途 |
| --- | --- |
| `chtr_lmdb_dataloader.py` | 读取 LMDB 中的图像、轨迹和元数据 |
| `chtr_dataloader.py` | 读取简化格式 manifest 及其外部资产 |
| `pack_chtr_lmdb.py` | 将简化 manifest 与资产封装为新 LMDB |
| `build_chtr_dataset.py`、`run_build_chtr.sh` | 数据构建底层入口，保留 VS 与骨架路线 |
| `continue_skeleton_dataset.py` | 复用完成的 mask 清理结果，分批生成单路径骨架轨迹 |
| `repack_all_clean.py` | 复用骨架批次，保留全部清理合格样本及 pending，封装全量库 |
| `continue_chtr_dataset.py` | 独立 VS 路线的分批续跑工具 |

当前统一全量入口是 [build_full_dataset.slurm](../reproduction/build_full_dataset.slurm)。
重新构建时使用新的运行和输出目录，并遵守已有输入、配置和代码的恢复校验。
