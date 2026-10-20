# 修改版第三方源码

本目录直接纳入当前工作区的两个修改版项目，不使用 submodule：

| 目录 | 上游 | 基线提交 |
| --- | --- | --- |
| `StrokeExtraction/` | https://github.com/MengLi-l1/StrokeExtraction | `2d273e0eaa1df448aa5354375bb1a0a9c131f66d` |
| `virtual_sketching/` | https://github.com/MarkMoHR/virtual_sketching | `41d0fd9db2eb9603ecc314bc1ece72d554271bda` |

本地数据加载、训练、推理及 TF1 兼容修改均随目录保留。移动前的源码哈希及来源记录在
[migration_20261020.json](migration_20261020.json) 中。
原项目的 `.git` 元数据移至工作区 `archive/third_party_git_20261020/`，没有删除 Git 历史。
因此 CHTR 可直接跟踪这些源码，不会把它们误加成只含提交指针的嵌套仓库。

本机权重、数据和环境也随原目录整体移动；根目录 `.gitignore` 排除这些运行资产，
它们不会因本次整理而自动成为上传内容。

## 路径与环境

新的代码默认从 `CHTR/third_party/` 加载依赖。工作区原位置保留兼容软链接：

```text
workspace/StrokeExtraction -> CHTR/third_party/StrokeExtraction
workspace/virtual_sketching -> CHTR/third_party/virtual_sketching
```

旧数据清单、历史配置、环境中的绝对路径和已经提交的作业可能仍引用这些位置。
保留软链接可继续读取这些资产，不改写历史内容指纹。
重新检出仓库后应根据各项目依赖文件重建环境，不应将本机 `.venv` 当作可分发环境。

PyTorch 环境默认使用 `StrokeExtraction/.venv/bin/python`。
Virtual Sketching 使用独立 Python 3.7 / TensorFlow 1 环境；`run_cpu.sh` 优先查找
`CHTR/.venvs/virtual-sketching/`，其次使用工作区原 `.venvs/virtual-sketching/`，
也支持 `VS_PYTHON` 显式指定。

## 许可证记录

Virtual Sketching 的原 Apache-2.0 许可证保留在其 `LICENSE` 中。
所检查的 StrokeExtraction 树未发现 LICENSE，公开再分发授权仍待核实。
CHTR 根目录的许可证不自动覆盖许可未明确的第三方代码或外部数据。
