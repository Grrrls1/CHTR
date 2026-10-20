# 本机原版 TF1 CPU 环境

批量处理更新（2026-09-15）：新增 [vectorize_strokes_batch.py](../CHTR/reproduction/pipeline/vectorize_strokes_batch.py)，模型常驻、多个随机候选、内容哈希恢复，并调用上游工具生成可视化。完整接入和审核命令见 [流水线脚本说明](../CHTR/reproduction/pipeline/README_zh.md)。1613 的全部 7 笔已使用新入口跑通；后续连续点轨迹的重绘质量仍未通过 0.95 门槛。

环境路径：`/home/lkh/workspace/.venvs/virtual-sketching`。采用 Python 3.7.12、TensorFlow 1.15.0 CPU 包，保留原网络、`tf.contrib` 和官方 checkpoint，不依赖 CUDA。固定依赖见 `requirements-cpu-tf1.txt`。

## 重建环境

```bash
conda create -p /home/lkh/workspace/.venvs/virtual-sketching --override-channels -c conda-forge python=3.7 pip -y
/home/lkh/workspace/.venvs/virtual-sketching/bin/python -m pip install -r /home/lkh/workspace/virtual_sketching/requirements-cpu-tf1.txt
```

旧版 TensorFlow 使用较旧的 NumPy 和 Protobuf API，不要单独升级这些依赖。

## 官方模型

下载来源为作者 README 链接下 `pretrained_models/used_for_testing/pretrain_clean_line_drawings.zip`，Google Drive 文件 ID：`16q8Z8vi5ZZwNV1mo9HXD_ap5nomzij7R`。

模型已放置于：

```text
outputs/snapshot/pretrain_clean_line_drawings/
  checkpoint
  model_config.json
  p2s-75000.index
  p2s-75000.data-00000-of-00001
```

## 运行

```bash
cd /home/lkh/workspace/virtual_sketching
bash run_cpu.sh --input muten.png --sample 1
```

无需激活 conda。脚本使用上述独立环境，清空外部 PYTHONPATH 并采用非交互图像后端，防止 ROS / 用户包影响原环境。可用 `VS_PYTHON` 指定另一个兼容环境的 Python。

输入实际位于 `sample_inputs/clean_line_drawings/`，不是直接放在 `sample_inputs/` 根目录。自己的图片也放在这个子目录：白底黑笔画，灰度或 RGB PNG，无透明背景。建议保持 256×256 整字画布。

输出位于 `outputs/sampling/clean_line_drawings__pretrain_clean_line_drawings/`，原始轨迹在其 `seq_data/` 中。输出为局部 Bézier 参数，不是最终 CHTR 六维点序列。`--sample 5` 生成五个候选，CPU 用时会增加；默认使用随机初始位置，重复结果可能不同。

当前脚本仅配置矢量化推理；重新训练还需要另外下载训练数据、神经渲染器和感知模型。

## 已完成验证（2026-09-10）

- 隔离 ROS 的 `PYTHONPATH` 后，`pip check` 通过。
- 上述 `muten.png --sample 1` 命令成功退出，加载官方 `p2s-75000`。TensorFlow 只初始化 Host/CPU；程序打印的 `Model using gpu.` 是上游固定日志，不代表实际使用了 GPU。
- 输入为 1024×1024；本次得到 `(858,7)` 曲线参数，8 轮，模型报告采样时间约 16.32 秒（不含环境启动、建图和渲染）。随机初始化会改变后续运行的曲线数及耗时。
- NPZ 字段、有限数值和轮次长度检查通过；生成的 1024×1024 渲染图已检查。
- 结果：[渲染图](outputs/sampling/clean_line_drawings__pretrain_clean_line_drawings/muten_0_pred.png)、[轨迹 NPZ](outputs/sampling/clean_line_drawings__pretrain_clean_line_drawings/seq_data/muten_0.npz)。顺序图在同级 `order/` 和 `order-compare/`。
- 已对 CHTR 适配器导出的 256×256 `1613_stroke000.png` 做隔离冒烟测试：生成 `(43,7)` 的有限数值轨迹、1 个绘制轮次，采样约 0.64 秒。重绘主体正常，但可见孤立小点，因此仍需 IoU 和人工质检。
- 完整安装版本已保存为 `requirements-cpu-tf1.lock.txt`，需要完全固定间接依赖时使用该文件重建。

这验证了本机官方样例的推理链路；汉字单笔画的轮廓质量、IoU 筛选以及 CHTR 数据封装仍需后续实现和验证。
