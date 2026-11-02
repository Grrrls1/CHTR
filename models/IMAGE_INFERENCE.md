# 普通图片与相机照片的两阶段轨迹推理

入口：[scripts/infer_image.py](scripts/infer_image.py)。
它将预处理后的单字图片交给现有 `infer_two_stage.py`，不修改模型结构或权重。
此前 `test_images_20261103/inputs/` 的预处理是临时执行的；本入口已核验对四张 `data/test/*.jpg` 产生完全相同的256×256输入。

## 单张图：预处理、推理和可视化

在项目根目录运行：

```bash
/home/lkh/anaconda3/envs/test/bin/python models/scripts/infer_image.py \
  --image data/test/36777.jpg \
  --output models/runs/photo_example \
  --device cuda:0
```

默认处理为：EXIF方向校正 → 透明区域合成白底 → 灰度化 → 等比例缩放到长边240 → 居中填白到256×256。
缩小用INTER_AREA，放大用INTER_LINEAR；宽高取整数时存在不足一个像素的舍入，不将字拉伸到正方形。
不自动裁剪墨迹边界，不去噪，不增粗，不默认二值化，原始文件保持不变。

输出目录必须为空或不存在，每张照片一个目录，直接包含：

- `input.png`：实际送入模型的256×256图。
- `preprocess.json`：原始尺寸、裁剪区域、缩放填白、阈值和正逆变换矩阵。
- `prediction.npz`：模型画布上的XYW、笔画offsets、mask、首点和框。
- `prediction_source_xy.npz`：逆映射到照片上的 `points_xy` 和 `stroke_offsets`。
- `summary.json`：原图/输入路径、模型信息、停止原因及耗时。
- `index.html`：整字覆盖、笔顺、起终点、方向和可播放轨迹，不另建可视化子目录。

`prediction_source_xy.npz` 的坐标属于**EXIF方向校正后的原图像素坐标**，不是原始传感器坐标或机器人坐标。
不自动裁剪越界预测；预测点可能位于原始ROI之外。
该文件只映射XY，不把模型W转换成真实笔宽、压力或机器人末端姿态。

原有 `timing_seconds.inference` 仍是单次、无预热的两阶段耗时，不含加载和预处理。
`pipeline_seconds` 另记录预处理、子进程（包含解释器启动、加载、推理和可视化）及流水线总耗时。

## 单字区域与可选处理

相机整帧中若包含桌面、工具或多个字，应先给定单字ROI。
使用 `--crop X Y WIDTH HEIGHT`，单位为方向校正后的原图像素；在该ROI内等比例缩放和填白。
目前不自动定位或识别字符，因此相机照片需已对准单字，或提供有效ROI。

- `--binarize otsu`：缩放后进行Otsu黑白二值化；默认 `none` 保留灰度。
- `--invert`：显式反色，用于白字黑底；默认要求黑字白底。
- `--margin 8`：控制长边留白，长边为 `256-2*margin`。
- `--resize pad-only`：不缩放，仅居中填白；图像须能放进扣除margin后的区域。已是256×256时可加 `--margin 0`。
- `--preprocess-only`：只输出 `input.png` 和 `preprocess.json`，不加载模型，便于调整相机ROI及检查笔迹。

例如只检查预处理输入：

```bash
/home/lkh/anaconda3/envs/test/bin/python models/scripts/infer_image.py \
  --image data/test/36777.jpg \
  --output models/runs/photo_preprocess_check \
  --preprocess-only
```

二值化不保证补回浅色或破碎笔迹；阴影、反光也可能被保留成前景。应先检查 `input.png`，确认字的结构没有被预处理破坏。

## 腕部相机接入边界

当前实现从**已保存的单张照片**开始，尚未接入相机采集、连续帧处理或自动单字检测。
在相机近似正对纸面、完整拍到一个字、图像清晰且光照均匀时，可直接用上述入口生成轨迹。
明显倾斜的照片需要先做纸面透视校正；镜头畸变明显时需要先去畸变，这些步骤当前没有自动完成。

后续完整链路可按以下顺序接入：

```text
腕部相机采集
  → 去畸变 / 纸面透视校正 / 单字ROI
  → 本脚本预处理
  → 两阶段轨迹推理
  → 纸面坐标与机器人坐标映射
```

若用于机器人实际书写，还需相机标定及纸面/机器人坐标映射；腕部相机位姿变化时，映射需要与该帧位姿对应。
当前脚本只导出图像轨迹，不发送运动指令；W也不是落笔压力。
逐张调用会重复加载模型；若之后需要连续拍照，应将模型加载和推理封装到常驻进程，复用权重并单独预热。
