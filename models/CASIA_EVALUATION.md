# CASIA-OLHWDB 图像到轨迹评估

本入口使用**当前两阶段权重做外部数据集评估**，不训练模型。默认数据为官方
CASIA-OLHWDB1.1 **测试集的中文单字轨迹**；输入图由真实轨迹渲染，但推理不读取真实mask、首点、笔画数或轨迹长度。

## 下载原始在线轨迹

从项目根目录运行：

```bash
/home/lkh/anaconda3/envs/test/bin/python models/scripts/download_casia_olhwdb.py \
  --version 1.1 --output data/CASIA-OLHWDB
```

脚本使用 `wget --continue` 下载原始POT测试包，保留ZIP和 `.part` 续传文件，完成后核验ZIP CRC，
解压到 `data/CASIA-OLHWDB/1.1/test/`。重复执行核验并跳过相同的已有文件，拒绝覆盖不同内容。
`--extract-only` 仅解压已下载ZIP；`--version 1.0` / `1.2` 可选择其他官方测试版本，不下载训练集。

也可只下载，之后调用脚本解压：

```bash
mkdir -p data/CASIA-OLHWDB
wget -c 'https://nlpr.ia.ac.cn/databases/Download/Online/CharData/Pot1.1Test.zip' \
  -P data/CASIA-OLHWDB
/home/lkh/anaconda3/envs/test/bin/python models/scripts/download_casia_olhwdb.py \
  --version 1.1 --output data/CASIA-OLHWDB --extract-only
```

官方原始测试包约47 MB（本次服务器返回48,698,629字节）。不要下载名为OLHWDB1.1tst的512维特征包，
也不要用HWDB的GNT图片包替代POT；前两者不提供本脚本需要的在线点序列。
官方下载页：[Download](https://nlpr.ia.ac.cn/databases/handwriting/Download.html)，
POT格式：[Online Database](https://nlpr.ia.ac.cn/databases/handwriting/Online_database.html)。
使用条件见官方页面的Conditions of Use链接。

## 安装附加指标依赖

本机解释器为Conda `test`，保留已安装的torch/torchvision：

```bash
/home/lkh/anaconda3/envs/test/bin/python -m pip install -r models/requirements-casia-eval.txt
```

附加依赖为 `pytorch-fid==0.3.0`、SciPy、Numba。Numba加速精确DTW；未安装时仍可运行Python实现，
但不适合全量测试。FID必须使用标准预训练Inception权重；第一次计算会下载权重到
`data/CASIA-OLHWDB/.torch/`，不会退回随机权重或其他网络。
下载及已有缓存均校验官方文件名中的SHA256前缀；不匹配的缓存会保留为 `.invalid-*` 后重新获取。
校验失败会停止FID计算，不输出伪造分数；网络中断时可用 `--resume` 补算。

本次开发验证将这些依赖安装在 `/tmp/casia-eval-deps`，没有修改现有训练环境。
如果复用此临时环境，可在后续Python命令前加 `PYTHONPATH=/tmp/casia-eval-deps`；`/tmp`被清理后须重新安装。

## 先检查接入，再计算完整指标

100张小集仅用于检查输入、权重和输出连接，明确跳过FID：

```bash
/home/lkh/anaconda3/envs/test/bin/python -u models/scripts/evaluate_casia_olhwdb.py \
  --pot-root data/CASIA-OLHWDB/1.1/test \
  --output models/runs/casia_olhwdb11_smoke \
  --device cuda:0 --warmup 3 --point-batch-size 8 \
  --line-width 8 --limit 100 --visualize-count 10 --skip-fid
```

全量中文测试，包括mIoU、DTW和标准2048维FID：

```bash
/home/lkh/anaconda3/envs/test/bin/python -u models/scripts/evaluate_casia_olhwdb.py \
  --pot-root data/CASIA-OLHWDB/1.1/test \
  --stroke-checkpoint models/checkpoints/ordered_stroke.pt \
  --point-checkpoint models/checkpoints/single_stroke.pt \
  --output models/runs/casia_olhwdb11_test \
  --device cuda:0 --warmup 3 --point-batch-size 8 \
  --line-width 8 --visualize-count 30
```

`--limit 0`是默认值，表示所有符合筛选的记录；`--limit N`是文件名排序后的前N条中文记录，
不是随机抽样，不能当作全量或书家均衡评估。默认仅保留解码后Unicode U+3400～U+9FFF的字符，
`--include-symbols`才纳入字母、数字、标点等记录。POT文件名作为writer ID。
默认选择POT测试目录，不混入训练数据；脚本无法判断用户提供的目录是否被人为混放，因此记录所有输入文件SHA256。

全量包含数十万字，按单图延迟可能运行很久，并生成每字一个压缩预测文件；运行前检查 `df -h .`。
输出目录必须为空，断点续跑将相同命令加 `--resume`。输入文件、权重和关键配置须一致。
已提交的逐图时间和指标保留，新进程重新预热；不重复推理已提交记录。
如先以 `--skip-fid` 完成某个固定子集，之后可用相同参数、去掉 `--skip-fid` 并加 `--resume` 补算FID，不重新推理。

## 数据与指标口径

POT提供字符GB标签、真实笔画数和逐笔XY点序列，使用 `(-1,0)` 分隔笔画、`(-1,-1)`结束字符。
解析器检查小端记录长度、坐标和笔画数，保留重复点与原始点序，不做平滑、简化、重排或DTW前重采样。
POT不提供本入口所需的真实笔宽、压力和逐点时间，不能计算这些物理量的真值误差。

归一化：整字所有轨迹点共享一个等比例变换，最长坐标跨度映射为239像素，包围框居中于256×256画布，默认边距8。
窄字不会在水平方向单独拉伸。每图保存原始包围框、缩放比例和偏移量。
输入为固定直径8像素的黑字白底渲染，除255后送入模型。这是本地256分辨率协议，不等价于其他论文的64分辨率或其他渲染方案。

| 输出字段 | 定义 | 越大/小越好 |
| --- | --- | --- |
| `metrics.miou` | 逐字前景IoU的算术平均；真实轨迹和预测XY均用相同固定宽度8渲染 | 大 |
| `metrics.binary_class_miou` | 每字前景/背景两类IoU均值，再跨字平均 | 大 |
| `metrics.miou_predicted_width` | 使用模型预测W渲染，与固定宽度的真实轨迹渲染比较；作为辅助项 | 大 |
| `metrics.dtw_sum_px_mean` | 每字精确DTW最小累积欧氏XY距离，之后跨字平均，单位为256画布像素 | 小 |
| `metrics.dtw_path_mean_px_mean` | 同一最优路径的累积距离/路径长度，之后跨字平均；**不是PEN-Net的LDTW** | 小 |
| `fid.value` | 全部所选真实/预测渲染图的2048维FID | 小 |

主mIoU和FID使用相同固定宽度，避免把模型预测直径和没有真实宽度的POT强行作监督比较。
固定宽度会改变指标，跨实验必须冻结；它与自适应选择宽度的AIoU不同。
DTW按笔顺拼接各笔的XY点，保留方向，但不向序列插入抬笔坐标，也不跨笔画绘制连接线。
因此DTW不能单独判断笔画切分正确性，另保存真实/预测笔数、笔数MAE和截断率。
空预测的IoU按0计（GT非空）；空序列DTW未定义，记为null并单列 `dtw_undefined_images`，绝不作为0混入均值。

FID不是逐字指标。它使用[pytorch-fid](https://github.com/mseitzer/pytorch-fid)的预训练FID InceptionV3，
灰度黑白图重复为RGB，使用该实现默认的299×299缩放与输入归一化。
特征均值和协方差按float64流式合并，内存不随样本数线性增加，最后使用该库的Fréchet距离实现。
少于2张无法计算，样本数不大于2048时会记录小样本标记；小集FID不应与大集FID直接比较。
只改用普通ImageNet Inception权重、改特征维度或换字符识别器，都不再是这里定义的标准FID。

当前权重并非本次在CASIA训练得到，结果应注明“外部数据集评估”；在划分、输入渲染、轨迹采样和指标实现没有对齐前，不能宣称与论文结果直接可比。

## 输出与进度

```text
models/runs/casia_olhwdb11_test/
  run_config.json                    数据/权重SHA256、渲染、筛选和指标定义
  summary.json                       mIoU、DTW、FID、停止/笔数与去预热耗时汇总
  progress.json                      推理阶段的轻量进度与当前前景mIoU
  per_image.jsonl                    每字记录，逐条提交，支持断点续跑
  per_image.csv                      推理阶段完成后生成的逐字表
  predictions/<POT文件相对路径>/<记录索引>_<GB标签>.npz
                                    GT/预测点与分段、固定宽度参考/预测mask
  examples/<POT文件相对路径>/<记录索引>_<GB标签>/
    input.png / fixed_width_reconstruction.png
    prediction.npz / summary.json
    overview.png / directions.png / strokes.png
  fid_real_stats.npz / fid_prediction_stats.npz
```

同一个字符在不同书家或记录中有独立目录/文件，不按汉字名称合并，避免重名覆盖。
默认只为前30个样本生成完整可视化，所有样本均保存压缩轨迹及逐字指标。
总览图的彩色宽度重建展示模型W；`fixed_width_reconstruction.png`才是主mIoU/FID使用的固定宽度渲染。
推理计时采用设备同步，不含模型加载、预热、输入渲染/传输、CPU指标、DTW编译、保存、可视化和后续FID。
完整summary默认每1000张更新；progress.json和日志每15秒或100张更新（完成当前图后检查）。
`status=computing_fid`表示推理已结束但FID仍在计算；`complete`才表示所选流程结束。
如显式跳过FID，`fid.status=explicitly_skipped`，不会填充一个虚构分数。

## 验证

```bash
/home/lkh/anaconda3/envs/test/bin/python -m unittest models.tests.test_casia_evaluation
```

测试覆盖官方GB字节序、ASCII标签、笔画分隔、截断文件、等比例坐标变换、DTW顺序/数值、
流式协方差，以及使用真实网络的合成POT输入到逐字产物、预热和断点续跑。
开发时另核验了官方ZIP首个真实POT文件；完整模型指标需要下载数据并运行上述评估命令，不能以合成测试结果代替。
