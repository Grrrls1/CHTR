# 柳公权楷书轨迹字典与demo查询

本次数据来源为 **`data/楷-柳公权/`**，不是HCSU子集。
目录有6,200幅静态GIF，覆盖5,818个单字：5,453字一种写法、348字两种、17字三种。
每个汉字目录的名字是索引键，GIF文件名作为写法ID，不自动转换繁简或替换缺失字。

## 转换全部源图片

```bash
/home/lkh/anaconda3/envs/test/bin/python -u models/scripts/build_trajectory_dictionary.py \
  --source 'data/楷-柳公权' \
  --output models/runs/liugongquan_kaishu_library_20261103 \
  --device cuda:0 --warmup 3 --point-batch-size 8
```

使用现有 `ordered_stroke.pt` / `single_stroke.pt`，只加载一次，不进行训练。
图像按 `infer_image.py` 的默认流程解码GIF调色板、转灰度、等比例缩放至长边240、居中填白到256×256。
默认不二值化、不去噪、不增粗，不修改源GIF；需要Otsu时可对独立输出目录加 `--binarize otsu`。
首批优先处理“你好中国柳公权永一人我”，之后继续全部图片，顺序不影响默认写法选择。
输出非空时使用同一命令加 `--resume`，核验源文件及权重SHA256，跳过已提交的完整预测。
`--limit N`只用于小集检查，不是完整字库。

## 输出结构

```text
models/runs/liugongquan_kaishu_library_20261103/
  dictionary.json                 汉字 → 默认写法、全部写法及轨迹/图片/预览相对路径
  summary.json                    全量进度、可用字符数、状态统计、去预热平均推理时间
  run_config.json                 全部源图片/权重指纹及预处理配置
  index.html                      输入单字，选择写法，预览输入图与笔画方向
  characters/<汉字>/
    <写法ID>.json                 完整像素XYW点序列、offsets、停止原因和来源
    <写法ID>.npz                  points、stroke_offsets、image_size
    <写法ID>.input.png            实际256×256输入
    <写法ID>.html                 该写法整字笔序、起终点与方向预览
```

**每个字一个目录，多个写法通过文件名前缀区分，没有额外可视化子目录。**
原始调色板GIF不在字库里重复复制；`input.png`、JSON、NPZ与HTML使字库可独立携带。
可直接用浏览器打开 `index.html`，它不依赖网络请求或远端服务。转换期间刷新页面查看最新进度。

每25张或15秒（当前图片完成后）刷新索引与进度。根summary的 `processed_images=6200` 且 `status=complete`
才表示全部转换完；字库也支持边转换边查询已经完成的字。
完成计数包括需要人工复核的空预测/截断样本，`available_characters`才是有默认可用候选的字数。

默认写法从非空、未触及笔画或点数上限的完整预测中，按源文件名稳定选第一项；不以未知真值的“准确率”挑选。
每幅图均保留预测，即使被标记 `needs_review`；这类写法不自动作为demo默认项。
`converted`仅表示转换完成并通过格式/停止检查，不代表笔顺、漏笔或方向已经正确；`human_reviewed`默认false。

## Python demo：直接用汉字查询，无需GPU和模型

```python
from models.trajectory_dictionary import TrajectoryDictionary

library = TrajectoryDictionary("models/runs/liugongquan_kaishu_library_20261103")
entry = library.lookup("你")

print(entry["variant_id"], entry["coordinates"])
for stroke in entry["strokes"]:
    # stroke 是按书写方向排列的 [x, y, width] 列表。
    # 各stroke是独立落笔段；不要把前一笔末点和后一笔起点当作落笔直线连接。
    print(stroke)

plan = library.lookup_text("你好")
```

查询工具只依赖Python标准库，可将字库目录及 `models/trajectory_dictionary.py` 带到demo电脑，
无需PyTorch、OpenCV或原始数据目录。
默认返回归一化坐标：XY除255，图像直径W除256，X向右、Y向下，不做每笔独立归一化。
`library.lookup("你", normalized=False)`返回256画布像素值。
`variant="源文件ID"`可显式选择同字其他写法，不修改默认项。
`library.available_characters()`返回已完成且有默认候选的字；不存在或未完成的字明确报错，不静默换字。
多字查询先解析所有字符，任一缺失会整体报错，不返回一段被截断的写字计划。

命令行导出demo数据：

```bash
/home/lkh/anaconda3/envs/test/bin/python models/scripts/lookup_trajectory.py \
  --library models/runs/liugongquan_kaishu_library_20261103 \
  --text '你好' --output /tmp/liugongquan_nihao.json
```

`--pixels`改为像素坐标；`--list`列出当前可用字符。输出JSON必须是新文件，避免覆盖已有demo计划。
字库索引保存的是相对路径，整体移动目录不影响查询。

## 机械臂书写接口边界

查询结果是有序的**字符局部图像坐标**，不是机器人运动指令。demo需要决定字的实际尺寸、排版位置和纸面坐标系，
再结合纸面/机器人标定映射到机械臂坐标；根据笔画分段安排抬笔、移到下一笔起点和落笔。
Y向下的图像坐标转换到其他坐标系时应显式处理轴方向。
W是模型预测的图像笔迹直径，不是压力、Z高度或速度；这些需要由书写工具和demo控制策略另行定义。
当前脚本不连接或控制机械臂，也不自动平滑/重采样预测，便于保留模型输出供复核。

## 验证

```bash
/home/lkh/anaconda3/envs/test/bin/python -m unittest models.tests.test_trajectory_dictionary
```

覆盖多写法保留、默认项、归一化、抬笔分段、缺字/异常分段拒绝、整体移动可用性和断点续跑不重载模型。
