# 字符分层数据划分

`full_character_v1.json` 用于两阶段书法迁移：按 `character` 分层，在字符内部划分整字图片。
默认比例 80% / 10% / 10%、seed=42。满足分组条件的字符在三个集合均有样本；
不足三个独立组的字符全部进入训练集。跨字符重复组保守地放入训练集并报告覆盖例外。
这评估已见字符的新图片泛化，不是字符独立或书家独立验证。

2026-10-26 已使用 `third_party/StrokeExtraction/.venv/bin/python` 核验本机 LMDB，
与现有 v1 清单的候选元数据指纹一致；ID互斥、覆盖、重复组隔离、字符分布及展开笔画数均通过。
直接复用此清单，未重新抽样或拆分 LMDB。结果见
[本机核验记录](../reports/local_dataset_split_20261026.json)。

| 集合 | 整字图片数 | 展开笔画数 |
| --- | ---: | ---: |
| train | 56,622 | 576,465 |
| val | 8,121 | 83,119 |
| test | 6,714 | 68,413 |

从 `/home/lkh/workspace/CHTR` 执行（依赖 Python 3.8+、lmdb、numpy、Pillow）。
已有清单应先核验后复用；仅数据或规则变化时生成新版本：

```bash
cd /home/lkh/workspace/CHTR
/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python models/scripts/create_dataset_split.py \
  --dataset data/chtr_mmah_lmdb \
  --output models/data_splits/full_character_v2.json
```

上例仅用于确需重建时生成新版本；生成后须显式更新两阶段配置的 `split_file`，不要混用版本。

本机复用清单前，可只读核验当前 LMDB 元数据与清单是否一致：

```bash
cd /home/lkh/workspace/CHTR
third_party/StrokeExtraction/.venv/bin/python - <<'PYTHON'
import json
from models.datasets.dataset_split import read_candidates, validate_split
with open('models/data_splits/full_character_v1.json', encoding='utf-8') as f:
    split = json.load(f)
rows, selection = read_candidates('data/chtr_mmah_lmdb', **split['filters'])
validate_split(split, rows)
print('PASS', selection, split['summary'])
PYTHON
```

这是 ID 清单划分：原始 LMDB 保持一份，训练和评估入口通过集合名选取样本，
无需额外创建 train / val / test 数据库目录。训练配置
`models/configs/single_stroke.json` 和 `single_stroke_smoke.json` 已指向此清单。
若指纹不匹配，先确认数据版本与筛选条件；不要为了通过校验而手改指纹或覆盖既有划分。

脚本只读 LMDB，已有输出拒绝覆盖。默认使用所有 ready 样本，不追加 IoU 筛选，
不代表这些样本已人工验收；可通过 `--min-trajectory-iou`、`--min-mask-iou` 指定筛选，
产生新的版本清单。迁移训练不得在划分后自行筛掉样本而继续宣称原字符覆盖成立。
数据或筛选规则变化应重新生成并冻结版本，不能调参后反复重抽测试集。

分组使用原图来源路径、规范化图及原图的实际灰度像素 SHA256，并使用规范化图
64-bit dHash、Hamming 距离 <= 2 进行近重复启发式检查，传递闭包内样本不跨集合。
它不能保证发现所有裁剪、旋转或其他近重复，也可能保守地合并相似字形；训练前应抽查重复组。
整个数据来源集合不是独立书写实例，不能将其作为一个不可拆分组；现有作者信息不足。
指纹覆盖候选样本完整元数据（包含已有资产哈希），加载不重新读取全部轨迹和 mask 校验内容，
不替代后续资产质量审核。

新两阶段 Dataset 使用 `models.datasets.dataset_split.load_split_ids(path, rows, name)`，
其中 `rows` 必须是通过相同筛选条件读取的完整候选元数据，`name` 为 `train`、`val` 或 `test`。
该接口校验指纹、ID、互斥、覆盖、分组和字符分布，然后返回指定集合 ID。
两阶段共用同一清单，单笔阶段在集合内部展开 `(sample_id, stroke_index)`。
训练集启用增强；验证和测试固定预处理。验证集负责调参、checkpoint 和阈值选择；
测试集仅用于最终验收。单笔阶段已通过共享元数据读取与validate_split接入清单，
见 [单笔阶段说明](../SINGLE_STROKE.md)；有序笔画阶段后续共用同一清单。
旧训练入口及 job237 不作改动，job237 已见图不能变为其独立测试集。

运行验证：

```bash
cd /home/lkh/workspace/CHTR
/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python -m unittest models.tests.test_dataset_split -v
```
