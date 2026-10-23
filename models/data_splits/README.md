# 字符分层数据划分

`full_character_v1.json` 用于两阶段书法迁移：按 `character` 分层，在字符内部划分整字图片。
默认比例 80% / 10% / 10%、seed=42。满足分组条件的字符在三个集合均有样本；
不足三个独立组的字符全部进入训练集。跨字符重复组保守地放入训练集并报告覆盖例外。
这评估已见字符的新图片泛化，不是字符独立或书家独立验证。

从仓库根目录生成（依赖 Python 3.8+、lmdb、numpy、Pillow）：

```bash
/home/lkh/anaconda3/bin/python models/scripts/create_dataset_split.py \
  --dataset data/chtr_mmah_lmdb \
  --output models/data_splits/full_character_v1.json
```

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
/home/lkh/anaconda3/bin/python -m unittest models.tests.test_dataset_split -v
```
