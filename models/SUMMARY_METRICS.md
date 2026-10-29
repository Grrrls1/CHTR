# 两阶段评估 summary.json 字段说明

适用于笔画层训练期间val及 `scripts/evaluate_ordered_stroke.py` 的val/test，
协议名为 `ordered_masks_to_point_trajectories_v2`。
实际流程：规范化整字 → 笔画层预测mask/首点 → 冻结点层生成轨迹 → 对照GT计算指标。
本说明中的示例值固定来自job679的epoch10完整val，不是随训练自动更新的实时值。

原始示例：[epoch10 summary.json](runs/ordered_stroke_full/job_679/train/eval_epoch_010/summary.json)。
训练与评估命令见 [ORDERED_STROKE](ORDERED_STROKE.md)。

## 读数约定

- IoU、召回率、正确率、终止率在JSON中存为0～1；例如0.616303表示61.6303%。
- 名称带 `_px` 的几何误差为256×256规范化画布上的像素值；width是完整宽度/直径。
- MAE、L2与loss通常越低越好；IoU、召回率与正确率通常越高越好。
- bias为有符号偏差：负值表示预测偏细，正值表示偏粗；理想值接近0。
- `aligned`只是按笔画序号配对，不表示已找到语义正确的对应笔画。漏笔后可能发生错配。
- 正常终止只说明模型主动停止，不说明停止位置、笔数或路径正确。

## 顶层结构与运行条件

| 字段 | 含义／示例 |
| --- | --- |
| `split` | 本次集合，`val`或`test`等；示例为val |
| `characters` | 实际评估字图数；8,121 |
| `strokes` | 这些字图中GT笔画总数；83,119，不是预测数 |
| `full_split` | 是否覆盖指定集合全部样本；true |
| `dataset_fingerprint` | 候选数据元信息指纹，用于核对数据版本 |
| `checkpoint.epoch` / `checkpoint.step` | 本次笔画层权重的训练位置；10 / 70,780 |
| `checkpoint.path` | 独立评估时记录的笔画层checkpoint路径；训练内验证可能没有此字段 |
| `point_checkpoint.path` | 实际加载的点层checkpoint路径；job605的best.pt |
| `point_checkpoint.epoch` / `point_checkpoint.step` | 点层训练位置；20 / 120,100 |
| `selection_metric` | 选best依据；`free.ordered_mask_iou`，不是点层整字IoU |
| `evaluation_protocol` | 评估口径版本；`ordered_masks_to_point_trajectories_v2` |
| `geometry_alignment` | 几何配对规则说明；按序号，缺失/额外笔画的按序IoU记0 |
| `validity_threshold` | 笔画层有效性阈值；0.5，低于它停止整字 |
| `mask_threshold` | mask前景阈值；0.5 |
| `decode_max_strokes` | 每字笔画保护上限；64 |
| `point_validity_threshold` | 点层有效性阈值；0.5，低于它停止当前笔 |
| `point_decode_max_points` | 单笔点数保护上限；400 |
| `visualization_ids` | 已保存图库的目录标识列表，不是评估全集；当前为sample_id的哈希目录名 |
| `visualized_characters` | 实际输出可视化字图数；30，不影响评估总量 |
| `test_used_for_model_selection` | test是否用于选模；当前false |

顶层 `teacher_forced` 等于 `stroke_layer.teacher_forced`；顶层 `free` 等于
`stroke_layer.free`。这是兼容别名，不是额外两组指标。

## stroke_layer.teacher_forced：真实历史条件下的笔画层

历史笔画mask来自GT，只衡量正确历史条件下下一笔预测能力，不代表自由生成效果。

| 字段 | epoch10示例 | 定义 |
| --- | ---: | --- |
| `cls` | 0.007134 | 笔画有效性BCE；K个有效位置及末尾invalid参与，padding不参与 |
| `mask` | 0.030945 | 有效笔画的像素BCE；logits上采样到256×256后计算 |
| `dice` | 0.146584 | 有效笔画的Dice损失，平滑常数1 |
| `box` | 0.021877 | 归一化xyxy框的四坐标平均L1 |
| `giou` | 0.280241 | 框损失 `1-GIoU` |
| `start` | 0.029593 | 归一化首点XY的坐标平均L1 |
| `width` | 0.016452 | 归一化首点直径的Charbonnier，epsilon=0.01/256且减去偏置 |
| `auxiliary` | 1.942215 | 中间decoder层加权损失的平均；9层结构使用前8层 |
| `loss` | 3.692680 | 最后一层加权损失，加上配置中的辅助权重乘auxiliary |
| `mask_iou` | 0.798989 | 有效笔画mask阈值化后与GT的IoU；先每字平均再跨字平均 |
| `valid_stroke_recall` | 99.8941% | 全部GT有效位置中，被预测为有效的比例 |
| `terminal_recall` | 98.9410% | 每字真实终止位置被预测为invalid的比例 |

基础loss项是未乘系数的最后一层值。当前最终层加权公式：

```text
4*cls + 5*mask + 5*dice + 5*box + 2*giou + 5*start + 1*width
```

总loss再加 `auxiliary_weight*auxiliary`，当前辅助权重1。
有效性loss先对每字K+1个有效监督槽平均；其余loss只对该字K个真实笔画平均，再对字平均。
没有终点loss或type分类loss。mask BCE背景像素较多，数值低不能单独证明前景完整或连通。

## stroke_layer.free：笔画层自主生成

历史使用自身预测mask，按顺序比较，不使用最优匹配。
设某字真实N笔、预测M笔：

```text
ordered_mask_iou = sum(IoU(GT[k], Pred[k]), k < min(N,M)) / max(N,M)
```

GT字至少有一笔；无预测时该字按序IoU为0。

| 字段 | epoch10示例 | 定义 |
| --- | ---: | --- |
| `ordered_mask_iou` | 0.652484 | 上述按序mask IoU，先逐字计算再对字平均 |
| `strict_sequence_mask50` | 47.3833% | N=M且所有同序mask IoU≥0.5的字占比；不是最优匹配后的Order Acc. |
| `count_exact` | 61.6303% | N=M的字占比；不能保证笔画内容、身份或顺序正确 |
| `stroke_count_mae` | 0.734146笔/字 | 每字 `abs(M-N)` 的平均 |
| `aligned_start_xy_mae_px` | 12.726 | 同序首点的XY平均绝对误差；不是欧氏距离 |
| `aligned_start_l2_px` | 19.833 | 同序首点的欧氏距离 |
| `aligned_start_width_mae_px` | 4.141 | 同序首点的宽度直径MAE |
| `aligned_start_width_bias_px` | −0.717 | 同序首点宽度的 `预测-GT` 平均偏差 |
| `aligned_strokes` | 77,839 | 各字 `min(N,M)` 的总和，是参与同序几何比较的数量 |
| `predicted_strokes` | 78,521 | 全部字预测笔画总数，即M之和 |
| `empty_masks` | 0 | 所有预测笔画中，0.5阈值化后没有前景像素的数量 |
| `terminated` | 100% | 自主invalid结束的字占比；这里是比例，不是数量 |
| `truncated` | 0% | 因笔画保护上限截断的字占比 |

首点几何误差按所有参与比较的同序笔画平均。没有任何共同序号时，这些误差为null。
`empty_masks=0`不保证单连通、无碎片或无孔洞；当前未做连通性清理。

## point_layer：预测mask/首点驱动的完整串联轨迹

点层权重冻结，自由生成只读取预测mask提取的原图和预测首点，不读取GT后续点或GT长度。
每个有效笔画实例生成一条轨迹，mask的多个连通域不自动拆成多条轨迹。

| 字段 | epoch10示例 | 定义 |
| --- | ---: | --- |
| `character_trajectory_iou` | 0.780681 | GT所有笔画重绘并集与预测所有笔画重绘并集的IoU，再对字平均；不检查笔顺 |
| `ordered_trajectory_iou` | 0.565632 | 同序轨迹重绘IoU求和除以max(N,M)，再对字平均；缺失/额外位置记0 |
| `aligned_xy_mae_px` | 12.977 | 同序轨迹按各自点序归一化进度重采样后的XY平均绝对误差 |
| `aligned_width_mae_px` | 3.595 | 同样重采样后的宽度直径MAE |
| `aligned_width_bias_px` | −0.544 | 同样重采样后的预测减GT宽度偏差 |
| `aligned_endpoint_l2_px` | 26.644 | 同序轨迹最后一个点的XY欧氏距离 |
| `aligned_point_count_mae` | 16.542点 | 同序轨迹 `abs(预测点数-GT点数)` 的平均 |
| `aligned_exact_point_count_rate` | 4.0134% | 同序轨迹点数完全相同的比例，不保证点坐标正确 |
| `aligned_strokes` | 77,839 | 参与几何比较的共同序号笔画数 |
| `predicted_strokes` | 78,521 | 点层生成轨迹数，对应笔画层有效实例数 |
| `missing_strokes` | 5,280 | 各字 `max(N-M,0)` 的总和，仅按笔数差计算 |
| `extra_strokes` | 682 | 各字 `max(M-N,0)` 的总和，仅按笔数差计算 |
| `terminated` | 78,515 | 点层主动invalid停止的笔画数量；这里是数量，不是比例 |
| `truncated` | 6 | 达到点数保护上限的笔画数量 |
| `terminated_rate` | 99.9924% | terminated / predicted_strokes |
| `truncated_rate` | 0.00764% | truncated / predicted_strokes |
| `condition` | 说明字符串 | 预测mask及首点作为输入，点层冻结 |
| `alignment` | 说明字符串 | 按笔画序号比较；aligned误差不计缺失/额外笔画，而按序IoU对它们记0 |
| `resampling` | 说明字符串 | 等序列进度重采样，包含预测首点，不是DTW |

重采样点数取两条轨迹点数较大值，至少2点。不同字的aligned误差按参与比较的笔画数量加权，
相当于每条共同序号笔画等权；并非全库所有轨迹点直接混在一起平均。
无共同笔画时aligned误差为null；无预测笔画时点层停止/截断率为null。
`missing_strokes/extra_strokes`不是基于语义匹配的漏检/误检计数：同字少一笔又多一笔，可能M=N但内容错误。

## 怎样解读示例

- TF mask IoU为0.799，自由mask IoU为0.652，提示真实历史与预测历史之间存在误差传播。
- 正常整字终止率100%，笔数正确率61.63%，说明主动结束不等于在正确笔数处结束。
- 整字轨迹并集IoU为0.781，按序轨迹IoU为0.566；形状覆盖较接近不代表分笔、顺序和路径正确。
- 首点L2误差19.83px、串联端点误差26.64px；mask清理之外还需检查首点及错配影响。
- 点层截断仅6笔，不能据此认定其他笔画的收笔位置正确。
- 这些是预测条件下的串联结果，不能与job605真实mask/首点条件下的独立点层指标直接横比。

本轮是val，不是test；只用于可视化的30字不能代表全体样本分布。
建议同时查看逐字 `characters.jsonl`，按笔数及失败类型分组分析，不只看整体平均。

## 与独立点层summary的区别

`evaluate_single_stroke.py`产生的summary使用 `teacher_forced` 和 `free` 描述点层，
条件为真实mask与真实首点；本文件的同名顶层字段描述笔画层。
独立点层的 `resampled_*` 误差排除给定首点；串联 `aligned_*` 包含预测首点。
两种summary含义应通过 `condition` / `evaluation_protocol` 与入口区分。

实现依据：[笔画层评估](training/ordered_stroke_evaluation.py)、[串联轨迹指标](training/two_stage_metrics.py)、
[笔画层损失](losses/ordered_stroke.py)。
