# CHTR 待测数据清单

用于补齐论文、PPT 和演讲稿。权重与数据均在服务器上，返回实测结果即可。尚未完成的项目填“待测”，当前无法开展或未实现的项目填 N/A 并说明原因。

回填日期：2026-11-13。以下来自已有原始评估文件，本次未重跑评估。第一、二部分共 30 个指标单元格，已有 14 个实测数值（书法 test 10 项、CASIA val 4 项）；G-HTR 的 11 项标记为 N/A，不计入实测完成项。

## 一、优先：书法测试集结果与比较限制

本模型 XYW 已完成书法测试集评估。按用户提供的信息，G-HTR 的代码与数据集尚未公开，目前无法开展同条件性能比较，因此该列统一标记 N/A（不可获得同条件实测结果），不再列为当前待测任务。G-HTR 仅保留方法层面的相关工作讨论，不据此声称本模型优于或劣于 G-HTR。

| 指标 | 本模型 XYW | G-HTR |
|---|---|---|
| 整字轨迹 IoU（CharIoU）↑ | 0.806810 | N/A |
| 按序轨迹 IoU（OrdIoU）↑ | 0.627623 | N/A |
| XY 坐标 MAE（px）↓ | 10.3582 | N/A |
| 宽度 MAE（px）↓ | 3.2166 | N/A |
| 宽度有符号偏差（px） | -0.3204 | N/A |
| 终点距离 L2（px）↓ | 21.5022 | N/A |
| 笔画数正确率（%）↑ | 72.1478 | N/A |
| 笔画数 MAE↓ | 0.489574 | N/A |
| 严格序列掩膜正确率（%）↑ | 55.3619 | N/A |
| 生成截断率（%）↓ | 笔画层 0%；点层 0.007538% | N/A |
| 单字推理时间（ms）↓ | 待测 | N/A |

本模型评估条件与来源：

- 数据：`data/chtr_mmah_lmdb`，`full_character_v1` 的 test，完整 6,714 个字符样本、68,413 笔画，`full_split=true`；测试集未参与模型选择。
- 权重：笔画层 `job_679/train/ordered_stroke.pt`（epoch 20，step 141560）；点层 `job_605/train/single_stroke.pt`（epoch 20，step 120100）。
- 画布：256×256；协议：`ordered_masks_to_point_trajectories_v2`；预测掩膜、预测首点，点模型冻结。XY/宽度/终点误差按序号配对笔画统计，排除未配对笔画；OrdIoU 对缺失/多余笔画有数量惩罚。
- 截断口径：笔画层按字符统计；点层按预测轨迹统计，5/66,332 笔，二者不是同一个分母。
- [原始 summary](../runs/ordered_stroke_full/job_679/test_job_877/summary.json)、[逐字结果](../runs/ordered_stroke_full/job_679/test_job_877/characters.jsonl)、[30 个样例画廊](../runs/ordered_stroke_full/job_679/test_job_877/index.html)。

已有验证集分数不代替测试集分数；当前没有与本轮测试配套的单字推理计时。

## 二、手写：CASIA-OLHWDB1.1

本模型已有**完整两阶段、纯图像输入**的验证集结果，回填如下。**本表为 val，不是官方 test 验收成绩；官方 test 仍待测。**

| 指标 | 本模型 |
|---|---|
| AIoU ↑ | 0.769386 |
| LPIPS ↓ | N/A（未实现） |
| LDTW ↓ | 14.5594 px |
| DTW ↓ | 4104.4316 |
| 笔画 Mask AP / AP50 ↑ | N/A（未实现 COCO Mask AP） |
| 顺序正确率（Order Acc.，%）↑ | N/A（未实现同口径实例匹配指标） |
| 笔画数正确率（%）↑ | 58.9005 |
| 单字推理时间（ms）↓ | 待测 |

评估条件与来源：

- CASIA-OLHWDB1.1 官方 train 内按 writer 划分的 val，完整 89,877 个字符样本、519,116 笔画，`full_split=true`。
- 笔画层：`job_1597/train/best.pt`，对应已完成的 epoch 6、step 151632；点层：`frozen_points/job1136_epoch004_step1120224.pt`，epoch 4、step 1120224。
- 256×256 画布，保留原始全点，不做 RDP；验证渲染宽度为每字恒定 8px，W 为合成标签。
- 两阶段串联协议为 `ordered_masks_to_point_trajectories_v2`；AIoU/DTW/LDTW 使用 `pen_net_equations_local256_v1`，按整字算术平均，`official_implementation_parity=false`，尚未核验与官方实现完全一致。
- [原始 summary](../runs/casia_olhwdb11/ordered_stroke/job_1597/train/eval_epoch_006/summary.json)、[逐字结果](../runs/casia_olhwdb11/ordered_stroke/job_1597/train/eval_epoch_006/characters.jsonl)、[30 个样例画廊](../runs/casia_olhwdb11/ordered_stroke/job_1597/train/eval_epoch_006/index.html)。
- epoch 7 尚无完整 summary，不纳入本次成绩。`strict_sequence_mask50` 不等于上述 Order Acc.。

真实掩膜/真实首点条件下的点层成绩见第四部分诊断附表，不填入完整模型表。

## 三、书法宽度消融

固定同一组预测中心线和笔画边界，比较以下三种宽度策略：

| 宽度策略 | CharIoU ↑ | OrdIoU ↑ | 宽度 MAE（px）↓ |
|---|---|---|---|
| 全局固定宽度 | 待测 | 待测 | 待测 |
| 整笔复制首点宽度 | 待测 | 待测 | 待测 |
| 逐点预测宽度 | 待测 | 待测 | 待测 |

另有已训练的 XY 基线时，补充 XY 与 XYW 的 CharIoU、OrdIoU、XY MAE 和笔画数正确率。

当前未找到符合本节要求的三策略配套结果。第一部分提供逐点预测宽度的端到端基准，但尚未完成固定相同中心线和笔画边界的三策略消融，因此本表仍保留待测。

## 四、掩膜与首点条件消融

固定点层模型及同一配对样本集合，返回四组结果：

| 掩膜 | 首点 XYW | 逐笔轨迹 IoU ↑ | XY MAE ↓ | 宽度 MAE ↓ | 终点 L2 ↓ |
|---|---|---|---|---|---|
| 真实 | 真实 | 待测 | 待测 | 待测 | 待测 |
| 真实 | 预测 | 待测 | 待测 | 待测 | 待测 |
| 预测 | 真实 | 待测 | 待测 | 待测 | 待测 |
| 预测 | 预测 | 待测 | 待测 | 待测 | 待测 |

同时返回配对笔画数和未匹配比例。

当前未找到同一配对样本集合上的完整四组消融。已有以下**真实掩膜＋真实首点 XYW 的独立点层诊断**，仅供单列报告，不拼入上述四组表：

| 数据与划分 | 笔画数 | 逐笔轨迹 IoU | XY MAE（px） | 宽度 MAE（px） | 终点 L2（px） |
|---|---:|---:|---:|---:|---:|
| 书法 test，job605 epoch20 | 68,413 | 0.839153 | 1.7063 | 1.1634 | 4.1519 |
| CASIA val，job1136 epoch4 | 519,116 | 0.905567 | 4.8327 | 0.0433 | 1.8965 |

两组点层诊断均从真实首点之后自由生成；重采样 XY/宽度误差排除给定首点。完整两阶段的对应误差包含预测首点，且只覆盖按序号可配对笔画，不能直接作严格配对消融。

CASIA 点层另有 AIoU 0.867841、LDTW 2.2241px、DTW 55.3046，均为 519,116 笔等权平均，不能与第二部分整字聚合成绩混用。

来源：[书法点层 test summary](../runs/single_stroke_full/job_605/test_job_675/summary.json)、[CASIA 点层 val summary](../runs/casia_olhwdb11/single_stroke/job_1136/train/eval_total_epoch_004/summary.json)。


## 五、汇报图片

- 同一批书法样例：输入图、参考轨迹、本模型预测及重绘图。已有 30 个测试集样例的 HTML 画廊与逐字 `character_trajectory.npz`（见第一部分）；G-HTR 图片对照因代码与数据集不可获得，当前不开展。
- 待从已有样例中选择 6～10 个字，覆盖简单字、复杂字、交叉粘连和明显粗细变化，包含典型成功与失败样例，并导出汇报图片。
- 待选 3～5 条笔画，返回参考与预测宽度曲线，以及三种宽度策略的重绘对照。现有 NPZ 可作后续整理素材，尚未找到已完成的曲线与三策略对照。
- CASIA 已有 30 个验证集两阶段可视化样例（见第二部分），包含输入、参考及预测；待选 3～5 个并导出 PNG/PDF。

## 六、有余力再补

- 单尺度与四尺度解码：CharIoU、OrdIoU、XY MAE、宽度 MAE。
- 未见字符测试：AIoU、LPIPS、LDTW、DTW、Order Acc.。
- 效率：笔画层耗时、点层耗时、总耗时、参数量、峰值显存。已有[旧版 CPU 探测记录](../runs/edge_cpu_probe_20261029/job_791/lean_t4_b1.json)，笔画层 39,854,658 参数、点层 11,978,500 参数，共 51,833,158；其笔画层为 epoch 10，仅 3 个样例，不能替代第一部分 epoch 20 测试模型的 GPU 计时与峰值显存。

## 回传内容

1. 上述结果表，CSV 或 Excel 均可。
2. 原始评估 summary，以及逐字结果文件（如有）。
3. 图片和宽度曲线，优先 PNG/PDF，保留样本 ID。
4. 每组结果附：数据集与划分、实际测试样本数、权重名称、画布分辨率、指标版本；计时附 GPU 型号和 batch。

**时间有限时，先交第一、二、五部分，再补消融。**
