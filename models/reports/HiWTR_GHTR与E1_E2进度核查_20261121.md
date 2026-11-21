# G-HTR 训练与 HiWTR E1/E2 消融进度核查

更新时间：2026-11-21T14:37:49.326050+08:00

## G-HTR（作业 3802）

- 当前已退出，不在运行或排队。训练在第 10,000 步后的首次验证发生 CUDA OutOfMemoryError。
- 训练目标 300,000 步，完成 10,000 步（3.33%）；epoch 8，累计字符曝光 479,856，训练计时约 26,547 秒（7小时22分27秒）。
- 最近100步训练 loss 平均 0.038397；最近1000步平均 0.038766。该数值是训练损失，不能替代自由生成验证结果。
- 已保存 `G-HTR/runs/job_3802/last.pt`（732,179,873 字节）；检查点元数据 best_score/best_step 均为 null，last_evaluated_step=0。没有 best.pt。
- `val_step_00010000/` 只有4个字的 NPZ 逐字预测及4行 characters.jsonl，计划验证8,121字（完成约0.049%），没有 summary.json，不能作为完整验证成绩。
- 局部产物中，GT笔画数 [1,1,1,2]，预测笔画数 [1,129,126,52]；部分宽度误差约千像素，说明这些样本的自由生成异常。样本量太少，不能推广为全验证集结论。
- 报错发生在 model.generate→decode→Transformer 自注意力；申请6.52GiB时OOM。配置 decode_max_steps=16384，当前每步重算完整历史的自注意力；长序列解码是优先排查因素，未记录失败样本的实际序列长度，不能给出精确触发长度。
- 本次未修改训练代码/配置，未重启训练。恢复前应排查自由生成的停止、宽度和长序列显存问题；直接原配置续训会先重新验证，可能再次OOM。

证据：`G-HTR/logs/ghtr-ai-3802.err`、`.out`，`G-HTR/runs/job_3802/train.jsonl`、`val_step_00010000/characters.jsonl`。

## HiWTR E1/E2 消融

消融已经启动过，并非一直没有执行：

1. 3803 smoke 已通过，54个条件，BOOK/CASIA各8字，噪声先1draw；耗时406.7秒。
2. 3804 已完成18个基础条件的val/test预测和验收：val 1,800条，test 8,762条。首点保留、固定匹配、E2同行共享上游、O11等于E2均通过。
3. 3804 在基础汇总时失败：GPU主进程已持有 BOOK LMDB，aggregate 又创建新的 Experiment 并尝试打开相同LMDB，触发 `lmdb.Error: ... is already open in this process`。
4. 3805 噪声val因此进入 DependencyNeverSatisfied，3806–3808也继续等待。这是流程错误，不是权重错误或GPU结果缺失。

已恢复：提交独立CPU基础汇总作业 **3828**，复用已验收预测，3805依赖改为 afterok:3828。后续仍为 3805 val扰动 → 3806 val验收 → 3807 test扰动 → 3808最终报告，GPU数组最大并发1。没有改动冻结权重、协议、源码或抽样清单，不重新推理基础条件，也没有对test选权重。

核查时3828运行中；扰动正式任务尚未开始，完整最终报告尚未生成。基础汇总成功后会生成 `summaries/base/REPORT.txt` 与 `models/reports/HiWTR_mask_start_stage_run_20261120_v1_base.txt`。完整报告预期为同名 `_final.txt`，不可把中间文件当最终结论。

注意：原status.json残留“3804 running”已纠正为恢复汇总状态；旧失败日志保留，恢复过程记录于 recovery_20261121.json。该恢复使用独立进程规避双重打开；原GPU base入口仍有嵌套汇总缺陷，后续不要直接重跑该入口。

实验目录：`CHTR/models/runs/start_noise_stage_cross/run_20261120_v1/`。
