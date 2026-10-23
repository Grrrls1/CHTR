# CHTR 工作交接

## 当前唯一决策

参考 [Handwriting Trajectory Recovery via Autoregressive Ordered Stroke Instance Prediction](https://arxiv.org/html/2609.02251v1)，
将“有序笔画实例预测 → 单笔轨迹生成”两阶段方法迁移到书法任务。
唯一有效待办是 [models/TODO.md](models/TODO.md)。
旧 free rollout、三任务分层 AR、扩散 / Flow Matching 的后续计划全部放弃，
包括它们的 GPU 调参、正式训练和消融。代码存在或旧文档中有复选框，不代表应继续执行。

已实现独立单笔轨迹阶段和固定划分接入，完成本机CPU smoke、小集训练与全量数据接口审核。
有序笔画阶段和两阶段串联尚未实现；未启动正式全库训练。
当前实现、命令和差异见 [单笔阶段说明](models/SINGLE_STROKE.md)，验证摘要见
[本机smoke记录](models/reports/single_stroke_smoke_20261023.json)。
不生成或提交 Slurm 文件，除非用户后续明确要求。

## 必须保留的 job237 基线

目录：`models/runs/full_separated_10gb/job_237/`。
保留 best.pt、last.pt、config.json、类别表、dataset_selection.json、训练日志、训练状态和全部现有评估。
不删除、覆盖、重新训练写入或将新实验作为该 run 的 resume。
新模型使用独立文件、模式、checkpoint 和输出目录。

job237：CNN/FPN、dim64、4头、笔画 / 点解码器各2层、最大64笔/1024点、memory80 tokens。
condition_mode=predicted，separate_branches=true；点历史训练为 teacher forcing。
整字与笔画条件来自视觉分支，首点用 point BOS 预测，不是直接给真实首点。
点层三分类 state 控制实际结束；辅助 last_head 不参与旧解码停止。
使用完整库71,457条ready，batch48，AdamW lr3e-4，完成100轮、148900步。
训练最后20轮 loss 降低约4.54%，第100轮最佳轮均值2.43398，因达到轮数上限结束。

重要历史区别：job237 宽度损失是归一化 Smooth L1；当前源码后来已改为像素 Charbonnier。
不可把当前 loss 代码反推为旧权重的实际训练历史；精确续训应先恢复历史实现 / 版本。
当前源码中新模型及测试通过，不等于已验证优于 job237。

## 已验证的 job237 评估结论

位置：`models/runs/full_separated_10gb/job_237/eval_epoch_100_20261020_144951/`。
summary.json、state_analysis.json、逐图NPZ / 报告与 index.html 是复查入口。
32张来自训练集，不是泛化评估：teacher-forced IoU 0.6793，free IoU 0.3097。
31/32自主输出整字结束，29/32笔数正确；自主结束不等于结束正确。
331笔按序配对共同笔画，点数MAE26.94点，中位数18点；未计缺失或额外笔画。
每图笔数MAE1.16笔：酹少1、翮少2、爨多34且达到64笔保护上限。
逐笔首点误差按图平均约10.29px；末点空间偏差按图平均约69.65px，勿与点数偏差混用。
按序配对在漏笔后可能错配，需说明指标局限。
teacher-forced 普通收笔精确率20.0%、召回率19.9%；drawing_end召回率46.9%。
last_head在真实笔画范围内32图均识别正确最后一笔，但未用于自由解码。
宽度拟合不足；mask/type较准确不保证轨迹正确。

## 新方案的明确接口

第一阶段：整字视觉特征 + 历史笔画mask编码，预测下一笔mask、框、有效性和起终点。
起点u_k扩展为(x_k,y_k,w_k)，增加首点宽度头，以真实轨迹首点宽度监督；终点首版仍为XY。
首点宽度使用独立损失项，不隐式混入论文的起点XY L1；第一阶段尚未实现，权重等细节待冻结。
第二阶段：单笔图像独立共享编码、P5→P4→P3→P2四尺度串行注意力、4层解码块，
teacher-forced点历史，首版必须同时预测XY、宽度和点有效性。
单笔训练预测第l个点时，输入此前全部真实历史点的XY和宽度（teacher forcing），
不是只给真实首点后反馈预测点；历史点embedding包含宽度，因果mask及移位防止目标泄漏。
首点XY和宽度均直接给真实值，从第二点开始预测XY、宽度和有效性，首点不计入重新预测损失。
自由解码使用预测历史；串联推理首点XY和宽度直接取笔画层预测的u_k三元组，第二阶段不重新预测首点。
给真实首点XY和宽度的自由解码属于条件诊断；完整image-only推理使用预测首点三元组，不能读取真实宽度。
第二阶段XY除以255、宽度直径除以256；XY sigmoid，宽度softplus+1e-6。
第二阶段损失为后续有效点XY L1 + 0.2×有效性BCE + 1×宽度Charbonnier，
宽度epsilon=0.01/256，减去epsilon偏置，各项先每笔平均再对笔平均。
无效结束位置只算BCE，首点和padding不算几何损失。第一阶段后续共用单位与导出转换。
首点与后续点宽度误差应分别报告。
单笔视觉输入为mask区域内的原图内容，区域外填白，保留位置并形成256×256画布；
训练使用真实mask，推理使用预测mask提取同一原图内容，不直接把二值mask当第二阶段图像。
原图必须映射到轨迹对应的规范化画布，不能直接混用original_image_key与image_key坐标。
跨阶段传单笔图像和预测起点，不是将第一阶段隐藏z直接作为第二阶段条件。
预测起点实际作为轨迹第一点；预测终点首版只做第一阶段辅助监督。
两阶段分别训练。论文并非完全自喂历史训练，不能把“AR推理”写成“free rollout训练”。
论文没有宽度输出；宽度是用户要求的首版书法扩展，单独记录损失、权重及粗细评估。
未明确的基础loss形式、网络超参数、图像构造细节与结束占位约定必须核验，不能猜测。

## 数据现状与限制

完整库 `data/chtr_mmah_lmdb`：73,606记录，71,457 ready / 2,149 pending；ready按元数据共727,997笔。
冻结库 `data/chtr_mmah_lmdb_1915`：1915 ready、8990笔、563字符类别；作者全为空，未人工审核。
单笔PNG在 masks_key 的NPZ内，如stroke000.png；轨迹NPZ有points、stroke_offsets等。
PNG编号与offsets按笔画顺序对应；抽查PNG为256×256白底黑笔画。
框可由mask推导，起终点可由轨迹首末点提取；计数不代表全部标签通过质量验收。
这些是自动管线监督，不能宣称为人工真实笔顺 / 采集轨迹。
用户确认方向 / 首点监督可信：pipeline中VS使用MMAH单笔画数据提供首点约束，保证起点正确。
本方案以此为前提；后续抽查用于核对资产匹配、坐标变换与数据接入，不再将方向可信性作为待定前提。
image_key是规范化图片，original_image_key是另一份原始书法图；不要直接混用坐标。
先按整字划分，再展开笔画；无作者信息不能宣称书家独立验证，检查近重复来源。

## 下一次接手的第一步

1. 读主TODO和单笔阶段说明，不继续旧计划；未明确的论文实现细节仍需核验。
2. 单笔阶段代码、训练/条件评估入口已实现，不重复新建；小集产物在
   `models/runs/single_stroke_smoke_v1/`，其中`fit_single_representative`为48点单笔128步拟合，
   `fit_multi_coverage`为8笔32步，`fixed32_epoch1`为32字334笔一轮训练。
3. 10项测试通过，包含增强与dropout下中途恢复逐参数一致、因果性、padding、停止与导出。
   48点单笔自由解码输出48点并正常停止；这是真实mask/首点条件下的训练样本拟合。
4. 全量实际读取71,457张整字、727,997笔，最长393点；取整中心点均在对应mask内。
   train/val/test展开笔数与清单一致；test只检查资产，不用于模型选择。
5. 本机CUDA不可用。固定32字充分拟合、留出自由解码质量和GPU容量尚待验收，
   之后实施有序笔画阶段（含u_k首点宽度头）和串联评估。

数据和训练资产被Git忽略；服务器仅git pull不会自动获得LMDB、权重或本地评估。
仓库当前路径是`/home/lkh/vscode/CHTR`，不要使用已迁移前的epoch100_bundle路径执行命令。
