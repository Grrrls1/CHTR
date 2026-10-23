# CHTR 工作交接

## 当前唯一决策

参考 [Handwriting Trajectory Recovery via Autoregressive Ordered Stroke Instance Prediction](https://arxiv.org/html/2609.02251v1)，
将“有序笔画实例预测 → 单笔轨迹生成”两阶段方法迁移到书法任务。
唯一有效待办是 [models/TODO.md](models/TODO.md)。
旧 free rollout、三任务分层 AR、扩散 / Flow Matching 的后续计划全部放弃，
包括它们的 GPU 调参、正式训练和消融。代码存在或旧文档中有复选框，不代表应继续执行。

本次只更新计划与说明，尚未实现论文两阶段网络，也未启动新训练。
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
第二阶段：单笔图像独立共享编码、多尺度视觉注意力、teacher-forced点历史，预测XY和点有效性。
跨阶段传单笔图像和预测起点，不是将第一阶段隐藏z直接作为第二阶段条件。
预测起点实际作为轨迹第一点；预测终点首版只做第一阶段辅助监督。
两阶段分别训练。论文并非完全自喂历史训练，不能把“AR推理”写成“free rollout训练”。
论文没有宽度输出；保留现有宽度标签，扩展宽度时单独记录偏离论文之处。
未明确的基础loss形式、网络超参数、图像构造与结束占位约定必须核验，不能猜测。

## 数据现状与限制

完整库 `data/chtr_mmah_lmdb`：73,606记录，71,457 ready / 2,149 pending；ready按元数据共727,997笔。
冻结库 `data/chtr_mmah_lmdb_1915`：1915 ready、8990笔、563字符类别；作者全为空，未人工审核。
单笔PNG在 masks_key 的NPZ内，如stroke000.png；轨迹NPZ有points、stroke_offsets等。
PNG编号与offsets按笔画顺序对应；抽查PNG为256×256白底黑笔画。
框可由mask推导，起终点可由轨迹首末点提取；计数不代表全部标签通过质量验收。
这些是自动管线监督，不能宣称为人工真实笔顺 / 采集轨迹。
image_key是规范化图片，original_image_key是另一份原始书法图；不要直接混用坐标。
先按整字划分，再展开笔画；无作者信息不能宣称书家独立验证，检查近重复来源。

## 下一次接手的第一步

1. 读主TODO，核验论文与数据生成约定，不继续旧计划。
2. 审核少量单笔方向 / 首尾标签，确定可信小集与固定整字划分。
3. 新增逐笔Dataset及独立第二阶段，先验证正确单笔输入和起点下的轨迹拟合。
4. 第二阶段验收后实施有序笔画阶段与串联评估。

数据和训练资产被Git忽略；服务器仅git pull不会自动获得LMDB、权重或本地评估。
仓库当前路径是`/home/lkh/vscode/CHTR`，不要使用已迁移前的epoch100_bundle路径执行命令。
