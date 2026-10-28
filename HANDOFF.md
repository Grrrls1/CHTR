# CHTR 工作交接

更新：2026-10-28。唯一有效计划见 [models/TODO](models/TODO.md)，当前操作见 [models/README](models/README.md)。
环境统一记录在 [ENVIRONMENT](models/ENVIRONMENT.md)，区分个人电脑与服务器，不重复询问已记录硬件。

## 当前状态

- 已实现“有序笔画实例预测 → 单笔轨迹生成”方案中的单笔阶段；第一阶段和串联尚未实现。
- 服务器训练job605已由用户要求彻底取消，进程已退出；保留全部checkpoint与现有产物。
- 选用 `models/runs/single_stroke_full/job_605/train/best.pt`，epoch20、step120100。
- 完整val 83,119笔：free轨迹IoU 0.8399036936，XY MAE 1.702859px，宽度MAE 1.160251px，端点L2 4.136157px。
- 独立test675已启动并进入逐批推理：全部68,413笔、batch32、30个字符可视化。输出 `models/runs/single_stroke_full/job_605/test_job_675/`；结果以summary.json为准。
- test672因I/O等待已取消。当前独立入口输出初始化阶段、15秒心跳、逐批进度/速度/ETA，心跳不代表数据读取一定有进展。
- 后续优先确认test完成及查看失败样本，不自行重新启动训练。新计算按用户指令通过Slurm执行。

## 当前单笔协议

代码见 [SINGLE_STROKE](models/SINGLE_STROKE.md)。实际读取器用LMDB单笔PNG的 `<128` 区域作为mask，
提取 `image_key` 规范化整字原图内容，区域外填255；保留256×256整字坐标。
当前不是直接将存储的单笔PNG灰度送入模型。源码曾有直接PNG版本，现已恢复为与job605匹配的输入逻辑。

共享ResNet18 → C5 Transformer → FPN，P5→P4→P3→P2串行跨注意力、4层点解码。
teacher forcing输入全部真实历史XY和宽度；给定真实首点三元组，从第二点开始监督。
自由生成反馈预测历史，阈值0.5、上限400，invalid候选不保存。
XY除以255、宽度直径除以256，XY sigmoid、宽度softplus+1e-6。
损失：后续点XY L1 + 0.2×validity BCE + 1×width Charbonnier；epsilon=0.01/256且减去偏置。
首点及padding不计几何损失，终止位置仅监督有效性。当前评估包含标签条件，不是image-only整字推理。

训练配置batch96、dim64、AdamW恒定lr3e-4、dropout0、200epoch上限；每100步保存checkpoint。
每10epoch完整val，按free轨迹IoU选best；patience10按验证次数计，min_delta=.001、min_epochs20。
独立test入口及个人电脑命令见 [TEST_README](models/TEST_README.md)。

## 数据与限制

完整LMDB：73,606记录，71,457 ready、2,149 pending；ready共727,997笔。
划分文件 `models/data_splits/full_character_v1.json`：train/val/test为56,622/8,121/6,714张、576,465/83,119/68,413笔。
同一整字全部笔画属于一个集合；按字符分层及来源/近重复分组，不是字符类别或书家隔离。
全量接口审核最长393点；自动管线标签不等于人工采集真实笔顺/轨迹。
用户确认方向与首点监督可信，后续抽查资产对应与坐标变换。
image_key与original_image_key坐标不可直接混用。数据、权重和运行输出被Git忽略，需单独迁移。

## 保留的job237历史基线

目录 `models/runs/full_separated_10gb/job_237/`，保留best/last、配置、类别表、日志和全部现有评估。
保留模型与评估代码，结构见 [ARCHITECTURE](models/ARCHITECTURE.md)，本地查看见 [LOCAL_EPOCH100](reproduction/LOCAL_EPOCH100.md)。
CNN/FPN、dim64、4头、两级各2层、最大64笔/1024点、memory80，condition_mode=predicted、separate_branches=true。
全库训练100轮、148900步；历史宽度loss为归一化Smooth L1，不能用后改的像素Charbonnier反推训练历史。
32张评估来自训练集：TF IoU 0.6793、free IoU 0.3097；31/32自主结束，29/32笔数正确。
按序匹配的331笔点数MAE26.94、首点误差约10.29px、末点误差约69.65px；漏笔后可能错配，未计缺失/额外笔画。
这些条件与job605不同，不能直接将分数视为泛化优劣。

## 下一阶段与清理记录

第一阶段预测mask、框、有效性、起点 `(x,y,width)` 和终点XY；先冻结基础loss与独立首点宽度监督。
串联传预测mask提取的单笔图和预测首点，第二阶段不重新预测第一点，不直接传隐藏z。
方法来源：[Handwriting Trajectory Recovery via Autoregressive Ordered Stroke Instance Prediction](https://arxiv.org/html/2609.02251v1)。
网络型号、loss细节等未核实项不能当成论文官方实现。

已按用户要求删除放弃的free rollout、旧三任务分层AR、扩散/Flow Matching专用文档、实现、配置与测试。
保留job237基线、当前单笔阶段、共享模块及数据生成管线；本次不删除任何训练权重、数据或已有实验产物。
