# CHTR 本地复现与扩展

从书法单字图像构建有序笔画及六维轨迹数据，训练图片到笔画轨迹的分层模型。
本仓库基于 CHTR 论文和相关开源项目开展本地复现与扩展，不代表原作者官方实现或官方 CHTR-110K 数据。

```text
data/                           全量数据构建、续跑与 LMDB 封装代码
models/                         PyTorch 模型、训练、评估与测试
reproduction/                   统一全量构建入口与流水线
third_party/StrokeExtraction/   带本地修改的三阶段笔画提取项目
third_party/virtual_sketching/   带本地修改的 TF1 矢量化项目
```

两个第三方项目以普通源码目录纳入，保留当前修改。其数据、环境和权重在本机目录中保留，
由 `.gitignore` 排除；上游版本、Git 历史位置及路径兼容说明见
[third_party/README.md](third_party/README.md)。

全量数据构建使用 [reproduction/build_full_dataset.slurm](reproduction/build_full_dataset.slurm)，
包含提取清理、分批骨架生成及全量 LMDB 重封装，支持只读计划和分阶段继续。
具体命令见 [reproduction/README_zh.md](reproduction/README_zh.md)。
当前训练库含 73,606 条记录，其中 71,457 条 ready、2,149 条 pending；模型说明见
[models/README.md](models/README.md)。外部原图和参考资源需要单独准备。

Virtual Sketching 使用独立旧 TensorFlow 1 环境，当前统一全量入口采用骨架路线。
原项目的 VS 方法仍可通过流水线单独运行。

当前自由训练使用全部 1915 样本、随机初始化、dim=128 和最多 500 epoch；早停沿用
job237 的训练损失规则，默认不加载旧权重。配置、Slurm 入口和恢复方法见
[FREE_ROLLOUT.md](models/FREE_ROLLOUT.md)。

保留原目录结构、携带源码与必要权重/数据的选择性打包，以及 Linux/NVIDIA 本地查看
job237 第 100 轮效果的方法，见 [本地评估与迁移](reproduction/LOCAL_EPOCH100.md)。
