# AI 模型结构图草稿

使用内置 image_gen 生成的 [model_architecture_draft.png](model_architecture_draft.png)。完整生成与编辑提示词见 [prompt.txt](prompt.txt)。

本图展示有序笔画特征、掩码与类别条件、逐点轨迹及训练监督；自由解码与可微重绘标为待实现。

图像生成经过多次修正仍未完全遵守精确连线，当前只能作为视觉草稿：缺失 `z_k → 24 类分类头` 与 `多尺度视觉特征 → 逐点 Transformer` 的直接连线，顶部像素特征连线的来源也不够清晰。正确关系应为：像素特征和 `z_k` 共同进入掩码头；`z_k` 驱动分类头；`z_k` 和类别软嵌入进入条件融合；视觉 tokens、条件融合及掩码偏置共同输入逐点解码器。图中的类别名称和笔画图片仅为示意，不表示真实样本标签。

实现和精确结构以 [ARCHITECTURE.md](../ARCHITECTURE.md) 的 Mermaid 图与网络代码为准。
