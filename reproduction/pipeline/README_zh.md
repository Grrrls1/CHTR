# CHTR 本地数据流水线脚本

实现日期：2026-09-15。这里实现根目录清单中的 12 个主流水线入口，并加入新数据导入、核验和候选联结入口，使用本地 `chtr-local-v1` 格式；不是 CHTR 作者公开格式。所有命令支持 `--help`。

## 环境与路径

除提取阶段外，可使用现有 TF1 CPU 环境运行所有脚本：

```bash
cd /home/lkh/workspace
CHTR_PYTHON=/home/lkh/workspace/.venvs/virtual-sketching/bin/python
CHTR_SCRIPTS=/home/lkh/workspace/CHTR/reproduction/pipeline
export PYTHONPATH=
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
```

`extract_strokes_batch.py` 必须使用安装了 StrokeExtraction 依赖的 Python，并且需要可用的 CUDA。2026-09-15 本机驱动不可用，因此本次没有重跑该阶段的模型前向；该封装复用现有 `ExtractStroke` 类，三模型每批只加载一次。VS 阶段必须使用 TF1 环境，模型也每批只加载一次。非模型阶段依赖 NumPy、OpenCV、Pillow，不导入 PyTorch 或 TensorFlow。

JSONL 中的 `*_path`、`input_png`、`npz`、`model_config` 相对路径以 manifest 所在目录为基准；输出 manifest 使用绝对路径。搬迁整个数据集时需重定位 manifest 路径。输出目录必须是新的或空的，审核修改写到新目录保留历史；只有 VS 批处理支持 `--resume`，会核对输入、模型、种子和缓存文件哈希。不要把新输出目录设为已有原始数据目录。

## 入口与数据约定

| 脚本 | 输入 | 主要输出 |
| --- | --- | --- |
| `prepare_sources.py` | 图像/font/GNT 来源 JSONL | 256 PNG、坐标变换、manifest |
| `build_reference_bank.py` | 带显式 24 类标签的参考 NPZ 清单 | 参考 mask、质心、来源哈希 |
| `extract_strokes_batch.py` | 规范化 manifest + 参考库 | 三阶段 NPZ、pending manifest、错误清单 |
| `clean_stroke_masks.py` | 提取 manifest + target + SDNet笔画先验 | 连通域清理、before/after/删除区可视化、待审核 manifest |
| `review_strokes.py` | 提取 manifest + 人工 decisions | 预览 HTML、编辑后的 mask、审批记录 |
| `export_vs_inputs.py` | 已批准 mask manifest | 白底黑笔画 PNG、stroke manifest |
| `export_clean_masks_for_vs.py` | 自动清理报告 + mask manifest | 按整字 IoU 自动筛选的探索性 VS 输入，明确不声明人工批准 |
| `vectorize_strokes_batch.py` | stroke manifest | 多候选 NPZ、上游工具可视化、配置/种子/哈希 |
| `decode_vs_npz.py` | 单份原生 NPZ + 模型配置 | 六维 points、subpath_offsets、曲线参数 |
| `render_and_filter.py` | 候选 manifest | 解码 NPZ、重绘/叠图、双重 IoU、最佳候选 |
| `select_single_path_eval.py` | `vs_qc_v3/best.jsonl` | 可重复 12 笔诊断集、静态图、GIF、SVG |
| `skeleton_path_trajectory.py` | 单笔 mask manifest | 骨架 + 距离变换宽度的严格单 subpath 六维基线 |
| `assemble_single_path.py` | 单路径逐笔 manifest | 探索性整字 NPZ、重绘、整字 IoU |
| `compare_single_path_routes.py` | VS/P5 两份逐笔 manifest | subpath、IoU、长度、过绘对比 CSV/JSON |
| `render_character_centerlines.py` | 单路径逐笔 manifest | 每个整字所有笔画合成的中心线图、笔序图和目标叠加图 |
| `pack_exploratory_example.py` | 单路径逐笔/整字 manifest | 跳过人工审核的低门槛示例数据集、原图、六维轨迹和对应说明 |
| `simplify_dataset_manifest.py` | 阶段 manifests + cleanup report | 简化 manifest、统一 trajectory_style、保留阶段 IoU 和 dataloader 质量字段 |
| `assemble_vs_exploratory.py` | 每笔最佳 VS 候选 | 自动门槛下的整字六维 NPZ、重绘图、拒绝清单 |
| `review_trajectories.py` | 每笔一个候选 + decisions | 子路径方向预览、编辑后轨迹、重新质检 |
| `pack_dataset.py` | 所有笔画均批准的 manifest | 整字 NPZ、图像、逐笔画 mask、metadata、split |
| `validate_dataset.py` | 最终 manifest | 结构、状态、逐笔画/整字 IoU、split 泄漏报告 |
| `run_pipeline.py` | 明确列出命令与输入输出文件的 JSON | 分阶段日志、内容哈希、恢复运行记录 |
| `import_calligraphy_dataset.py` | 单字目录与 `label_character.csv` | 256 PNG、未知书家来源 manifest |
| `import_ccse_kaiti_sequence.py` | Lee-xeo `fort_graphics.json` 与 CCSE-Kaiti COCO | 有序 NPZ、预览、CCSE-25 类别表 |
| `verify_sequence_import.py` | 有序 manifest、原始笔顺 JSON、CCSE | 全量文件哈希、容器字段及类别表报告 |
| `link_calligraphy_sequence.py` | 两个导入 manifest | 同字候选清单、覆盖统计、可选直接重合诊断 |

### 1. 来源与规范化

`sources.jsonl` 每行选择一种来源，例如：

```json
{"sample_id":"writer01_char0001","image_path":"images/0001.png","character":"一","writer_id":"writer01","source":"licensed_raster","polarity":"black_on_white"}
{"font_path":"fonts/example.ttf","character":"一","style_id":"font01","source":"licensed_font","font_size":192}
{"gnt_path":"casia/writer01.gnt","writer_id":"writer01","source":"CASIA-HWDB","encoding":"gb18030"}
```

字库文件和 CASIA 原始数据由用户提供。GNT 解析检查样本长度、宽高和截断；writer ID 必须由来源明确提供。字体渲染需人工确认字体覆盖所需字符，避免将缺字方框作为汉字。PNG 可指定 `white_on_black`，脚本会反色；不自动猜测笔画极性。保持宽高比缩放并居中留白，记录实际整数缩放尺寸、各轴比例和 padding。

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/prepare_sources.py" --manifest sources.jsonl --output chtr_dataset/normalized
```

### 2. 参考库与提取

参考清单示例：

```json
{"character":"迟","npz_path":"reference/chi.npz","source":"RHSEDB reference","category_system":"StrokeExtraction-24"}
```

NPZ 必须包含 `reference_single_image[N,256,256]`（0/1）和整数 `stroke_label[N]`（0..23）。1..30 笔，所有 mask 非空。同字只保留一个参考。Make Me a Hanzi SVG 和论文 25 类标签不能直接导入，需先提供已验证的栅格化和类别映射。

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/build_reference_bank.py" --manifest references.jsonl --output chtr_dataset/references
# 在具备 StrokeExtraction 依赖和 CUDA 的环境中执行：
python "$CHTR_SCRIPTS/extract_strokes_batch.py" --manifest chtr_dataset/normalized/manifest.jsonl --references chtr_dataset/references/manifest.jsonl --output chtr_dataset/extracted --gpu 0

python "$CHTR_SCRIPTS/clean_stroke_masks.py" --manifest chtr_dataset/extracted/manifest.jsonl --output chtr_dataset/mask_cleanup --whole-iou 0.87
```

清理阶段默认将 mask 限制在 target 邻域，删除极小毛刺和被其他笔画高度覆盖的离散块；短间隙只有在 target 与 SDNet 对应参考笔画共同支持的通道内才连接。随后执行 5×5 闭运算、3×3 开运算的受约束边缘平滑，填充不超过 256 像素的内部孔洞，并强制每笔只保留最大连通块。使用 `--allow-multiple-components` 可关闭最后一步。`comparison.png` 中红色为删除像素、绿色为新增像素，所有动作写入逐样本 `cleanup.json`。

提取失败样本写入 `errors.jsonl`，只要有失败则命令退出码为 1，避免调度器把不完整批次视作完成。

[examples/1613_extracted.jsonl](examples/1613_extracted.jsonl) 是 1613 结果的历史清单模板，路径已指向
`CHTR/third_party/StrokeExtraction/results/1613/`。当前本机没有该样例目录，使用前需先生成对应结果。
该示例不猜测作者 ID；入库之前必须补充真实 writer/style 来源信息。

### 3. 笔画区域审核

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/review_strokes.py" --manifest chtr_dataset/mask_cleanup/manifest.jsonl --output chtr_dataset/mask_preview
```

打开 `mask_preview/index.html` 查看逐笔画图片，并编辑 `decisions.template.json` 的副本。模板默认 `pending`，不会自动批准。每条批准/拒绝需填写 `reviewer`；`source_sha256` 防止审核应用到旧输入。

`strokes` 数组定义编辑后的顺序：

- 调整顺序：重排数组项；每项 `label` 是最终类别 ID。
- 合并：`{"indices":[0,1],"label":0}` 对指定 mask 取并集。
- 删除：从数组移除不需要的项。
- 区域修正/拆分：使用外部图像编辑器保存 256×256 白底黑笔画 PNG，再用 `{"mask_path":"edited/part1.png","label":0}` 替换一个项或增加多个项。路径相对于 decisions 文件。

这是 HTML 预览与文件/终端交互审核，不提供浏览器内画笔编辑器。也可使用 `--interactive mask_preview/decisions.template.json`，在终端逐条输入 JSON 编辑和批准。

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/review_strokes.py" --manifest chtr_dataset/extracted/manifest.jsonl --decisions mask_decisions.json --output chtr_dataset/reviewed_strokes
"$CHTR_PYTHON" "$CHTR_SCRIPTS/export_vs_inputs.py" --manifest chtr_dataset/reviewed_strokes/manifest.jsonl --output chtr_dataset/vs_inputs
```

导出拒绝 pending/rejected 或审核后被改动的 mask；输出保留原图、作者、笔顺、类别和审核记录。

### 4. 矢量化与质检

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/vectorize_strokes_batch.py" --manifest chtr_dataset/vs_inputs/manifest.jsonl --output chtr_dataset/vs_raw --samples 3 --seed 1613
# 中断后继续同一批次：加 --resume，参数须保持一致。
"$CHTR_PYTHON" "$CHTR_SCRIPTS/render_and_filter.py" --manifest chtr_dataset/vs_raw/manifest.jsonl --output chtr_dataset/qc
```

VS 批处理每个候选使用由 stroke ID 和基础 seed 导出的随机种子，保存模型配置、checkpoint 内容哈希和输入哈希。逐笔画错误被隔离、记录，命令最终返回非零。模型常驻；每个候选调用上游 `tools/visualize_drawing.py` 的 `display_strokes_final`，输出 4 张可视化。

质检会生成：

- `manifest.jsonl`：所有候选；`best.jsonl`：每笔优先选通过者，否则保留最高 IoU 的拒绝者供审核。
- `render.png`：从六维点轨迹用连续圆刷重绘。
- `native_replay.png`：独立 NumPy 重放上游量化、窗口裁剪和双线性粘贴。
- `overlay.png`：红色是目标 mask，蓝色是点轨迹，黑色是交集。
- `index.html`：输入/结果对照及带箭头的子路径。

IoU 统一按黑底强度 `>=128`（白底图 `<128`）计算；全空返回 0。通过需要源 mask IoU ≥0.95、点轨迹与上游渲染 IoU ≥0.95、独立原生重放 IoU ≥0.999，且轨迹满足边界等条件。降低诊断阈值不能绕过最终入库的 0.95 门槛。

单独解码：

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/decode_vs_npz.py" --file candidate.npz --model-config CHTR/third_party/virtual_sketching/outputs/snapshot/pretrain_clean_line_drawings/model_config.json --output decoded.npz
```

当前支持 clean 模型的 256 画布、128 raster、`cursor_type=next`、每轮宽度重置。保存未裁剪曲线端点；下一段光标按上游规则裁剪，若出现位置跳变则新建子路径。越界轨迹拒绝入库，不能靠静默裁剪掩盖问题。

### 5. 轨迹审核、整字封装和验证

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/review_trajectories.py" --manifest chtr_dataset/qc/best.jsonl --output chtr_dataset/trajectory_preview
```

打开 HTML，按起点编号及箭头确认方向。复制 decisions 模板，`subpaths` 支持重排及 `reverse:true`，每条子路径必须恰好出现一次。反向同步点宽并重算笔状态。复杂修正可提供 `replacement_path` NPZ，字段为 `points[T,6]` 和 `subpath_offsets`；所有修改都会重新渲染并检查源 mask 和上游一致性，不允许批准失败候选。

```bash
"$CHTR_PYTHON" "$CHTR_SCRIPTS/review_trajectories.py" --manifest chtr_dataset/qc/best.jsonl --decisions trajectory_decisions.json --output chtr_dataset/reviewed_trajectories
"$CHTR_PYTHON" "$CHTR_SCRIPTS/pack_dataset.py" --manifest chtr_dataset/reviewed_trajectories/manifest.jsonl --output chtr_dataset/final --seed 1613
"$CHTR_PYTHON" "$CHTR_SCRIPTS/validate_dataset.py" --manifest chtr_dataset/final/manifest.jsonl --report chtr_dataset/final/validation.json
```

任意笔画缺失、重复、未审批、文件哈希变化、质检失败或作者来源缺失都会阻止整字入库；额外检查整字重绘 IoU。最终 NPZ 含 float32 `points`、整数 `stroke_offsets`、`subpath_offsets`、`stroke_types`。状态是到达该点之后的动作；所有子路径末点为 `[0,1,0]`，整字最后一点为 `[0,0,1]`，其余 `[1,0,0]`。第三列是像素直径。

split 根据 seed 和 writer/style ID 的哈希确定。同一作者/字体不会跨 split；validator 还检查重复图像跨 split 泄漏。小数据集可能只有一个 split，不能把单字测试当成训练/评测数据集。

### 6. 调度与恢复

`run_pipeline.py --config config.json --dry-run` 只显示计划。实际运行默认遇到失败就停止，日志写到 report 同级目录；`--resume` 仅跳过代码、输入和输出哈希一致的完成阶段。所有 input/output 都应列出具体文件（模型文件、decisions、manifest 等），不能用目录代替；依赖目录内资产应逐个列为 input。

配置示例：

```json
{
  "report": "run/report.json",
  "stages": [{
    "id": "sources",
    "script": "prepare_sources",
    "python": "/home/lkh/workspace/.venvs/virtual-sketching/bin/python",
    "args": ["--manifest", "sources.jsonl", "--output", "normalized"],
    "inputs": ["sources.jsonl", "images/0001.png"],
    "outputs": ["normalized/manifest.jsonl", "normalized/writer01_char0001.png"]
  }]
}
```

路径相对于配置文件目录。审核应用阶段可声明 `requires_decisions`；缺文件时停止，不自动填写审批人或通过状态。不同阶段通过 `python` 选择各自解释器。修改代码/输入后应使用新版本输出目录，已有结果不会被调度器强行覆盖。

## 实测与限制

2026-09-15，新批处理入口对 1613 的 7 笔各生成 1 个候选（seed=1613），随后运行解码、独立重放、质检和轨迹预览。结果位于 [pipeline_runs/1613_20260915](../pipeline_runs/1613_20260915)。

独立原生重放与上游工具的二值 IoU 全部为 1.0；连续六维点重绘与上游 IoU 约 0.81–0.89，源 mask IoU 约 0.76–0.88，7 个候选全部未通过 0.95 质检，保持 pending/rejected，不生成伪造的通过数据。上游在局部 2 倍画布取整画圆、缩小并再裁剪插值，连续直径表示仍存在栅格化近似。需要继续改进点轨迹渲染/宽度拟合与候选质量，才能得到合格真实样本。

合成数据测试覆盖轴交换、宽度变化、移动/换轮、子路径断开、空数据、GNT 截断、审核哈希、方向反转、质检拒绝、完整封装与 split 泄漏。测试命令：

```bash
"$CHTR_PYTHON" -m unittest discover -s "$CHTR_SCRIPTS/tests" -v
```

通用参考库、类别映射、CASIA/字库授权资源与人工审核仍需实际提供。脚本可执行不代表已复现 CHTR-110K，也不代表已验证 GPU 批量提取或真实 GNT 全集。

## 7. 书法单字与有序楷体模板候选（2026-09-17）

已有权威产物：

- `datasets/processed/chinese-calligraphy-221030/`：138,499 张 256 PNG，7,328 字；逐图书家和书体未知，全部 `pack_eligible=false`。
- `datasets/processed/ccse-kaiti-sequence-v2/`：9,523 字的按笔顺 NPZ、预览图、`category_map.json`。总路径数 112,099；9,510 字有完整 CCSE-25 类别，13 字因上游标签缺失而将全部笔画类别置为 `-1`，原残缺标签保存在 `raw_partial_stroke_label`。
- `datasets/processed/calligraphy-kaiti-candidates-v1/`：138,499 行同字候选及统计。6,360 个书法字有模板，968 字没有；13 个类别缺失字中有 5 字出现在书法集，对应 108 张书法图。无模板共 6,535 张。其余 131,856 张为待审核候选。

小批量使用已有产物的示例（不会重新生成全量图像）：

```bash
cd /home/lkh/workspace
PY=.venvs/virtual-sketching/bin/python
PIPE=CHTR/reproduction/pipeline
$PY $PIPE/link_calligraphy_sequence.py \
  --calligraphy datasets/processed/chinese-calligraphy-221030/manifest.jsonl \
  --sequences datasets/processed/ccse-kaiti-sequence-v2/manifest.jsonl \
  --output datasets/processed/my-candidates-v1 --audit-samples 12
$PY $PIPE/verify_sequence_import.py \
  --manifest datasets/processed/ccse-kaiti-sequence-v2/manifest.jsonl \
  --graphics Chinese-Character-Stroke-Sequence-Dataset/fort_graphics.json \
  --ccse datasets/ccse-kaiti-2021 \
  --report datasets/processed/my-sequence-verification.json
```

导入器各自支持 `--help`；它们要求完整原包布局且写入全新空目录，不能当作小样本 CLI 使用。小样本覆盖在 `tests/test_new_imports.py` 中。`overlap_diagnostics.jsonl` 给出 256 画布上的直接二值 IoU，仅用于发现位置或尺度差异，不执行配准，不是提取或审核通过分数。联结 manifest 中 `match_basis=unicode_character_only`、`registration_status=not_evaluated`、`review_status=pending`、`strokeextraction_24_compatible=false`；它不能直接输入 `build_reference_bank.py`。在人工核对字形、笔画位置和类别体系前，不要将 CCSE-25 的 1..25 ID 转为 StrokeExtraction 的 0..23 ID。

书法原包是互联网收集图片，包内未见逐图许可；Lee-xeo 的笔顺记录结合 CCSE 和 Make Me A Hanzi 来源。商业使用或再分发之前，需要用户提供逐图授权证明、真实书家/书体元数据，以及上游衍生数据的许可核对结果。现有书法清单无法通过 `pack_dataset.py` 的真实作者来源检查。后续若只用于探索，应隔离为未批准候选；在具备 GPU 的服务器上先对少量图像做配准和人工检查，再决定是否扩大提取。

## 8. RHSEDB 完整参考库（2026-09-17）

`select_rhsedb_references.py` 对 RHSEDB 的 train/test 目录按 NPZ 内 `name` 字段去重，优先选择 train 中按数字文件 ID 排序的第一份有效原生 24 类参考；仅缺字时使用 test。它检查单笔画 mask 非空、形状和类别范围，输出每字一行的输入清单。随后使用现有 `build_reference_bank.py` 构建参考库：

```bash
cd /home/lkh/workspace
PY=CHTR/third_party/StrokeExtraction/.venv/bin/python
PIPE=CHTR/reproduction/pipeline
RUN=CHTR/reproduction/pipeline_runs/rhsedb_reference_bank_20260917
$PY $PIPE/select_rhsedb_references.py --rhsedb CHTR/third_party/StrokeExtraction/dataset/RHSEDB --output "$RUN/selection"
$PY $PIPE/build_reference_bank.py --manifest "$RUN/selection/manifest.jsonl" --output "$RUN/bank"
```

已生成的 [bank manifest](../pipeline_runs/rhsedb_reference_bank_20260917/bank/manifest.jsonl) 包含 3,424 个不同汉字，各一份模板；3,408 份来自 train，16 份来自 test。与书法单字集交集 3,394 字、82,273 张图；详细见 [覆盖统计](../pipeline_runs/rhsedb_reference_bank_20260917/coverage.json) 和 [核验报告](../pipeline_runs/rhsedb_reference_bank_20260917/verification.json)。这只是可供提取的同字参考，并不保证书法字形与 RHSEDB 楷体能成功配准，也不代表已有书家来源、人工审核或六维轨迹。若做 RHSEDB 自身的测试集评估，应排除来自 test 的 16 份模板，避免测试来源泄漏。

## 9. 官方 Make Me A Hanzi 候选（2026-09-17）

原始数据来自 [skishore/makemeahanzi](https://github.com/skishore/makemeahanzi) 的 `graphics.txt`，固定提交 `bddc96d41bef78427ed0e034e9f7e31d71fd1b92`；本地在 `datasets/sources/makemeahanzi/repo_filtered/`，含官方 `COPYING` 与 `APL/`。`graphics.txt` 的 Git blob 为 `ef25a9208f7c4daf942d6f12be0a0f85075d140a`。只导入图形数据，不将 `dictionary.txt` 的不同许可混入。

```bash
cd /home/lkh/workspace
PY=.venvs/virtual-sketching/bin/python
PIPE=CHTR/reproduction/pipeline
$PY $PIPE/import_makemeahanzi.py \\
  --graphics datasets/sources/makemeahanzi/repo_filtered/graphics.txt \\
  --source-commit bddc96d41bef78427ed0e034e9f7e31d71fd1b92 \\
  --output datasets/processed/makemeahanzi-ordered-new --workers 4
$PY $PIPE/link_template_sources.py \\
  --calligraphy datasets/processed/chinese-calligraphy-221030/manifest.jsonl \\
  --ccse datasets/processed/ccse-kaiti-sequence-v2/manifest.jsonl \\
  --makemeahanzi datasets/processed/makemeahanzi-ordered-v1/manifest.jsonl \\
  --rhsedb CHTR/reproduction/pipeline_runs/rhsedb_reference_bank_20260917/bank/manifest.jsonl \\
  --source-comparison datasets/processed/makemeahanzi-ordered-v1/source_comparison.json \\
  --output datasets/processed/calligraphy-template-candidates-new
```

权威产物 [makemeahanzi-ordered-v1](../../../datasets/processed/makemeahanzi-ordered-v1/manifest.jsonl) 含 9,574 字、112,617 个有序 mask 与 median，全部 `stroke_label=-1`、`category_system=unlabeled`。与 Lee-xeo/CCSE v2 共有的 9,523 字，其 `strokes` 和 `medians` 均与官方原始记录逐字完全相同，属于共享上游几何数据；官方独有 51 字，其中只有“鎩”出现在当前书法集，涉及 19 张图。比对见 `source_comparison.json`。

[联合候选清单](../../../datasets/processed/calligraphy-template-candidates-v2/manifest.jsonl) 每张书法图列出可用的 Lee-xeo/CCSE、官方 Make Me A Hanzi 和 RHSEDB 候选，并分别保留类别体系、来源和配准状态。官方图形没有笔画类别，不能直接作为 StrokeExtraction-24 参考；同字候选仍需配准及人工审核。官方 `graphics.txt` 的许可说明在 `COPYING`：图形数据适用 Arphic Public License；使用与再分发时应保留并核查相应条款。

## 10. RHSEDB 标注笔画 → 六维轨迹实验数据集（IoU 0.70）

此流程直接用 RHSEDB 的 `target_single_image`，无需先跑 StrokeExtraction。它与本说明前面的严格 0.95 人工审核封装相互独立；输出 schema 为 `rhsedb-6d-exploratory-v1`，没有伪造 writer ID 或防泄漏 split。

```bash
cd /home/lkh/workspace
PY=.venvs/virtual-sketching/bin/python
PIPE=CHTR/reproduction/pipeline
RUN=CHTR/reproduction/pipeline_runs/rhsedb_6d_10_20260917
$PY $PIPE/import_rhsedb_targets.py --rhsedb CHTR/third_party/StrokeExtraction/dataset/RHSEDB --split test --limit 10 --output "$RUN/inputs"
$PY $PIPE/vectorize_strokes_batch.py --manifest "$RUN/inputs/manifest.jsonl" --output "$RUN/vs" --samples 3 --seed 1613
$PY $PIPE/render_and_filter.py --manifest "$RUN/vs/manifest.jsonl" --output "$RUN/qc_070_v2" --iou 0.7 --fidelity 0.7
$PY $PIPE/pack_rhsedb_exploratory.py --manifest "$RUN/qc_070_v2/best.jsonl" --output datasets/processed/rhsedb-6d-iou070-new --iou 0.7
$PY $PIPE/validate_rhsedb_exploratory.py --manifest datasets/processed/rhsedb-6d-iou070-new/manifest.jsonl --report datasets/processed/rhsedb-6d-iou070-new/validation.json
```

实际运行又对“框”“腕”追加各 5 个候选并做整字组合选择；最终权威产物为 [rhsedb-6d-iou070-10-v3](../../../datasets/processed/rhsedb-6d-iou070-10-v3/README.md)：10 字输入、8 字通过、2 字整字 IoU 未达 0.70，83 笔入库。首轮与中间版本 `-v1/-v2` 仅供对比。`render_and_filter.py` 仍要求独立原生重放 IoU ≥ 0.999；封装与验证会重新计算所有逐笔、整字 IoU。这个实验集不满足原 `pack_dataset.py`/`validate_dataset.py` 的 0.95、人工审核和 writer/style split 要求。

## 11. MMAH 微调模型、mask 清理与书法六维实验（2026-09-22）

### 模型与参考库

MMAH 有序笔画已通过显式 CCSE-25 → StrokeExtraction-24 名称映射生成实验参考 bank：

```text
CHTR/reproduction/pipeline_runs/mmah_reference_bank_20260918/manifest.jsonl
```

微调权重：

```text
CHTR/third_party/StrokeExtraction/out/SDNet_MMAH_CCSEDB_v4/model/sdnet_model.pth
CHTR/third_party/StrokeExtraction/out/SegNet_MMAH_CCSEDB_v4/model/model.pth
CHTR/third_party/StrokeExtraction/out/ExtractNet_MMAH_CCSEDB_v4/model/model_extract.pth
```

混合微调数据 `CHTR/third_party/StrokeExtraction/dataset/MMAH_CCSEDB_v4/` 含 18,080 条训练记录：30% RHSEDB、20% 原始 MMAH→CCSE Kaiti、50% MMAH→增强 CCSE Kaiti。二值字段统一为 `bool`；原始 CCSE segmentation 中存在大量空白或残缺实例，这一点会限制真实楷体监督质量，详见该数据集 README。

### 40 张书法评估

运行目录：

```text
CHTR/reproduction/pipeline_runs/mmah_finetuned_eval_gpu3_20260922
```

测试覆盖 1～20 笔字符，每字 2 张，共 40 张。StrokeExtraction 完整输出 29 张，11 张因至少一个空 mask 失败。29 张成功样本的整字 mask 并集 IoU 平均 0.920、中位数 0.934；这些并集指标不能替代逐笔归属、连通性与顺序审核。

主要入口：

```text
report.json                         提取汇总
mask_overlays/index.html            彩色逐笔 mask 叠加 target
stage_diagnosis/                    SDNet/SegNet/ExtractNet 阶段对照
mask_review/index.html              原始逐笔 mask
```

### mask 清理 v3 constrained

推荐清理产物：

```text
mask_cleanup_v3_constrained/index.html
mask_cleanup_v3_constrained/report.json
mask_cleanup_v3_constrained/manifest.jsonl
```

`clean_stroke_masks.py` 当前执行：target 邻域约束、极小块清除、边缘毛刺剪除、其他笔画冗余块删除、target 与 SDNet参考共同支持的短间隙连接、受约束 5×5 闭运算和 3×3 开运算、≤256 像素内部孔洞填充，以及最终每笔只保留最大连通块。红色表示删除像素，绿色表示新增像素。29 张均处理成功，22 张清理后整字 IoU≥0.87且为 `clean_candidate`；清理目标是逐笔语义与单连通，不保证整字 IoU上升。

正式审核应从清理 manifest 开始：

```bash
PY=/home/lkh/workspace/CHTR/third_party/StrokeExtraction/.venv/bin/python
PIPE=/home/lkh/workspace/CHTR/reproduction/pipeline
RUN=/home/lkh/workspace/CHTR/reproduction/pipeline_runs/mmah_finetuned_eval_gpu3_20260922
$PY "$PIPE/review_strokes.py" \
  --manifest "$RUN/mask_cleanup_v3_constrained/manifest.jsonl" \
  --output "$RUN/mask_review_cleaned_v3"
```

### 探索性 VS 与六维结果

22 张、167 笔按 `clean_candidate && whole IoU≥0.87` 自动进入探索性 VS；此选择明确标记为 `auto_selected_exploratory`，不声明人工审核：

```text
vs_inputs_v3/       167 个白底黑色逐笔 PNG
vs_candidates_v3/   每笔 3 个原生 VS NPZ，共 501 个，无运行错误
vs_qc_v3/           每候选解码的 trajectory.npz 和重绘 QC
vs_6d_v3/           完整字符六维 sample.npz 与整字可视化
```

501 个候选中 362 个通过逐笔 IoU/上游一致性 0.87；每笔取最佳后 135/167 笔通过。22 张中 10 张所有笔和整字重绘均通过 0.87，整字 IoU 平均 0.897、中位数 0.900。最终 NPZ：

```text
vs_6d_v3/<sample_id>/sample.npz
```

包含 `points[T,6]`、`subpath_offsets`、`stroke_offsets`、`stroke_types` 和 `image_size`。六维为 `[x, y, width, continue, stroke_end, drawing_end]`。10 份 NPZ 均已独立验证结构、状态和重绘 IoU。

### VS 多子路径问题

高栅格 IoU 不代表真实一次书写。167 笔最佳候选中，159 笔已经只有 1 个 round，但解码 subpath 中位数仍为 9（范围 1～25），只有 6 笔为单 subpath。这说明主要问题是同一 round 内模型多次预测 pen-up，以及较窄圆笔头为填满宽书法 mask 而反复描画；单纯把最大 round 数设为 1 不能解决。

轮廓不应直接送 VS，因为两侧边界会诱导双线。下一阶段应比较：

1. VS 单 round、较短步数、较宽笔头、MMAH median/骨架端点起笔，以及受 mask 骨架约束的 subpath 连接；
2. 直接使用“mask 中轴骨架 + MMAH方向 + 距离变换宽度”构建六维轨迹；
3. 将骨架膨胀为约 6 像素细线送 VS，只取中心轨迹，再从原 mask 距离变换恢复宽度。

### VS 可视化工具

以下工具读取原生 `candidate.npz`，输出到 NPZ 旁同名目录：

```bash
cd /home/lkh/workspace/CHTR/third_party/virtual_sketching
VS_PY=/home/lkh/workspace/.venvs/virtual-sketching/bin/python
$VS_PY tools/visualize_drawing.py --file /absolute/path/to/candidate.npz
$VS_PY tools/gif_making.py --file /absolute/path/to/candidate.npz
$VS_PY tools/svg_conversion.py --file /absolute/path/to/candidate.npz --svg_type single
```

`visualize_drawing.py`输出最终重绘、按 primitive 着色的重叠/去重叠图和包含移动段的顺序图；`gif_making.py`生成带 cursor 与局部窗口框的逐步 GIF；`svg_conversion.py`导出绝对坐标 SVG，适合浏览器检查和路径编辑。

详细续接说明见工作区根目录 [HANDOFF_VS_SINGLE_STROKE_2026-09-22.md](../../../HANDOFF_VS_SINGLE_STROKE_2026-09-22.md)，后续任务见 [WORKLIST_VS_SINGLE_STROKE_2026-09-22.md](../../../WORKLIST_VS_SINGLE_STROKE_2026-09-22.md)。
