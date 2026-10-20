"""Pack a small no-human-review CHTR example dataset for inspection."""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (arrays, digest, gray, iou, output_dir, records, render,
                    require, validate_points, write_json, write_records)


LABEL_NAMES = [
    '点', '横', '竖', '撇', '提', '横折', '横撇/横钩', '横折钩',
    '竖弯钩', '横折弯钩/横斜钩', '竖折折钩', '横折折/横折弯',
    '横折提', '横折折折钩/横撇弯钩', '竖折撇/竖折折', '横折折撇',
    '捺', '斜钩', '弯钩', '竖钩', '竖折/竖弯', '撇折', '撇点', '竖提',
]
SEGMENT_GROUPS = [0, 1, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                  6, 6, 6, 2, 4, 4, 4, 2]


README_TEMPLATE = '''# CHTR exploratory dataset example

本目录是一个用于检查格式和对应关系的 exploratory 示例，不是正式审核通过的数据集。

- 样本：{samples} 个汉字，{strokes} 笔。
- IoU 标记阈值：{threshold:.2f}。
- IoU 阈值只作为质量字段写入；默认保留全部样本，由 dataloader 负责筛选。
- 人工审核：跳过；`review_status=exploratory_no_human_review`。
- 轨迹参考风格：`{style_id}` / `{style_name}`。
- 原书法图的真实书家/书体：当前未知，不能将参考风格解释为原图作者风格。

## 每个样本目录

- `original_calligraphy.jpg`：原始书法图像。
- `image_256.png`：流水线使用的 256×256 规范化图像。
- `trajectory.npz`：整字六维轨迹，字段包括 `points[T,6]`、`subpath_offsets`、`stroke_offsets`、`stroke_types`。
- `stroke_masks/strokeXXX.png`：对应的白底黑单笔画输入。
- `metadata.json`：图像、汉字、风格、笔画区间、IoU 和源 hash 的对应说明。

## 对应关系

对某个样本，`metadata.json` 的 `character` 是汉字；`stroke_map[i]` 对应语义第 `i` 笔，
其 `point_range` 是 `trajectory.npz` 中的点区间，`mask_path` 是该笔输入 mask，
`stroke_iou` 是该笔轨迹渲染与 mask 的 IoU。整字图像对应同一 metadata 的 `image_256`，
轨迹坐标为 256×256 画布坐标，x 向右、y 向下，第三列 width 是像素直径。

`stroke_types` 在 NPZ 中仍保存为 0..23 的整数；名称请通过顶层 `stroke_type_map.json` 解释。
`centerline.png` 等可视化仍属于辅助审查资产；训练/读取轨迹应使用 `trajectory.npz`。
'''


def pack_one(sample, stroke_rows, output, threshold, style_id, style_name):
    sample_id = sample['sample_id']
    rows = sorted(stroke_rows, key=lambda row: row['stroke_index'])
    expected = int(sample.get('expected_strokes', sample.get('stroke_count')))
    require(len(rows) == expected and [r['stroke_index'] for r in rows] == list(range(expected)),
            'Incomplete stroke set: ' + sample_id)
    first_stroke = rows[0]
    char_source = Path(sample.get('source_image_path', first_stroke['source_image_path'])).resolve()
    image_source = Path(sample.get('image_path', first_stroke['image_path'])).resolve()
    require(char_source.is_file() and image_source.is_file(), 'Missing source image: ' + sample_id)
    trajectory_source = Path(sample['_trajectory_path']).resolve()
    require(trajectory_source.is_file(), 'Missing trajectory: ' + sample_id)
    z = arrays(trajectory_source)
    validate_points(z['points'], z['subpath_offsets'], bounds=False)
    require('stroke_offsets' in z and len(z['stroke_offsets']) == expected + 1,
            'Missing stroke offsets: ' + sample_id)
    target = gray(image_source) < 128
    rendered = render(z['points'], z['subpath_offsets'])
    whole_iou = iou(rendered >= 128, target)
    directory = output / sample_id
    directory.mkdir()
    original_path = directory / ('original_calligraphy' + char_source.suffix)
    image_path = directory / 'image_256.png'
    trajectory_path = directory / 'trajectory.npz'
    shutil.copy2(char_source, original_path)
    shutil.copy2(image_source, image_path)
    shutil.copy2(trajectory_source, trajectory_path)
    masks_dir = directory / 'stroke_masks'
    masks_dir.mkdir()
    stroke_map = []
    below = []
    for index, row in enumerate(rows):
        mask_source = Path(row['input_png']).resolve()
        require(mask_source.is_file(), 'Missing stroke mask: ' + row['stroke_id'])
        mask_path = masks_dir / ('stroke%03d.png' % index)
        shutil.copy2(mask_source, mask_path)
        score = float(row['qc']['iou'])
        status = 'passed_iou_threshold' if score >= threshold else 'below_iou_threshold'
        if status != 'passed_iou_threshold':
            below.append(dict(stroke_index=index, stroke_id=row['stroke_id'], iou=score))
        lo, hi = [int(v) for v in z['stroke_offsets'][index:index + 2]]
        stroke_label = int(row['stroke_label'])
        stroke_map.append(dict(stroke_index=index, stroke_id=row['stroke_id'],
                               stroke_label=stroke_label,
                               stroke_type_name=LABEL_NAMES[stroke_label],
                               stroke_type_source='StrokeExtraction-24/RHSEDB name mapping',
                               point_range=[lo, hi], mask_path=str(mask_path.relative_to(directory)),
                               mask_sha256=digest(mask_path), stroke_iou=score,
                               status=status, trajectory_sha256=digest(trajectory_path)))
    sample_status = 'passed_iou_threshold' if whole_iou >= threshold and not below else 'included_with_quality_warnings'
    metadata = dict(schema='chtr-exploratory-example-v1', sample_id=sample_id,
                    character=sample['character'], character_name=sample['character'],
                    style_id=style_id, style_name=style_name,
                    style_role='trajectory_reference_style',
                    original_image_style='unknown_calligraphy_style',
                    review_status='exploratory_no_human_review', human_review_performed=False,
                    quality_threshold=threshold, sample_status=sample_status,
                    whole_iou=whole_iou, below_threshold_strokes=below,
                    source=dict(original_calligraphy=str(original_path.name),
                               original_calligraphy_sha256=digest(original_path),
                               image_256=str(image_path.name), image_256_sha256=digest(image_path),
                               source_image_path=str(char_source), source_sha256=digest(char_source),
                               source=sample.get('source', first_stroke.get('source')),
                               provenance_status=sample.get('provenance_status', first_stroke.get('provenance_status'))),
                    trajectory=dict(path=str(trajectory_path.name), sha256=digest(trajectory_path),
                                    fields=['points[T,6]', 'subpath_offsets', 'stroke_offsets', 'stroke_types'],
                                    coordinates='x right, y down; 256x256 canvas',
                                    width='pixel diameter', state_semantics='action after arrival'),
                    stroke_count=expected, stroke_map=stroke_map,
                    stroke_type_map_path='../stroke_type_map.json')
    write_json(directory / 'metadata.json', metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strokes-manifest', required=True)
    parser.add_argument('--characters-manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou', type=float, default=.80)
    parser.add_argument('--style-id', default='mmah-songti')
    parser.add_argument('--style-name', default='MMAH宋体')
    parser.add_argument('--filter-whole-iou', action='store_true',
                        help='Optional hard filter; omitted by default so all samples are retained')
    args = parser.parse_args()
    require(0 < args.iou <= 1, 'invalid IoU threshold')
    stroke_rows = defaultdict(list)
    for row in records(args.strokes_manifest):
        stroke_rows[row['sample_id']].append(row)
    characters = records(args.characters_manifest)
    require(characters, 'empty character manifest')
    require(len({row['sample_id'] for row in characters}) == len(characters),
            'duplicate character sample_id')
    available = set(stroke_rows)
    require({row['sample_id'] for row in characters} <= available,
            'character manifest is not covered by stroke manifest')
    out = output_dir(args.output)
    packed = []
    rejected = []
    for character in sorted(characters, key=lambda row: row['sample_id']):
        whole_iou = float(character['whole_iou'])
        if args.filter_whole_iou and whole_iou < args.iou:
            rejected.append(dict(sample_id=character['sample_id'], character=character['character'],
                                 whole_iou=whole_iou, threshold=args.iou,
                                 reason='whole_character_render_iou_below_threshold'))
            continue
        item = dict(character, _trajectory_path=character['npz_path'])
        packed.append(pack_one(item, stroke_rows[character['sample_id']], out,
                               args.iou, args.style_id, args.style_name))
    for item in packed:
        item.pop('_trajectory_path', None)
    type_map = dict(schema='chtr-stroke-type-map-v1', category_system='StrokeExtraction-24',
                    source='StrokeExtraction/utils.py LABEL_NAMES and RHSEDB stroke_name',
                    types=[dict(id=i, name_zh=name, segnet_group=SEGMENT_GROUPS[i])
                           for i, name in enumerate(LABEL_NAMES)])
    write_json(out / 'stroke_type_map.json', type_map)
    write_records(out / 'manifest.jsonl', packed)
    if args.filter_whole_iou:
        write_records(out / 'rejected.jsonl', rejected)
    warning_strokes = sum(len(item['below_threshold_strokes']) for item in packed)
    write_json(out / 'summary.json', dict(schema='chtr-exploratory-example-v1',
        samples=len(packed), total_strokes=sum(item['stroke_count'] for item in packed),
        threshold=args.iou, style_id=args.style_id, style_name=args.style_name,
        input_characters=len(characters), rejected_characters=len(rejected),
        human_review_performed=False, samples_with_quality_warnings=sum(
            item['sample_status'] != 'passed_iou_threshold' for item in packed),
        below_threshold_strokes=warning_strokes,
        filter_applied=args.filter_whole_iou,
        note='Example dataset only; all samples are retained by default and dataloader-side IoU filtering is expected.'))
    readme = README_TEMPLATE.format(samples=len(packed),
        strokes=sum(item['stroke_count'] for item in packed), threshold=args.iou,
        style_id=args.style_id, style_name=args.style_name)
    (out / 'README.md').write_text(readme, encoding='utf-8')
    print('Packed exploratory example: %d characters, %d strokes, %d below threshold' %
          (len(packed), sum(item['stroke_count'] for item in packed), warning_strokes))


if __name__ == '__main__':
    main()
