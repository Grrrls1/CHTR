"""Create a compact, dataloader-friendly manifest with stage IoU metrics."""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import arrays, digest, output_dir, read_json, records, require, write_json, write_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-package', required=True)
    parser.add_argument('--strokes-manifest', required=True)
    parser.add_argument('--characters-manifest', required=True)
    parser.add_argument('--cleanup-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quality-threshold', type=float, default=.87)
    parser.add_argument('--trajectory-style', default='MMAH宋体')
    args = parser.parse_args()
    require(0 < args.quality_threshold <= 1, 'invalid quality threshold')
    source_package = Path(args.source_package).resolve()
    stroke_rows = defaultdict(list)
    for row in records(args.strokes_manifest):
        stroke_rows[row['sample_id']].append(row)
    characters = records(args.characters_manifest)
    cleanup = read_json(args.cleanup_report)
    cleanup_by_id = {row['sample_id']: row for row in cleanup['samples']}
    out = output_dir(args.output)
    simplified = []
    type_map_source = source_package / 'stroke_type_map.json'
    if type_map_source.is_file():
        shutil.copy2(type_map_source, out / 'stroke_type_map.json')
    for character in sorted(characters, key=lambda row: row['sample_id']):
        sample_id = character['sample_id']
        rows = sorted(stroke_rows[sample_id], key=lambda row: row['stroke_index'])
        require(rows, 'missing strokes: ' + sample_id)
        source_sample = source_package / sample_id
        require(source_sample.is_dir(), 'missing source package sample: ' + sample_id)
        destination = out / 'samples' / sample_id
        shutil.copytree(source_sample, destination)
        source_metadata = read_json(destination / 'metadata.json') if (destination / 'metadata.json').is_file() else {}
        source_info = source_metadata.get('source', {})
        with np.load(destination / 'trajectory.npz', allow_pickle=False) as z:
            points = z['points']; stroke_offsets = z['stroke_offsets']; labels = z['stroke_types']
            require(len(labels) == len(rows), 'stroke type count mismatch: ' + sample_id)
            point_ranges = [[int(a), int(b)] for a, b in zip(stroke_offsets[:-1], stroke_offsets[1:])]
        per_iou = [float(row['qc']['iou']) for row in rows]
        whole_iou = float(character['whole_iou'])
        cleanup_row = cleanup_by_id[sample_id]
        type_names = []
        for row in rows:
            metadata = read_json(destination / 'metadata.json') if (destination / 'metadata.json').is_file() else {}
            mapped = metadata.get('stroke_map', [])[int(row['stroke_index'])] if metadata.get('stroke_map') else {}
            type_names.append(mapped.get('stroke_type_name'))
        item = dict(
            schema='chtr-simplified-manifest-v1',
            sample_id=sample_id,
            character=character['character'],
            style='trajectory_style',
            trajectory_style=args.trajectory_style,
            original_style=None,
            author=None,
            manual_review=False,
            review_status='not_performed',
            human_review_performed=False,
            image_path=str(Path('samples') / sample_id / 'image_256.png'),
            original_image_path=str(Path('samples') / sample_id / 'original_calligraphy.jpg'),
            trajectory_path=str(Path('samples') / sample_id / 'trajectory.npz'),
            image_sha256=digest(destination / 'image_256.png'),
            original_image_sha256=digest(destination / 'original_calligraphy.jpg'),
            trajectory_sha256=digest(destination / 'trajectory.npz'),
            stroke_count=len(rows),
            stroke_types=[int(v) for v in labels.tolist()],
            stroke_type_names=type_names,
            stage_iou=dict(
                mask_extraction_whole=float(cleanup_row['before']['iou']),
                mask_cleanup_whole=float(cleanup_row['after']['iou']),
                trajectory_stroke_min=float(min(per_iou)),
                trajectory_stroke_mean=float(np.mean(per_iou)),
                trajectory_stroke_median=float(np.median(per_iou)),
                trajectory_stroke_max=float(max(per_iou)),
                trajectory_whole=float(whole_iou),
                upstream_fidelity=None,
                native_replay=None,
            ),
            quality=dict(
                score_name='stage_iou.trajectory_whole', score=whole_iou,
                threshold=args.quality_threshold, passed=whole_iou >= args.quality_threshold,
                all_strokes_passed=min(per_iou) >= args.quality_threshold,
                filter_in_dataloader=True,
            ),
            stroke_map=[dict(stroke_index=int(row['stroke_index']), stroke_id=row['stroke_id'],
                             stroke_label=int(row['stroke_label']),
                             stroke_type_name=type_names[int(row['stroke_index'])],
                             point_range=point_ranges[int(row['stroke_index'])],
                             stroke_iou=float(row['qc']['iou']),
                             stroke_quality_passed=float(row['qc']['iou']) >= args.quality_threshold,
                             mask_path=str(Path('samples') / sample_id / 'stroke_masks' /
                                           ('stroke%03d.png' % row['stroke_index'])))
                        for row in rows],
            provenance=dict(source=source_info.get('source', character.get('source')),
                            provenance_status=source_info.get('provenance_status', character.get('provenance_status')),
                            source_image_path=source_info.get('source_image_path', character.get('source_image_path'))),
            trajectory_fields=['points[T,6]', 'subpath_offsets', 'stroke_offsets', 'stroke_types'],
            quality_notes='Retained for dataloader-side filtering; no manual review performed.',
        )
        simplified.append(item)
    write_records(out / 'manifest.jsonl', simplified)
    write_json(out / 'summary.json', dict(schema='chtr-simplified-manifest-v1', samples=len(simplified),
        total_strokes=sum(row['stroke_count'] for row in simplified),
        quality_threshold=args.quality_threshold, filter_field='quality.score',
        passed=sum(row['quality']['passed'] for row in simplified),
        all_strokes_passed=sum(row['quality']['all_strokes_passed'] for row in simplified),
        human_review_performed=False, style='trajectory_style', trajectory_style=args.trajectory_style,
        note='All retained samples remain exploratory; dataloader should apply its own minimum IoU.'))
    readme = '# Simplified CHTR manifest\n\n'
    readme += '字段 `style` 固定为 `trajectory_style`，实际轨迹参考风格写在 `trajectory_style`；原图作者、原图风格和人工审核均未知/未执行。\n\n'
    readme += '各阶段 IoU 位于 `stage_iou`：mask extraction、mask cleanup、逐笔 trajectory、整字 trajectory。\n'
    readme += '模型 dataloader 建议按 `quality.score >= min_iou` 筛选；当前 manifest 保留所有样本，不在打包阶段删除低分样本。\n'
    (out / 'README.md').write_text(readme, encoding='utf-8')
    print('Simplified manifest: %d samples, %d strokes' %
          (len(simplified), sum(row['stroke_count'] for row in simplified)))


if __name__ == '__main__':
    main()
