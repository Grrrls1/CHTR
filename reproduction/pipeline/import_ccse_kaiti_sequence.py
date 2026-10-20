"""Build ordered CCSE-25 masks using Lee-xeo fort_graphics/Make Me A Hanzi paths."""
import argparse
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cairosvg
import cv2
import numpy as np
from PIL import Image

from common import SCHEMA, digest, identifier, output_dir, require, write_json, write_records


def rasterize(record, output):
    character = record['character']
    strokes = record['strokes']
    raw_labels = np.asarray(record['category_id'], dtype=np.int64)
    require(len(character) == 1 and strokes and len(raw_labels) <= len(strokes), 'invalid sequence record')
    require(np.all((raw_labels >= 1) & (raw_labels <= 25)), 'CCSE category must be 1..25')
    category_status = 'complete' if len(raw_labels) == len(strokes) else 'unknown_due_to_upstream_missing_annotations'
    # When annotations are incomplete, their alignment to individual paths is
    # not documented. Mark the entire character unknown instead of shifting
    # labels after the first missing stroke.
    labels = raw_labels if category_status == 'complete' else np.full(len(strokes), -1, dtype=np.int64)
    count = len(strokes)
    groups = []
    for index, path_data in enumerate(strokes):
        groups.append('<g transform="translate(%d,0) scale(1,-1) translate(0,-900)"><path d="%s" fill="black"/></g>' %
                      (index * 1024, path_data.replace('&', '&amp;').replace('"', '&quot;')))
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="1024" viewBox="0 0 %d 1024">%s</svg>' %
           (count * 1024, count * 1024, ''.join(groups))).encode()
    rendered = cairosvg.svg2png(bytestring=svg, output_width=count * 256, output_height=256)
    rgba = np.asarray(Image.open(io.BytesIO(rendered)).convert('RGBA'))
    masks = np.stack([(rgba[:, i * 256:(i + 1) * 256, 3] > 0).astype(np.uint8) for i in range(count)])
    require(np.all(masks.reshape(count, -1).sum(1) > 0), 'empty rendered stroke')
    medians = record.get('medians', [])
    require(len(medians) == count, 'median/stroke count mismatch')
    # Make Me A Hanzi coordinates use y-up after the upstream SVG transform.
    median_arrays = [np.asarray([[x / 4, (900 - y) / 4] for x, y in median], np.float32) for median in medians]
    sample_id = identifier('ccsek_u%05X' % ord(character))
    destination = output / (sample_id + '.npz')
    payload = dict(
        character=np.array(character), image_id=np.array(str(record['image_id'])),
        stroke_label=labels, ordered_stroke_masks=masks,
        stroke_order=np.arange(count, dtype=np.int64), category_system=np.array('CCSE-25'),
        raw_partial_stroke_label=raw_labels, category_status=np.array(category_status),
        source=np.array('Lee-xeo fort_graphics + Make Me A Hanzi'))
    for index, median in enumerate(median_arrays):
        payload['median_%03d' % index] = median
    np.savez_compressed(str(destination), **payload)
    combined = (1 - np.any(masks, axis=0).astype(np.uint8)) * 255
    preview = output / (sample_id + '.png')
    require(cv2.imwrite(str(preview), combined), 'cannot write preview')
    return dict(
        schema=SCHEMA, sample_id=sample_id, character=character,
        ordered_strokes_path=str(destination.resolve()), preview_path=str(preview.resolve()),
        image_id=str(record['image_id']), stroke_count=count,
        stroke_labels=labels.tolist(), raw_partial_stroke_labels=raw_labels.tolist(),
        category_status=category_status, category_system='CCSE-25', order_source='Make Me A Hanzi',
        source='Lee-xeo/Chinese-Character-Stroke-Sequence-Dataset',
        ordered_strokes_sha256=digest(destination), strokeextraction_24_compatible=False)



def match_ccse_records(records, images, annotations):
    """Match on Unicode character and compare unordered CCSE class counts."""
    image_by_character = {image['font']: image for image in images.values()}
    require(len(image_by_character) == len(images), 'duplicate CCSE characters')
    graphic_by_character = {r['character']: r for r in records}
    require(len(graphic_by_character) == len(records), 'duplicate sequence characters')
    require(set(graphic_by_character) == set(image_by_character), 'sequence/CCSE character sets differ')
    annotation_labels = {}
    for annotation in annotations:
        annotation_labels.setdefault(annotation['image_id'], []).append(annotation['category_id'])
    mismatches = []
    for character, record in graphic_by_character.items():
        image_id = image_by_character[character]['id']
        if Counter(record['category_id']) != Counter(annotation_labels.get(image_id, [])):
            mismatches.append(dict(character=character, image_id=image_id, kind='category_multiset',
                                   sequence=record['category_id'], coco=annotation_labels.get(image_id, [])))
    return image_by_character, mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graphics', required=True)
    parser.add_argument('--ccse', required=True, help='Extracted CCSE-Kai root containing annotations/')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    require(args.workers > 0, 'workers must be positive')
    graphics_path = Path(args.graphics).resolve()
    records = json.loads(graphics_path.read_text(encoding='utf-8'))
    require(len(records) == 9523 and len({r['character'] for r in records}) == len(records), 'unexpected sequence data')
    ccse = Path(args.ccse).resolve()
    images, annotations, categories = {}, [], None
    for split in ('train', 'val', 'test'):
        annotation_path = ccse / 'annotations' / ('instances_%s2021.json' % split)
        data = json.loads(annotation_path.read_text(encoding='utf-8'))
        categories = categories or data['categories']
        for image in data['images']:
            require(image['id'] not in images, 'duplicate COCO image ID')
            images[image['id']] = dict(image, split=split)
        annotations.extend(data['annotations'])
    require(len(images) == 9523 and len(annotations) == 112024 and len(categories) == 25, 'unexpected CCSE counts')
    image_by_character, mismatches = match_ccse_records(records, images, annotations)
    output = output_dir(args.output)
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [(record, pool.submit(rasterize, record, output)) for record in records]
        for index, (record, future) in enumerate(futures, 1):
            try:
                row = future.result()
                ccse_image = image_by_character[record['character']]
                row['ccse_image_id'] = ccse_image['id']
                row['ccse_image_path'] = str((ccse / (ccse_image['split'] + '2021') / ccse_image['file_name']).resolve())
                row['ccse_split'] = ccse_image['split']
                rows.append(row)
            except Exception as exc:
                errors.append(dict(character=record.get('character'), image_id=record.get('image_id'), error=str(exc)))
            if index % 500 == 0:
                print('processed %d/%d' % (index, len(records)), flush=True)
    rows.sort(key=lambda row: int(row['image_id']))
    write_records(output / 'manifest.jsonl', rows)
    write_records(output / 'errors.jsonl', errors)
    write_json(output / 'category_map.json', dict(
        category_system='CCSE-25', categories=categories,
        note='No verified mapping to StrokeExtraction internal 24-class IDs is supplied.'))
    write_json(output / 'alignment_report.json', dict(
        sequence_records=len(records), ccse_images=len(images), ccse_annotations=len(annotations),
        generated=len(rows), errors=len(errors), metadata_mismatches=len(mismatches),
        mismatches=mismatches[:100], graphics_sha256=digest(graphics_path),
        ccse_archive_sha256='c49f76249a463c840c3c28ed363e3cdcfa419c5d2a82dc6963f4c0295fdf6eaa'))
    if errors or mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
