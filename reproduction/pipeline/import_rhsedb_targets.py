"""Export annotated RHSEDB target strokes directly as VS inputs."""
import argparse
from pathlib import Path

import numpy as np

from common import SCHEMA, digest, iou, masks_and_labels, output_dir, png, require, write_json, write_records


def export_record(path, split, out):
    with np.load(path, allow_pickle=False) as z:
        character = str(z['name'])
        target = z['target_image'].astype(np.uint8)
        masks = z['target_single_image'].astype(np.uint8)
        labels = z['stroke_label'].astype(np.int64)
    require(len(character) == 1 and target.shape == (1, 256, 256), 'invalid RHSEDB target: ' + str(path))
    masks_and_labels(masks, labels)
    require(np.isin(target, [0, 1]).all() and target.any(), 'invalid/empty target image')
    sample_id = 'rhsedb_%s_%06d' % (split, int(path.stem))
    directory = out / sample_id
    directory.mkdir()
    image_path = png(directory / 'image.png', (1 - target[0]) * 255)
    source_hash = digest(path)
    rows = []
    for i, mask in enumerate(masks):
        stroke_id = sample_id + '_stroke%03d' % i
        input_png = png(directory / (stroke_id + '.png'), (1 - mask) * 255)
        rows.append(dict(schema=SCHEMA, sample_id=sample_id, stroke_id=stroke_id,
                         character=character, source='RHSEDB', source_split=split,
                         source_npz_path=str(path.resolve()), source_npz_sha256=source_hash,
                         writer_status='unknown_in_local_npz', split_status='not_assigned',
                         image_path=image_path, stroke_index=i, expected_strokes=len(masks),
                         stroke_label=int(labels[i]), category_system='StrokeExtraction-24',
                         order_source='RHSEDB target_single_image index',
                         mask_source='RHSEDB annotated target_single_image',
                         input_png=input_png, input_sha256=digest(input_png)))
    return rows, dict(sample_id=sample_id, character=character, strokes=len(masks),
                      annotated_mask_union_iou=iou(np.any(masks, axis=0), target[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rhsedb', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=('train', 'test'), default='test')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--ids', nargs='*', type=int, help='Explicit numeric NPZ IDs; overrides limit')
    args = p.parse_args()
    require(args.limit > 0, 'limit must be positive')
    directory = Path(args.rhsedb).resolve() / args.split
    if args.ids:
        require(len(set(args.ids)) == len(args.ids) and min(args.ids) >= 0, 'invalid/duplicate IDs')
        paths = [directory / ('%d.npz' % i) for i in args.ids]
    else:
        paths = sorted(directory.glob('*.npz'), key=lambda path: int(path.stem))[:args.limit]
    require(paths and all(path.is_file() for path in paths), 'missing RHSEDB files')
    out = output_dir(args.output)
    rows, images, errors = [], [], []
    for path in paths:
        try:
            new_rows, image = export_record(path, args.split, out)
            rows.extend(new_rows)
            images.append(image)
        except Exception as exc:
            errors.append(dict(source_npz_path=str(path), error=str(exc)))
    write_records(out / 'manifest.jsonl', rows)
    write_records(out / 'images.jsonl', images)
    write_records(out / 'errors.jsonl', errors)
    write_json(out / 'summary.json', dict(requested_images=len(paths), exported_images=len(images),
                                         strokes=len(rows), errors=len(errors), source_split=args.split,
                                         note='Annotated RHSEDB target strokes; writer identity unavailable in local NPZ.'))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
